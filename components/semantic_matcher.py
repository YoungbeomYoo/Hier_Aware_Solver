"""
Semantic Matcher — Qwen3-Embedding 기반 key_elements 시맨틱 매칭

핵심 아이디어:
- Question key_elements와 Memory node key_elements를 embedding으로 비교
- 각 q_element vs 각 node_element의 cosine similarity → 노드별 총합 점수
- Top-K 노드 선택 (threshold 대신 상대 순위 → "다 켜지는" 문제 방지)

Memory 빌더에서 key_elements가 합집합(union)으로 상위 레벨에 모이므로,
Level_1 노드의 key_elements와 비교하면 leaf-level 정밀 매칭 가능.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


# key_elements 카테고리 (메모리 빌더와 동일)
KE_CATEGORIES = ["actions", "objects", "persons", "attributes", "locations", "text_ocr"]


def flatten_key_elements(ke: dict, categories: list[str] | None = None) -> list[str]:
    """key_elements dict를 flat list로 변환.

    Args:
        ke: {"actions": [...], "objects": [...], ...}
        categories: 포함할 카테고리 (None이면 전부)

    Returns:
        ["cut onion", "spatula", "woman in blue apron", ...]
    """
    cats = categories or KE_CATEGORIES
    texts = []
    for cat in cats:
        for val in ke.get(cat, []):
            text = str(val).strip()
            if text:
                texts.append(text)
    return texts


class SemanticMatcher:
    """Embedding 기반 key_elements 시맨틱 매칭.

    Qwen3-Embedding-0.6B를 로드하여 question key_elements와
    memory node key_elements의 유사도를 계산.

    Args:
        model_path: Qwen3-Embedding 모델 경로
        device: torch device (None → auto)
        top_k: 선택할 최대 노드 수
        batch_size: embedding 배치 크기
        score_mode: "sum" (q_element별 max sim 합산) 또는 "avg" (평균)
    """

    def __init__(
        self,
        model_path: str = "/scratch2/youngbeom/ckpt/Qwen3-Embedding-0.6B",
        device: str | None = None,
        top_k: int = 30,
        batch_size: int = 64,
        score_mode: str = "sum",
        use_key_elements: bool = True,
    ):
        self.model_path = model_path
        self.top_k = top_k
        self.batch_size = batch_size
        self.score_mode = score_mode
        self.use_key_elements = use_key_elements
        self._model = None
        self._tokenizer = None

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _load_model(self):
        """Lazy loading — 처음 사용 시에만 모델 로드."""
        if self._model is not None:
            return

        print(f"[SemanticMatcher] Loading {self.model_path} ...")
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(self._device)
        self._model.eval()
        print(f"[SemanticMatcher] Loaded on {self._device}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트를 normalized embedding으로 변환.

        Returns:
            np.ndarray shape (N, hidden_dim), L2-normalized
        """
        self._load_model()

        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)

        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self._device)

            outputs = self._model(**inputs)
            # Last hidden state의 [CLS] token (first token) 또는 mean pooling
            # Qwen3-Embedding은 last_hidden_state[:, 0] 사용
            embs = outputs.last_hidden_state[:, 0]  # (batch, hidden_dim)
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
            all_embs.append(embs.cpu().float().numpy())

        return np.concatenate(all_embs, axis=0)

    def score_nodes(
        self,
        q_elements: list[str],
        nodes: list[dict],
        categories: list[str] | None = None,
    ) -> list[dict]:
        """Question elements와 노드들의 시맨틱 매칭 점수 계산.

        Args:
            q_elements: question에서 추출한 key elements (flat list)
            nodes: [{"key_elements": {...}, "summary": "...", ...}, ...]
            categories: 비교할 카테고리 (None → 전부)

        Returns:
            [{"node_idx": int, "score": float, "top_matches": [...], "n_elements": int}, ...]
            score 내림차순 정렬.
        """
        if not q_elements or not nodes:
            return []

        # 1. Question elements embedding
        q_embs = self.encode(q_elements)  # (Q, D)

        # 2. 각 노드의 elements 수집 + 일괄 embedding
        node_elements = []
        node_offsets = []  # (start_idx, end_idx) per node
        for node in nodes:
            elems = []
            if self.use_key_elements:
                ke = node.get("key_elements", {})
                elems = flatten_key_elements(ke, categories)
            # summary도 element로 추가 (노드 대표 텍스트)
            summary = node.get("summary", "").strip()
            if summary:
                elems.append(summary)
            start = len(node_elements)
            node_elements.extend(elems)
            node_offsets.append((start, len(node_elements), len(elems)))

        if not node_elements:
            return [{"node_idx": i, "score": 0.0, "top_matches": [], "n_elements": 0}
                    for i in range(len(nodes))]

        # 3. 전체 node elements 일괄 embedding
        n_embs = self.encode(node_elements)  # (N_total, D)

        # 4. 노드별 점수 계산
        results = []
        for idx, (start, end, n_elems) in enumerate(node_offsets):
            if n_elems == 0:
                results.append({
                    "node_idx": idx,
                    "score": 0.0,
                    "top_matches": [],
                    "n_elements": 0,
                })
                continue

            # q_embs (Q, D) vs node_embs (M, D) → sim_matrix (Q, M)
            node_emb_slice = n_embs[start:end]  # (M, D)
            sim_matrix = q_embs @ node_emb_slice.T  # (Q, M)

            # 각 q_element에 대해 best match 찾기
            max_sims = sim_matrix.max(axis=1)  # (Q,) — 각 q_element의 최대 유사도
            best_indices = sim_matrix.argmax(axis=1)  # (Q,)

            if self.score_mode == "sum":
                score = float(max_sims.sum())
            else:  # avg
                score = float(max_sims.mean())

            # Top matches (디버깅용)
            top_matches = []
            sorted_q_idx = np.argsort(-max_sims)[:5]
            for qi in sorted_q_idx:
                ni = best_indices[qi]
                top_matches.append({
                    "q_element": q_elements[qi],
                    "matched_to": node_elements[start + ni],
                    "similarity": round(float(max_sims[qi]), 4),
                })

            results.append({
                "node_idx": idx,
                "score": round(score, 4),
                "top_matches": top_matches,
                "n_elements": n_elems,
            })

        # Score 내림차순 정렬
        results.sort(key=lambda x: -x["score"])
        return results

    def select_top_nodes(
        self,
        q_elements: list[str],
        tree: dict,
        level: str = "Level_1",
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict:
        """Tree에서 지정 레벨의 top-K 노드 선택.

        Args:
            q_elements: question key elements (flat list)
            tree: streaming_memory_tree dict
            level: 매칭할 레벨 ("Level_1", "Level_2", ...)
            categories: 비교할 카테고리
            top_k: override top_k (None → self.top_k)

        Returns:
            {
                "selected_indices": [int, ...],
                "scores": [{"node_idx": ..., "score": ..., "top_matches": ...}, ...],
                "total_nodes": int,
                "level": str,
            }
        """
        k = top_k or self.top_k

        if level not in tree:
            return {
                "selected_indices": [],
                "scores": [],
                "total_nodes": 0,
                "level": level,
            }

        nodes = tree[level]
        scores = self.score_nodes(q_elements, nodes, categories)
        selected = scores[:k]
        selected_indices = [s["node_idx"] for s in selected]

        return {
            "selected_indices": selected_indices,
            "scores": selected,
            "total_nodes": len(nodes),
            "level": level,
        }

    def extract_question_elements(
        self,
        question: str,
        options: list[str],
        cues: list[str] | None = None,
    ) -> list[str]:
        """Question + options에서 비교용 key elements 추출.

        LLM 없이 rule-based로 추출. cues가 있으면 그것도 포함.

        Returns:
            flat list of elements for embedding
        """
        elements = []

        # 1. 기존 cues (query_analyzer가 추출한 것)
        if cues:
            elements.extend(cues)

        # 2. Question 자체 (짧게)
        q_clean = question.strip()
        if q_clean:
            elements.append(q_clean)

        # 3. Options (각 선지)
        for opt in options:
            opt_clean = opt.strip()
            if opt_clean and len(opt_clean) > 2:
                elements.append(opt_clean)

        # Deduplicate
        seen = set()
        unique = []
        for e in elements:
            e_lower = e.lower().strip()
            if e_lower not in seen:
                seen.add(e_lower)
                unique.append(e)

        return unique
