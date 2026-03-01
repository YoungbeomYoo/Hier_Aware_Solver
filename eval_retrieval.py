#!/usr/bin/env python3
"""
Retrieval Quality Evaluation — time_reference 기반

LVBench의 time_reference (정답 시간 구간)을 ground truth로 사용하여
tree filter의 검색 품질을 측정.

평가 지표:
- Recall@K: top-K priority leaves 중 time_reference와 overlap하는 leaf가 있는 비율
- Precision: priority leaves 중 time_reference와 overlap하는 비율
- Activation rate: 전체 leaf 대비 priority leaf 비율
- Mean Rank: 정답 leaf의 평균 순위

비교 대상:
1. Random baseline: 랜덤으로 K개 선택
2. Leaf-only flat: leaf의 key_elements만으로 cue 매칭 (hierarchy 무시)
3. Hierarchical flat: build() — 전 레벨 key_elements 매칭
4. Hierarchical structured: build_structured() — 카테고리 교차 매칭

Usage:
    # Rule-based cues (CPU only, GPU 불필요)
    python eval_retrieval.py --qid_list output/lvbench_mini_qid_list.tsv

    # 기존 pipeline 결과의 cue 재사용
    python eval_retrieval.py --qid_list output/lvbench_mini_qid_list.tsv \
        --cue_source output/lvbench_mini/by_qid
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.tree_filter import FilteredTreeBuilder
from components.query_analyzer import QueryAnalyzer
from components.time_router import TimeRouter


def parse_time_reference(time_ref: str) -> tuple[float, float] | None:
    """HH:MM:SS-HH:MM:SS or MM:SS-MM:SS → (start_sec, end_sec)."""
    if not time_ref:
        return None
    parts = time_ref.split("-")
    if len(parts) != 2:
        return None

    def to_sec(s):
        tokens = s.strip().split(":")
        tokens = [float(t) for t in tokens]
        if len(tokens) == 3:
            return tokens[0] * 3600 + tokens[1] * 60 + tokens[2]
        elif len(tokens) == 2:
            return tokens[0] * 60 + tokens[1]
        return tokens[0]

    try:
        return (to_sec(parts[0]), to_sec(parts[1]))
    except (ValueError, IndexError):
        return None


def overlap(a_start, a_end, b_start, b_end) -> bool:
    """두 구간이 겹치는지 확인."""
    return a_start < b_end and b_start < a_end


def overlap_iou(a_start, a_end, b_start, b_end) -> float:
    """두 구간의 IoU."""
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_start >= inter_end:
        return 0.0
    inter = inter_end - inter_start
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def load_questions(qid_list_path: str, question_path: str):
    """QID list + question JSONL → question dicts with time_reference."""
    # Load QID → video mapping
    qid_vid = {}
    with open(qid_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid_vid[parts[0]] = parts[1]

    # Load questions
    all_qa = {}
    with open(question_path) as f:
        for line in f:
            entry = json.loads(line)
            vid = entry["key"]
            for qa in entry.get("qa", []):
                uid = str(qa.get("uid", ""))
                if uid in qid_vid and qid_vid[uid] == vid:
                    # Parse question text and options
                    raw_q = qa.get("question", "")
                    match = re.search(r"\n\(A\)", raw_q)
                    if match:
                        question_text = raw_q[:match.start()].strip()
                        options_text = raw_q[match.start():]
                        options = re.findall(r"\([A-D]\)\s*[^\n]+", options_text)
                        options = [re.sub(r"^\(([A-D])\)\s*", r"\1. ", o) for o in options]
                    else:
                        question_text = raw_q
                        options = []

                    all_qa[uid] = {
                        "question_id": uid,
                        "video_id": vid,
                        "question": question_text,
                        "options": options,
                        "answer": qa.get("answer", ""),
                        "time_reference": qa.get("time_reference", ""),
                        "question_type": qa.get("question_type", []),
                    }
    return all_qa


def load_memory(mem_dir: str, video_id: str) -> dict:
    """Load memory tree for a video."""
    for suffix in ["_synced.json", ".json"]:
        path = os.path.join(mem_dir, f"{video_id}{suffix}")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def get_all_leaves(tree: dict) -> list[dict]:
    """Extract all leaves with time ranges from tree."""
    leaves = []
    for l1_node in tree.get("Level_1", []):
        for child in l1_node.get("children", []):
            if "start_time" in child:
                leaves.append({
                    "start_time": float(child["start_time"]),
                    "end_time": float(child["end_time"]),
                    "node": child,
                })
    return leaves


def eval_retrieval(
    questions: dict,
    mem_dir: str,
    methods: list[str] = None,
    k_values: list[int] = None,
):
    """Run retrieval evaluation.

    Methods:
    - random: 랜덤 K개
    - leaf_only: leaf key_elements만 매칭 (hierarchy 없음)
    - hier_flat: build() — 전 레벨 매칭
    - hier_structured: build_structured() — 카테고리 교차
    """
    if methods is None:
        methods = ["random", "leaf_only", "hier_flat", "hier_structured"]
    if k_values is None:
        k_values = [5, 10, 20]

    tree_filter = FilteredTreeBuilder(match_threshold=1, use_key_elements=True)
    analyzer = QueryAnalyzer(llm_fn=None)  # rule-based only

    results = {m: defaultdict(list) for m in methods}
    skipped = 0
    processed = 0

    # Group questions by video for efficiency
    by_video = defaultdict(list)
    for qid, qa in questions.items():
        by_video[qa["video_id"]].append(qa)

    for vid, qa_list in sorted(by_video.items()):
        mem = load_memory(mem_dir, vid)
        tree = mem.get("streaming_memory_tree", {})
        if not tree:
            skipped += len(qa_list)
            continue

        all_leaves = get_all_leaves(tree)
        n_leaves = len(all_leaves)
        if n_leaves == 0:
            skipped += len(qa_list)
            continue

        for qa in qa_list:
            gt_range = parse_time_reference(qa["time_reference"])
            if gt_range is None:
                skipped += 1
                continue

            gt_start, gt_end = gt_range
            question = qa["question"]
            options = qa["options"]
            time_ref = qa.get("time_reference", "")
            processed += 1

            # --- Analyze question (rule-based) ---
            analysis = analyzer.analyze(question, options, time_ref)
            cues = analysis["cues"]
            target_fields = analysis.get("target_fields", ["summary"])

            # Also get structured cues (rule-based fallback)
            analysis_s = analyzer.analyze_structured(question, options, time_ref)
            structured_cues = analysis_s.get("structured_cues", {})

            for method in methods:
                if method == "random":
                    # Random baseline — average over 100 trials
                    for K in k_values:
                        hits = 0
                        trials = 100
                        for _ in range(trials):
                            sampled = random.sample(all_leaves, min(K, n_leaves))
                            hit = any(
                                overlap(l["start_time"], l["end_time"], gt_start, gt_end)
                                for l in sampled
                            )
                            hits += int(hit)
                        results[method][f"recall@{K}"].append(hits / trials)

                    # Activation rate = K/n_leaves (conceptual)
                    results[method]["n_leaves"].append(n_leaves)

                elif method == "leaf_only":
                    # Match cues only against leaf-level key_elements
                    scored = []
                    for leaf in all_leaves:
                        node = leaf["node"]
                        ke = node.get("key_elements", {})
                        caption = node.get("caption", "")
                        summary = node.get("summary", "")
                        text = caption + " " + summary
                        for field in target_fields:
                            vals = ke.get(field, [])
                            if isinstance(vals, list):
                                text += " " + " ".join(str(v) for v in vals)
                            elif isinstance(vals, str):
                                text += " " + vals
                        text_lower = text.lower()

                        score = sum(1 for c in cues if c.lower() in text_lower)
                        scored.append((score, leaf))

                    scored.sort(key=lambda x: -x[0])
                    priority = [s[1] for s in scored if s[0] > 0]

                    for K in k_values:
                        top_k = priority[:K] if priority else scored[:K]
                        top_k = [s[1] if isinstance(s, tuple) else s for s in top_k[:K]]
                        hit = any(
                            overlap(l["start_time"], l["end_time"], gt_start, gt_end)
                            for l in top_k
                        )
                        results[method][f"recall@{K}"].append(int(hit))

                    results[method]["n_priority"].append(len(priority))
                    results[method]["n_leaves"].append(n_leaves)
                    if priority:
                        results[method]["precision"].append(
                            sum(1 for l in priority
                                if overlap(l["start_time"], l["end_time"], gt_start, gt_end))
                            / len(priority)
                        )
                    # Mean rank of first hit
                    for rank, (_, leaf) in enumerate(scored):
                        if overlap(leaf["start_time"], leaf["end_time"], gt_start, gt_end):
                            results[method]["first_hit_rank"].append(rank + 1)
                            break
                    else:
                        results[method]["first_hit_rank"].append(n_leaves + 1)

                elif method == "hier_flat":
                    filtered = tree_filter.build(tree, cues, target_fields)
                    priority = filtered["priority_leaves"]
                    all_l = filtered["all_leaves"]

                    for K in k_values:
                        top_k = priority[:K]
                        hit = any(
                            overlap(l["start_time"], l["end_time"], gt_start, gt_end)
                            for l in top_k
                        )
                        results[method][f"recall@{K}"].append(int(hit))

                    results[method]["n_priority"].append(len(priority))
                    results[method]["n_leaves"].append(len(all_l))
                    if priority:
                        results[method]["precision"].append(
                            sum(1 for l in priority
                                if overlap(l["start_time"], l["end_time"], gt_start, gt_end))
                            / len(priority)
                        )
                    # Mean rank
                    for rank, leaf in enumerate(priority):
                        if overlap(leaf["start_time"], leaf["end_time"], gt_start, gt_end):
                            results[method]["first_hit_rank"].append(rank + 1)
                            break
                    else:
                        results[method]["first_hit_rank"].append(len(all_l) + 1)

                elif method == "hier_structured":
                    filtered = tree_filter.build_structured(
                        tree, structured_cues, min_category_matches=1,
                    )
                    priority = filtered["priority_leaves"]
                    all_l = filtered["all_leaves"]

                    # Fallback if empty
                    if not priority:
                        filtered = tree_filter.build(tree, cues, target_fields)
                        priority = filtered["priority_leaves"]
                        all_l = filtered["all_leaves"]

                    for K in k_values:
                        top_k = priority[:K]
                        hit = any(
                            overlap(l["start_time"], l["end_time"], gt_start, gt_end)
                            for l in top_k
                        )
                        results[method][f"recall@{K}"].append(int(hit))

                    results[method]["n_priority"].append(len(priority))
                    results[method]["n_leaves"].append(len(all_l))
                    if priority:
                        results[method]["precision"].append(
                            sum(1 for l in priority
                                if overlap(l["start_time"], l["end_time"], gt_start, gt_end))
                            / len(priority)
                        )
                    for rank, leaf in enumerate(priority):
                        if overlap(leaf["start_time"], leaf["end_time"], gt_start, gt_end):
                            results[method]["first_hit_rank"].append(rank + 1)
                            break
                    else:
                        results[method]["first_hit_rank"].append(len(all_l) + 1)

    # === Aggregate ===
    print(f"\n{'='*70}")
    print(f"Retrieval Quality Evaluation — {processed} questions ({skipped} skipped)")
    print(f"{'='*70}")

    # Header
    k_cols = " ".join(f"R@{K:>2}" for K in k_values)
    print(f"\n{'Method':25s} | {k_cols} | {'Prec':>5} | {'Rank':>5} | {'Active':>6} | {'Total':>5}")
    print("-" * 80)

    for method in methods:
        r = results[method]
        recall_strs = []
        for K in k_values:
            vals = r.get(f"recall@{K}", [])
            if vals:
                recall_strs.append(f"{sum(vals)/len(vals)*100:5.1f}")
            else:
                recall_strs.append("  N/A")

        prec = r.get("precision", [])
        prec_str = f"{sum(prec)/len(prec)*100:5.1f}" if prec else "  N/A"

        rank = r.get("first_hit_rank", [])
        rank_str = f"{sum(rank)/len(rank):5.1f}" if rank else "  N/A"

        n_pri = r.get("n_priority", [])
        n_pri_str = f"{sum(n_pri)/len(n_pri):6.1f}" if n_pri else "   N/A"

        n_lv = r.get("n_leaves", [])
        n_lv_str = f"{sum(n_lv)/len(n_lv):5.0f}" if n_lv else "  N/A"

        print(f"{method:25s} | {' '.join(recall_strs)} | {prec_str} | {rank_str} | {n_pri_str} | {n_lv_str}")

    # === By question type ===
    print(f"\n{'='*70}")
    print("Recall@10 by Question Type")
    print(f"{'='*70}")

    type_results = {m: defaultdict(list) for m in methods}
    for qid, qa in questions.items():
        gt_range = parse_time_reference(qa["time_reference"])
        if gt_range is None:
            continue
        for qt in qa.get("question_type", ["unknown"]):
            for method in methods:
                vals = results[method].get("recall@10", [])
                # We need per-question tracking... reconstruct from order
                pass  # We'll compute this separately

    # Simpler: re-iterate with type tracking
    by_type_method = {m: defaultdict(lambda: {"hit": 0, "total": 0}) for m in methods}
    q_list = list(questions.values())
    for i, qa in enumerate(q_list):
        gt_range = parse_time_reference(qa["time_reference"])
        if gt_range is None:
            continue
        for qt in qa.get("question_type", ["unknown"]):
            for method in methods:
                r10 = results[method].get("recall@10", [])
                if i < len(r10):
                    by_type_method[method][qt]["total"] += 1
                    by_type_method[method][qt]["hit"] += r10[i]

    # Print
    all_types = sorted(set(
        qt for qa in questions.values() for qt in qa.get("question_type", ["unknown"])
    ))
    header = f"{'Type':30s} | " + " | ".join(f"{m:>12s}" for m in methods)
    print(header)
    print("-" * len(header))
    for qt in all_types:
        vals = []
        for m in methods:
            d = by_type_method[m][qt]
            if d["total"] > 0:
                vals.append(f"{d['hit']/d['total']*100:11.1f}%")
            else:
                vals.append(f"{'N/A':>12s}")
        print(f"{qt:30s} | " + " | ".join(vals))

    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieval Quality Evaluation")
    parser.add_argument("--qid_list", type=str, required=True,
                        help="QID list TSV (uid\\tvideo_id)")
    parser.add_argument("--question_path", type=str,
                        default="/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl")
    parser.add_argument("--mem_dir", type=str,
                        default="/lustre/youngbeom/DyHiStreamMem/poc/results/LVBench/stage2_v9")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    questions = load_questions(args.qid_list, args.question_path)
    print(f"Loaded {len(questions)} questions")

    eval_retrieval(questions, args.mem_dir)


if __name__ == "__main__":
    main()
