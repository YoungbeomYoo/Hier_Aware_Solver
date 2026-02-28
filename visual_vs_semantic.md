# Visual vs Semantic Pipeline 비교

> tree_search pipeline 내에서 노드 선택 방식만 다르고, 나머지 (hop loop, judge, navigation)는 동일.

---

## 한눈에 보기

| | Visual (Exact Match) | Semantic (Embedding) |
|---|---|---|
| **노드 선택** | cue 키워드가 caption/key_elements에 **정확히 포함**되는 leaf | Qwen3-Embedding-0.6B로 **코사인 유사도** 기반 L1 노드 선택 |
| **매칭 단위** | 개별 leaf | Level_1 (상위 노드) → 하위 leaf 전부 활성화 |
| **활성화 기준** | `match_count >= 1` (cue 1개 이상 exact match) | parent L1이 top-K 안에 들면 활성화 |
| **정렬** | match_count 내림차순 → start_time | semantic_score 내림차순 → match_count |
| **후보 범위** | 보수적 (정확히 매칭된 leaf만) | 공격적 (top-30 L1의 모든 leaf) |
| **추가 비용** | 없음 (string matching) | embedding forward pass 1회 |
| **Config 차이** | `semantic_matcher` 섹션 없음 | `semantic_matcher` 섹션 있음 |

---

## 파이프라인 흐름

```
┌────────────────────────────────────────────────────────┐
│  STAGE 1: Query Analysis (공통)                         │
│  QueryAnalyzer → cues, target_fields, question_type     │
└──────────────────────┬─────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   ┌──────▼──────┐          ┌──────▼──────┐
   │   Visual    │          │  Semantic   │
   │  (skip)     │          │  STAGE 1.5  │
   └──────┬──────┘          └──────┬──────┘
          │                        │
          │               ┌────────▼────────────────┐
          │               │ q_elements 추출          │
          │               │  = cues + question       │
          │               │  + options (각각)         │
          │               │                          │
          │               │ L1 노드 전부 임베딩       │
          │               │  → cosine sim 계산        │
          │               │  → top-30 L1 선택         │
          │               │  → semantic_scores 반환   │
          │               └────────┬────────────────┘
          │                        │
   ┌──────▼────────────────────────▼──────┐
   │  STAGE 2: Build Filtered Tree        │
   │  tree_filter.build(                  │
   │    tree, cues, target_fields,        │
   │    semantic_scores=...)              │
   └──────────────────┬──────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
   ┌──────▼──────┐        ┌──────▼──────┐
   │   Visual    │        │  Semantic   │
   │             │        │             │
   │ 각 leaf의   │        │ top-30 L1   │
   │ caption에서 │        │ 아래의 모든  │
   │ cue exact   │        │ leaf 활성화  │
   │ match 검색  │        │             │
   │             │        │ 정렬:       │
   │ match >= 1  │        │ semantic    │
   │ → 활성화    │        │ _score      │
   │             │        │ 내림차순    │
   │ 정렬:       │        │             │
   │ match_count │        │             │
   │ 내림차순    │        │             │
   └──────┬──────┘        └──────┬──────┘
          │                      │
          └──────────┬───────────┘
                     │
   ┌─────────────────▼───────────────────┐
   │  STAGE 3-5: Hop Loop (공통)          │
   │                                      │
   │  for hop in 1..max_hops:             │
   │    1. context_assembler              │
   │       → target caption + 부모 summary │
   │    2. judge (text-only)              │
   │       → answerable? confidence?      │
   │    3. high conf → 종료               │
   │       medium → visual 시도           │
   │       low → 다음 hop                 │
   │    4. navigate_next                  │
   │       → sibling / temporal / unseen  │
   └─────────────────────────────────────┘
```

---

## Visual: Exact Match 상세

```python
# tree_filter.py — _match_node()
def _match_node(node, cues_lower, fields):
    # 1. 검색 대상 텍스트 구성
    text = node["summary"] + " " + node["caption"]
    for field in ["actions", "objects", "persons", ...]:
        text += " ".join(node["key_elements"][field])
    text = text.lower()

    # 2. 각 cue가 text에 정확히 포함되는지 체크
    matched = []
    for cue in cues_lower:
        if cue in text:          # ← exact substring match
            matched.append(cue)

    return len(matched), matched
```

**예시:**
```
Cues: ["cooking", "knife", "woman"]

Leaf A: caption="woman cutting vegetables with knife"
  → "cooking" ✗, "knife" ✓, "woman" ✓ → score=2, 활성화 ✓

Leaf B: caption="man preparing ingredients"
  → "cooking" ✗, "knife" ✗, "woman" ✗ → score=0, 활성화 ✗

Leaf C: caption="cooking show opening sequence"
  → "cooking" ✓ → score=1, 활성화 ✓

priority_leaves = [Leaf A (score=2), Leaf C (score=1)]
```

**장점**: False positive 거의 없음. "knife"면 진짜 칼이 언급된 leaf만 선택.
**단점**: Recall 낮음. 동의어/유의어 놓침 ("blade" ≠ "knife", "culinary" ≠ "cooking").

---

## Semantic: Embedding Match 상세

```python
# semantic_matcher.py — select_top_nodes()
def select_top_nodes(q_elements, tree, level="Level_1"):
    # 1. q_elements 임베딩
    q_embs = encode(q_elements)  # [N_q, dim]

    # 2. 각 L1 노드 임베딩
    for l1_node in tree["Level_1"]:
        node_texts = flatten_key_elements(l1_node) + [l1_node["summary"]]
        node_embs = encode(node_texts)  # [N_n, dim]

        # 3. 유사도 행렬 계산
        sim_matrix = q_embs @ node_embs.T  # [N_q, N_n]

        # 4. 각 q_element의 최대 유사도
        max_sims = sim_matrix.max(dim=1)   # [N_q]

        # 5. 점수 = max_sims의 합
        score = max_sims.sum()  # score_mode="sum"

    # 6. top-30 L1 반환
    return sorted(scores, reverse=True)[:top_k]
```

**예시:**
```
Q: "How is the woman preparing the meal?"
q_elements: ["How is the woman preparing the meal?",
             "A. chopping", "B. boiling", "C. frying", "D. baking"]

L1[0]: key_elements=["woman in kitchen", "cutting board", "vegetable soup"]
  → q_emb("chopping") · n_emb("cutting board") = 0.95  ✓ good
  → score = 8.5

L1[17]: key_elements=["BASKET", "holding microphone", "giant head sculpture"]
  → q_emb("B. Black") · n_emb("BASKET") = 1.00         ✗ BAD (surface pattern)
  → q_emb("guitar") · n_emb("giant head sculpture") = 0.996  ✗ BAD
  → score = 8.8  ← 엉뚱한데 점수 높음!

top-30에 L1[17]이 포함됨 → 그 아래 leaf 전부 활성화
```

**장점**: 동의어/유의어 커버 ("cutting" ↔ "chopping", "prepare" ↔ "cook")
**단점**: **False positive 심각** — Qwen3-Embedding-0.6B가 표면 패턴(글자 겹침, 토큰 유사)에 속음

---

## Subset 실험 결과 (30문제)

```
                                     Accuracy  AvgRel  %HasRel  Winner
visual3 (exact match)                 46.7%    0.068    42.3%   18/30
semantic (embedding)                  46.7%    0.056    37.4%   12/30
visual2 (구버전)                       43.3%     N/A      N/A    0/30
```

### Semantic Embedding Match 품질
```
Good matches (단어 겹침 있음): 122
Bad matches (sim>0.99인데 무관): 274
Bad ratio: 69.2%  ← 매칭의 70%가 의미 없는 false positive
```

### Bad Match 사례
| q_element | matched_to | similarity | 판정 |
|-----------|-----------|-----------|------|
| "Gypsies" | "Ghazni" | 0.9997 | BAD — 글자 유사 |
| "massacres" | "shield" | 0.9950 | BAD — 무관 |
| "B. Black" | "BASKET" | 1.0009 | BAD — B 대문자 |
| "guitar" | "giant head sculpture" | 0.9962 | BAD — "gi" 접두사 |
| "financial" | "African descent" | 1.0009 | BAD — 부분 토큰 겹침 |

### LVBench Target Time Hit (semantic, 45문제)
```
Target time에 닿은 비율:  48.9% (22/45)
50% 이상 커버:            33.3% (15/45)
Miss 시 target까지 거리:  중앙값 86초 (1.4분)
```

---

## 핵심 차이 정리

### 같은 부분 (공통)
- Query analysis (cue 추출)
- Hop loop 구조 (max 5 hops)
- Judge (text-only / visual)
- Navigation strategy (sibling, temporal, unseen)
- Context assembly
- Answer extraction

### 다른 부분 (Stage 1.5 + Stage 2)

| 단계 | Visual | Semantic |
|------|--------|----------|
| **Stage 1.5** | 없음 | L1 노드 전부 임베딩 → top-30 선택 |
| **Leaf 활성화** | cue ∈ caption (exact) | parent L1 ∈ top-30 |
| **후보 수** | 적음 (정확한 leaf만) | 많음 (top-30 L1의 모든 leaf) |
| **정렬 기준** | match_count | semantic_score |
| **False positive** | 거의 없음 | 69.2% (embedding 품질 문제) |
| **False negative** | 높을 수 있음 (동의어 놓침) | 낮음 |
| **비용** | O(n) string match | O(n×m) embedding + matmul |

---

## 개선 방향

1. **Semantic matcher 교체**: Qwen3-Embedding-0.6B → 더 큰 모델 or task-specific fine-tuned
2. **Hybrid**: exact match 먼저 → 결과 부족하면 semantic fallback
3. **Score threshold**: top-K 대신 score > threshold로 필터링 (low-quality 매칭 제거)
4. **Negative filtering**: bad match 패턴 (대문자 1글자, 짧은 토큰) 사전 필터링
5. **Reranking**: semantic top-30 선택 후 LLM으로 rerank (비용 vs 품질 트레이드오프)
