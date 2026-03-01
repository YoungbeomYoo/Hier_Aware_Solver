# Agent System Comparison: Ours vs VideoLucy

## 1. 전체 파이프라인 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OURS (Tree Search)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [0. Planner]  →  [1. Query Analyzer]  →  [1.5 Semantic Matcher]           │
│       LLM              LLM/Rule               Embedding                    │
│    (optional)                                 (optional)                    │
│                                                                             │
│       ↓                                                                     │
│  [2. Tree Filter]  →  [2.5 Phase 0 (G1)]  →  [2.7 A2 Recovery Cue]       │
│     Non-LLM            LLM (force prompt)       LLM (optional)            │
│                         coarse overview          시간대+focus 추론          │
│                         → high conf? done     →  [3. Target Select]       │
│                                                                             │
│                    →  ┌─── Hop Loop (max 5) ───┐                           │
│                       │  4a. Context Assembly   │                           │
│                       │       Non-LLM           │                           │
│                       │          ↓              │                           │
│                       │  4b. Judge (text)       │                           │
│                       │       LLM               │                           │
│                       │          ↓              │                           │
│                       │  4b'. Visual Judge      │                           │
│                       │       VLM (optional)    │                           │
│                       │          ↓              │                           │
│                       │  4c. History Compact    │                           │
│                       │       Non-LLM           │                           │
│                       │          ↓              │                           │
│                       │  4d. Navigate Next      │                           │
│                       │       Non-LLM / LLM    │                           │
│                       └────────────────────────┘                           │
│                                  ↓                                          │
│                       [5. Forced Answer / Random]                           │
│                               LLM                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              VideoLucy                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Phase 1]  Coarse Memory: VLM caption → LLM summarize                     │
│                    ↓                                                        │
│  [Phase 2]  Answer with Coarse Memory (LLM) → Confidence?                  │
│                    ↓ (False)                                                │
│             ┌─── Fine Search Loop (max 5) ───┐                             │
│  [Phase 3]  │  3a. Question Type Judge (LLM) │                             │
│             │          ↓                      │                             │
│             │  3b. Time Period Selector (LLM) │                             │
│             │          ↓                      │                             │
│             │  3c. Fine Memory Extract (VLM)  │                             │
│             │          ↓                      │                             │
│             │  3d. Answer w/ Fine Mem (LLM)   │                             │
│             │       → Confidence? True → DONE │                             │
│             └─────────────────────────────────┘                             │
│                    ↓ (still False)                                           │
│  [Phase 4]  Must Answer (LLM) — 강제 답변                                   │
│                    ↓                                                        │
│  [+]        Answer Judge (LLM) — 자유형 답변 → MCQ 매핑                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ablation 축 정의

각 축을 독립적으로 갈아끼울 수 있는 ablation 포인트:

### A. 질문 분석 (Query Analysis)
| ID | 설정 | 설명 |
|----|------|------|
| A0 | Rule-only | regex로 cue/time 추출 (LLM 없음) |
| A1 | LLM-based | LLM이 question type + cues 추출 (현재 default) |

### B. 노드 선택 (Tree Filter)
| ID | 설정 | 설명 |
|----|------|------|
| B0 | 전체 (no filter) | threshold=0, 모든 leaf 활성화 |
| B1 | Exact key matching | key_elements exact substring match (현재 default) |
| B3 | LLM-based selection | LLM이 직접 어떤 노드 볼지 결정 |

### C. Judge / Solvability 전략
| ID | 설정 | 설명 |
|----|------|------|
| C0 | strict | 기존 기본 prompt. high/medium/low 3단계 |
| C1 | relaxed | 관대한 기준 |
| C2 | videolucy | VideoLucy prompt. True/False binary + 보수적 추론 |
| C3 | videolucy + answer_judge | VideoLucy prompt + 별도 answer-to-MCQ 매핑 agent 추가 |

### D. Visual 전략 (Frame 활용)
| ID | 설정 | 설명 |
|----|------|------|
| D0 | 없음 (text-only) | 메모리 텍스트만으로 판단 |
| D1 | Visual Judge (1-stage) | judge가 필요시 VLM 1회 호출 |
| D2 | Two-Stage Visual | Scout → LLM Focus Select → Focus VLM → Rejudge |
| D3 | Always Visual | 매 hop마다 무조건 VLM frame captioning |
| D5 | Visual Context Enrichment | hop loop에서 C3 judge low/medium → frame caption → context 추가 → C3 재판단 |

### E. 탐색 전략 (Navigation)
| ID | 설정 | 설명 |
|----|------|------|
| E0 | Tree-based | 현재 방식. tree structure 따라 이동 (현재 default) |

### F. History 관리
| ID | 설정 | 설명 |
|----|------|------|
| F0 | HistoryAccumulator | 삭제 없이 전부 누적 |
| F1 | HistoryCompactor | 압축 + 상태 추적 (compact) |

### G. Phase 0 (Coarse-First)
| ID | 설정 | 설명 |
|----|------|------|
| G0 | 없음 | 기존 방식 — 바로 hop loop 진입 |
| G1 | coarse_first | Level_N~1 전체 summary → force prompt로 1차 답변 시도 → high conf면 early return |

---

## 3. Ablation 실험 전체 추적표 (130문제 subset)

### Baseline

- **VideoLucy (subset 130문제)**: 60/130 = **46.2%**
- **VideoLucy (full 900문제)**: 467/900 = **51.9%**

### 실험 결과표

| # | 실험명 | C | D | F | 기타 | Acc | vs VL | Round |
|---|--------|---|---|---|------|-----|-------|-------|
| — | **VideoLucy** | — | — | — | — | **46.2%** | — | ref |
| 1 | baseline_exact | C0 | D1 | F1 | | 32.3% | -13.8% | R1 |
| 2 | videolucy_prompt | C2 | D1 | F1 | | 32.3% | -13.8% | R1 |
| 3 | strict_accumulate | C0 | D1 | F0 | | 34.6% | -11.5% | R1 |
| 4 | videolucy_accumulate | C2 | D1 | F0 | | 34.6% | -11.5% | R1 |
| 5 | text_only | C2 | D0 | F0 | | 42.3% | -3.8% | R1 |
| 6 | twostage | C2 | D2 | F0 | | 38.5% | -7.7% | R1 |
| 7 | text_only_compact | C2 | D0 | F1 | | 43.1% | -3.1% | R2 |
| 8 | text_only_strict | C0 | D0 | F0 | | 34.6% | -11.5% | R2 |
| 9 | no_caption | C2 | D1nc | F0 | | 36.9% | -9.2% | R2 |
| 10 | no_query | C2 | D0 | F0 | A0 | 40.8% | -5.4% | R3 |
| 11 | llm_select | C2 | D0 | F0 | B3 | 42.3% | -3.8% | R3 |
| 12 | always_visual | C2 | D3 | F1 | | 34.6% | -11.5% | R4 |
| 13 | always_visual_acc | C2 | D3 | F0 | | 34.6% | -11.5% | R4 |
| 14 | strict_twostage | C0 | D2 | F0 | | 38.5% | -7.7% | R5 |
| 15 | strict_always_visual | C0 | D3 | F0 | | 34.6% | -11.5% | R5 |
| 16 | relaxed_text_only | C1 | D0 | F0 | | 42.3% | -3.8% | R6 |
| 17 | relaxed_visual | C1 | D1 | F0 | | 34.6% | -11.5% | R6 |
| 18 | relaxed_twostage | C1 | D2 | F0 | | 38.5% | -7.7% | R6 |
| 19 | relaxed_always_visual | C1 | D3 | F0 | | 34.6% | -11.5% | R6 |
| 20 | answerjudge_text_only | C3 | D0 | F0 | | 43.1% | -3.1% | R6 |
| 21 | answerjudge_visual | C3 | D1 | F0 | | 34.6% | -11.5% | R6 |
| 22 | answerjudge_twostage | C3 | D2 | F0 | | 38.5% | -7.7% | R6 |
| 23 | answerjudge_always_visual | C3 | D3 | F0 | | 34.6% | -11.5% | R6 |
| 24 | **c3_compact** | **C3** | **D0** | **F1** | | **45.4%** | **-0.8%** | **R7** |
| 25 | no_filter | C3 | D0 | F0 | B0 | 42.3% | -3.8% | R7 |
| 26 | no_filter_compact | C3 | D0 | F1 | B0 | 43.8% | -2.3% | R7 |
| 27 | visual_relaxed_rejudge | C2 | D1 | F0 | Cv | 32.3% | -13.8% | R7 |
| 28 | **coarse_first** | **C3** | **D0** | **F1** | **G1** | **46.9%** | **+0.8%** | **R8** |
| 29 | visual_enrich | C3 | D5 | F1 | G1 | 46.9% | +0.8% | R9 |

### C×D 전체 그리드 (F0 고정, 130문제)

| | D0 (text-only) | D1 (1-stage) | D2 (two-stage) | D3 (always-vis) |
|---|---|---|---|---|
| **C0 (strict)** | 34.6% | 34.6% | 38.5% | 34.6% |
| **C1 (relaxed)** | 42.3% | 34.6% | 38.5% | 34.6% |
| **C2 (videolucy)** | 42.3% | 34.6% | 38.5% | 34.6% |
| **C3 (VL+AJ)** | **43.1%** | 34.6% | 38.5% | 34.6% |

---

## 4. Full Run 결과 (900문제)

### 축별 config 분해

| Config | A | B | C | D | E | F | G | Acc | vs VL |
|--------|---|---|---|---|---|---|---|-----|-------|
| VideoLucy | — | — | — | — | — | — | — | **51.9%** | — |
| `full_best_c3d0` | A1 | B1 | C3 | D0 | E0 | **F0** | G0 | 47.7% | -4.2%p |
| `full_best_c3d0f1` | A1 | B1 | C3 | D0 | E0 | **F1** | G0 | 47.4% | -4.4%p |
| **`full_best_c3d0f1_g1`** | A1 | B1 | C3 | D0 | E0 | **F1** | **G1** | **53.2%** | **+1.3%p** |
| `full_best_c3d5f1_g1` | A1 | B1 | C3 | D5 | E0 | **F1** | **G1** | **52.9%** | +1.0%p |

- `c3d0` → `c3d0f1`: F0→F1 변경만. Full에서는 **-0.3%p** (subset +2.3%p 효과 소멸)
- `c3d0f1` → `c3d0f1_g1`: G0→G1 변경만. **+5.8%p** — Phase 0 단독 효과
- A1/B1/C3/D0/E0은 세 config 모두 동일 (best ablation 세팅 고정)

### Task Type별 비교 (Full 900문제)

| Task Type | N | C3D0F0 | C3D0F1 | **G1** | VLucy | G1-F1 | G1-VL | 비고 |
|-----------|---|--------|--------|--------|-------|-------|-------|------|
| Object Reasoning | 240 | 49.2% | 47.9% | 49.2% | 51.7% | +1.2% | -2.5% | |
| Action Reasoning | 180 | 41.7% | 43.9% | **48.3%** | 46.1% | +4.4% | **+2.2%** | G1이 VL 역전 |
| Information Synopsis | 163 | 65.0% | 64.4% | **69.9%** | 74.2% | +5.5% | -4.3% | Phase 0 타겟, gap 축소 |
| Temporal Reasoning | 91 | 39.6% | 36.3% | **50.5%** | 40.7% | **+14.3%** | **+9.9%** | G1 최대 개선 |
| Action Recognition | 63 | 42.9% | 41.3% | **49.2%** | 34.9% | +7.9% | **+14.3%** | G1이 VL 대폭 상회 |
| Object Recognition | 54 | 38.9% | 44.4% | **50.0%** | 48.1% | +5.6% | **+1.9%** | G1이 VL 역전 |
| **Counting Problem** | **48** | 22.9% | 25.0% | **45.8%** | 37.5% | **+20.8%** | **+8.3%** | **G1 최대 점프** |
| Attribute Perception | 27 | 59.3% | 59.3% | **63.0%** | 63.0% | +3.7% | +0.0% | VL과 동률 |
| OCR Problems | 14 | 50.0% | 42.9% | 42.9% | 35.7% | +0.0% | **+7.1%** | 우리가 우세 |
| Spatial Reasoning | 11 | 72.7% | 72.7% | 72.7% | 81.8% | +0.0% | -9.1% | |
| Temporal Perception | 6 | 50.0% | 33.3% | 50.0% | 66.7% | +16.7% | -16.7% | N 작음 |
| Spatial Perception | 3 | 33.3% | 33.3% | 0.0% | 33.3% | -33.3% | -33.3% | N 작음 |

### 4.4 Full Run: D5 (C3+D5+F1+G1) — Visual Context Enrichment (900Q)

**Overall Accuracy:**

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| D5 (C3+D5+F1+G1) | 476 | 900 | 52.9% |
| G1 (C3+D0+F1+G1) | 479 | 900 | 53.2% |
| VideoLucy | 467 | 900 | 51.9% |

D5 vs G1: **-0.3%p** (visual enrichment slightly hurts)
D5 vs VL: **+1.0%p** (still beats VideoLucy)

**Phase 0 vs Hop Loop:**

| | D5 | G1 |
|--|----|----|
| Phase 0 | 442/806 = 54.8% | 442/806 = 54.8% |
| Hop Loop | 34/94 = 36.2% | 37/94 = 39.4% |
| Visual Used | 94 | 88 |

Phase 0 is identical (visual only applies in hop loop). Hop loop dropped from 39.4% to 36.2% (-3.2%p).

**Hop Loop Change Detail:**
- Gained (G1✗→D5✓): 13
- Lost (G1✓→D5✗): 16
- Both correct: 21
- Both wrong: 44
- Net: -3

**Per Task Type (Full 900Q):**

| Task Type | N | D5 | G1 | VL | D5-G1 | D5-VL |
|-----------|---|----|----|----|----|-----|
| Action Reasoning | 180 | 46.1% | 48.3% | 46.1% | -2.2% | +0.0% |
| Action Recognition | 63 | 44.4% | 49.2% | 34.9% | -4.8% | +9.5% |
| Attribute Perception | 27 | 66.7% | 63.0% | 63.0% | +3.7% | +3.7% |
| Counting Problem | 48 | 47.9% | 45.8% | 37.5% | +2.1% | +10.4% |
| Information Synopsis | 163 | 70.6% | 69.9% | 74.2% | +0.6% | -3.7% |
| OCR Problems | 14 | 35.7% | 42.9% | 35.7% | -7.1% | +0.0% |
| Object Reasoning | 240 | 48.3% | 49.2% | 51.7% | -0.8% | -3.3% |
| Object Recognition | 54 | 51.9% | 50.0% | 48.1% | +1.9% | +3.7% |
| Temporal Reasoning | 91 | 53.8% | 50.5% | 40.7% | +3.3% | +13.2% |

**Visual Enrichment Effectiveness (Hop Loop Only, 94Q):**

| Task Type | N | D5 | G1 | Diff |
|-----------|---|----|----|------|
| Action Reasoning | 30 | 36.7% | 50.0% | -13.3% |
| Action Recognition | 9 | 33.3% | 66.7% | -33.3% |
| Temporal Reasoning | 7 | 57.1% | 14.3% | +42.9% |
| Counting Problem | 5 | 40.0% | 20.0% | +20.0% |
| Object Reasoning | 25 | 24.0% | 32.0% | -8.0% |
| Object Recognition | 6 | 50.0% | 33.3% | +16.7% |
| Attribute Perception | 2 | 50.0% | 0.0% | +50.0% |
| Information Synopsis | 4 | 75.0% | 50.0% | +25.0% |
| OCR Problems | 5 | 20.0% | 40.0% | -20.0% |

**D5 Conclusion:**
Visual caption enrichment is a double-edged sword. It helps significantly for Temporal Reasoning (+42.9%p) and Counting (+20.0%p) in the hop loop, but severely hurts Action Recognition (-33.3%p) and Action Reasoning (-13.3%p). The VLM captions appear to introduce noise that misleads the C3 text judge for action-related questions. The net effect on hop loop is -3.2%p, dragging overall accuracy down by -0.3%p.

This suggests that visual enrichment needs to be **selective** — applied only to question types where visual information is genuinely useful (temporal, counting, attribute) and avoided for action-related questions where text summaries are more reliable.

---

## 5. Phase 0 (G1) 상세 분석

### 동작 원리

1. Hop loop 진입 전, Level_N ~ Level_1 **전체** summary를 시간순 concat
2. **Force prompt**로 LLM에게 강제 답변 요청 (거부 불가)
3. high confidence → **early return** (hop loop skip)
4. medium/low → Phase 0 답변을 baseline으로 보관, hop loop 계속 진행

### 실험 결과

**Full Run (900문제):**

| | C3+D0+F1 | **C3+D0+F1+G1** | VideoLucy |
|---|---|---|---|
| Accuracy | 47.4% | **53.2% (+5.8%p)** | 51.9% |
| Phase 0 early return | — | 806/900 (89.6%) | — |
| Phase 0 accuracy | — | 442/806 (54.8%) | — |
| Hop loop accuracy | — | 37/94 (39.4%) | — |

**130문제 subset (R8):**

| | Base (C3+D0+F1) | +Phase 0 (G1) | VideoLucy |
|---|---|---|---|
| Accuracy | 45.4% | **46.9% (+1.5%p)** | 46.2% |
| Phase 0 early return | — | 122/130 (93.8%) | — |

**Information Synopsis 163문제:**

| | Base (C3+D0+F1) | +Phase 0 (G1) | VideoLucy |
|---|---|---|---|
| Accuracy | 64.4% | **69.9% (+5.5%p)** | 74.2% |
| vs VL gap | -9.8%p | **-4.3%p** | — |

### Phase/Hop 상세 분석

**Ours (G1) — Method별 정확도:**

| Method | Correct/Total | Acc |
|--------|---------------|-----|
| phase0_coarse (early return) | 442/806 | 54.8% |
| tree_search_hop3 | 2/3 | 66.7% |
| tree_search_hop4 | 1/5 | 20.0% |
| tree_search_hop5 | 33/83 | 39.8% |
| tree_search_fallback | 1/3 | 33.3% |

**Per-Category Phase 0 vs Hop Loop:**

| Task Type | P0 early (acc) | Hop (acc) | G1 | VL |
|-----------|----------------|-----------|-----|-----|
| Action Reasoning | 72/150 (48.0%) | 15/30 (50.0%) | 48.3% | 46.1% |
| Action Recognition | 25/54 (46.3%) | 6/9 (66.7%) | 49.2% | 34.9% |
| Attribute Perception | 17/25 (68.0%) | 0/2 (0.0%) | 63.0% | 63.0% |
| Counting Problem | 21/43 (48.8%) | 1/5 (20.0%) | 45.8% | 37.5% |
| **Information Synopsis** | **112/159 (70.4%)** | 2/4 (50.0%) | **69.9%** | **74.2%** |
| OCR Problems | 4/9 (44.4%) | 2/5 (40.0%) | 42.9% | 35.7% |
| **Object Reasoning** | **110/215 (51.2%)** | **8/25 (32.0%)** | **49.2%** | **51.7%** |
| Object Recognition | 25/48 (52.1%) | 2/6 (33.3%) | 50.0% | 48.1% |
| Spatial Reasoning | 8/11 (72.7%) | — | 72.7% | 81.8% |
| Temporal Reasoning | 45/84 (53.6%) | 1/7 (14.3%) | 50.5% | 40.7% |

**VideoLucy Phase 분포 (688/900 로그 파싱):**

| Phase | N (%) | Acc | 설명 |
|-------|-------|-----|------|
| Phase 2 (Coarse) | 89 (12.9%) | **84.3%** | coarse memory로 즉시 답변 |
| Phase 3 (Fine) | 100 (14.5%) | **65.0%** | fine search loop로 해결 |
| Phase 4 (Must Answer) | 499 (72.5%) | 42.7% | 강제 답변 |

**Ours vs VideoLucy 전략 차이:**
- VL: 보수적. coarse에서 12.9%만 early return, 정확도 84.3%
- Ours: 공격적. Phase 0에서 89.6% early return, 정확도 54.8%
- VL의 72.5%가 Phase 4(강제)까지 가서 42.7% → 우리 hop loop(39.4%)과 비슷
- **핵심 차이**: VL은 Phase 3(fine search)에서 14.5%를 65.0%로 해결 — 우리 hop loop은 이에 대응하는 구간이 약함

---

## 6. Full Ablation Study (900Q) — 최종 결과

### 6.1 Overall Accuracy

| Config | Axes | Correct | Acc | vs Best | vs VL |
|--------|------|---------|-----|---------|-------|
| **R10a Flat Baseline** | Flat (no tree) | 493/900 | **54.8%** | **+1.6%p** | +2.9% |
| R10b No KE | -KE | 481/900 | 53.4% | +0.2%p | +1.5% |
| -C3→C0 (Strict judge) | A1 B1 C0 D0 F1 G1 | 485/900 | 53.9% | +0.7%p | +2.0% |
| R10c No Caption | A1 B1 C3 D0 F1 G1 -Cap | 481/900 | 53.4% | +0.2%p | +1.6% |
| **Best** | A1 B1 C3 D0 F1 G1 | 479/900 | **53.2%** | — | +1.3% |
| -AJ (no Answer Judge) | A1 B1 C2 D0 F1 G1 | 479/900 | 53.2% | +0.0%p | +1.3% |
| -B1 (no Tree Filter) | A1 B0 C3 D0 F1 G1 | 478/900 | 53.1% | -0.1%p | +1.2% |
| D5 (Visual Enrich) | A1 B1 C3 D5 F1 G1 | 476/900 | 52.9% | -0.3%p | +1.0% |
| -F1 (no Compact History) | A1 B1 C3 D0 F0 G1 | 476/900 | 52.9% | -0.3%p | +1.0% |
| -A1 (Rule-only Query) | A0 B1 C3 D0 F1 G1 | 475/900 | 52.8% | -0.4%p | +0.9% |
| VideoLucy | (baseline) | 467/900 | 51.9% | -1.3%p | — |
| -G1 (no Phase 0, w/F1) | A1 B1 C3 D0 F1 G0 | 427/900 | 47.4% | -5.8%p | -4.4% |
| -G1 (no Phase 0, w/F0) | A1 B1 C3 D0 F0 G0 | 429/900 | 47.7% | -5.5%p | -4.2% |

### 6.2 Ablation Contribution Ranking

| Rank | Feature | Removed → Acc | Contribution |
|------|---------|---------------|--------------|
| #1 | **G1 (Phase 0)** | 47.4% | **+5.8%p** |
| #2 | A1 (LLM Query) | 52.8% | +0.4%p |
| #3 | F1 (Compact History) | 52.9% | +0.3%p |
| #4 | B1 (Tree Filter) | 53.1% | +0.1%p |
| #5 | AJ (Answer Judge) | 53.2% | +0.0%p |
| #6 | C3 (VL+AJ Judge) | 53.9% | **-0.7%p** (제거 시 오히려 개선) |

**G1(Phase 0)이 유일하게 의미 있는 기여자.** 나머지 컴포넌트(A, B, C, F)는 ±0.4%p 이내 — 실질적 기여 없음.
특히 C3(VideoLucy prompt + AJ)는 **제거 시 +0.7%p 개선** — hop loop에서 C0(strict)이 45.7%로 C3(39.4%)보다 +6.3%p 높음.

### 6.3 Phase 0 vs Hop Loop

| Config | P0 N | P0 Acc | Hop N | Hop Acc | Total |
|--------|------|--------|-------|---------|-------|
| Best (C3) | 806 | 54.8% | 94 | 39.4% | 53.2% |
| -C3→C0 (Strict) | 806 | 54.8% | 94 | **45.7%** | **53.9%** |
| R10c (-Caption) | 806 | 54.8% | 94 | 41.5% | 53.4% |
| -AJ (C2) | 806 | 54.8% | 94 | 39.4% | 53.2% |
| -B1 (no filter) | 806 | 54.8% | 94 | 38.3% | 53.1% |
| -F1 (accumulate) | 806 | 54.8% | 94 | 36.2% | 52.9% |
| -A1 (rule-only) | 806 | 54.8% | 94 | 35.1% | 52.8% |

**Phase 0은 모든 config에서 동일 (54.8%)** — 변경은 hop loop에서만 영향.
**핵심**: hop loop이 39.4%로 Phase 0(54.8%)보다 -15.4%p 낮아 전체를 끌어내림.
C0(strict judge)가 hop loop에서 45.7%로 가장 높음 — C3(videolucy)가 hop loop에서는 역효과.

### 6.4 R10: Hierarchical Structure Ablation

| Experiment | N | Accuracy | vs Best | P0/Flat Acc | Hop Acc | 해석 |
|-----------|---|----------|---------|-------------|---------|------|
| **R10a Flat Baseline** | 900 | **54.8%** | **+1.6%p** | 54.8% (flat) | N/A | **트리 탐색보다 flat dump가 우수** |
| R10b No KE | 900 | 53.4% | +0.2%p | 54.8% (806) | 41.5% (94) | KE 제거해도 성능 유지 |
| R10c No Caption | 900 | 53.4% | +0.2%p | 54.8% (806) | 41.5% (94) | Caption 제거 영향 무시할 수준 |
| Best | 900 | 53.2% | — | 54.8% (806) | 39.4% (94) | — |

**R10a의 의미**: Flat(54.8%) = Phase 0 accuracy(54.8%). Flat은 hop loop 없이 1회 판단.
Best(53.2%)는 Phase 0(54.8%) + hop loop(39.4%)의 가중 평균.
→ **hop loop(39.4%)이 Phase 0(54.8%) 대비 -15.4%p 낮아 전체를 깎는 구조**.
→ Hierarchical navigation이 아닌 **Phase 0(coarse overview)**가 실제 성능 원천.

**R10b 최종 결과 (900/900)**: KE 제거 시 53.4%로 Best(53.2%)와 거의 동일 (+0.2%p).
Phase 0 accuracy 동일(54.8%), hop loop은 41.5%로 Best hop(39.4%)보다 +2.1%p 높음.
→ key_elements가 hop loop에서 미약하게 해로운 정도. structured index의 retrieval 기여 없음.

**R10c 최종 결과**: Caption 제거는 +0.2%p. Hop loop에서 41.5%로 Best(39.4%)보다 +2.1%p.
→ Raw caption이 judge에 불필요한 정보를 추가해 hop loop 정확도를 떨어뜨리는 효과.
→ Summary만으로 충분하며, caption은 context 길이만 늘릴 뿐 정보 이득 없음.

### 6.5 Per Task Type (Full 900Q)

| Task Type | N | Best | Flat | NoKE | NoCap | C0 | -G1 | -A1 | -B1 | -F1 | D5 | VL |
|-----------|---|------|------|------|-------|-----|-----|-----|-----|-----|-----|-----|
| Object Reasoning | 240 | 49.2% | 50.8% | 50.0% | 50.8% | 49.2% | 47.9% | 48.8% | 48.8% | 49.6% | 48.3% | 51.7% |
| Action Reasoning | 180 | 48.3% | 49.4% | 47.2% | 47.2% | 49.4% | 43.9% | 47.2% | 47.8% | 46.7% | 46.1% | 46.1% |
| Information Synopsis | 163 | 69.9% | **76.7%** | 69.3% | 69.9% | 69.9% | 64.4% | 69.9% | 69.9% | 69.3% | 70.6% | 74.2% |
| Temporal Reasoning | 91 | 50.5% | 48.4% | 52.7% | 51.6% | **53.8%** | 36.3% | 53.8% | 50.5% | 52.7% | 53.8% | 40.7% |
| Action Recognition | 63 | 49.2% | 47.6% | 47.6% | 47.6% | 49.2% | 41.3% | 44.4% | 49.2% | 47.6% | 44.4% | 34.9% |
| Object Recognition | 54 | 50.0% | 50.0% | 50.0% | 51.9% | **53.7%** | 44.4% | 48.1% | 50.0% | 48.1% | 51.9% | 48.1% |
| Counting Problem | 48 | **45.8%** | 39.6% | 47.9% | 45.8% | 45.8% | 25.0% | 45.8% | 47.9% | 45.8% | 47.9% | 37.5% |
| Attribute Perception | 27 | 63.0% | 63.0% | 66.7% | 63.0% | 66.7% | 59.3% | 63.0% | 63.0% | 63.0% | 66.7% | 63.0% |
| OCR Problems | 14 | **42.9%** | 42.9% | 42.9% | 35.7% | 28.6% | 42.9% | 42.9% | 42.9% | 42.9% | 35.7% | 35.7% |
| Spatial Reasoning | 11 | 72.7% | 72.7% | 72.7% | 72.7% | 72.7% | 72.7% | 72.7% | 72.7% | 72.7% | 72.7% | 81.8% |
| **TOTAL** | **900** | **53.2%** | **54.8%** | **53.4%** | **53.4%** | **53.9%** | **47.4%** | **52.8%** | **53.1%** | **52.9%** | **52.9%** | **51.9%** |

**Flat vs Best — Best가 이기는 카테고리 (hierarchical이 유리한 영역):**
- **Counting Problem**: Best 45.8% vs Flat 39.6% (+6.2%p) ◀ hop loop이 counting에는 유리
- **Temporal Reasoning**: Best 50.5% vs Flat 48.4% (+2.1%p) ◀ 시간 관련 탐색에서 tree 유리
- **Action Recognition**: Best 49.2% vs Flat 47.6% (+1.6%p)

**Flat vs Best — Flat이 이기는 카테고리:**
- **Information Synopsis**: Flat 76.7% vs Best 69.9% (+6.7%p) ▶ 전체 맥락 1회 판단이 압도
- **Object Reasoning**: Flat 50.8% vs Best 49.2% (+1.7%p)
- **Action Reasoning**: Flat 49.4% vs Best 48.3% (+1.1%p)

### 6.6 핵심 발견 (Updated)

1. **Phase 0(G1)이 유일한 대형 기여자**: +5.8%p. 나머지 컴포넌트는 ±0.4%p 이내
2. **Flat Baseline(54.8%) > Best(53.2%)**: 트리 탐색이 오히려 성능 하락. hop loop이 병목
3. **Hop loop이 전체를 깎음**: Phase 0 = 54.8%, hop loop = 39.4%, 가중평균 = 53.2%
4. **C3(VideoLucy prompt)가 hop loop에서 역효과**: C0(strict) hop 45.7% vs C3 hop 39.4%
5. **Caption/KE 영향 거의 없음**: R10b(-KE)=53.4%, R10c(-Cap)=53.4% ≈ Best. summary만으로 충분
6. **Answer Judge 기여 0**: C2=C3 (53.2% 동일)
7. **Info Synopsis**: Flat(76.7%) vs Best(69.9%) vs VL(74.2%) — Flat이 VL도 압도
8. **Counting/Temporal**: Best > Flat — **hop loop이 특정 시간대 탐색에는 유리**
9. **성능 원천은 Phase 0 = coarse overview 전략**: 전체 요약을 보고 판단하는 것이 핵심
10. **Tree 구축은 필수, Tree 탐색은 문제**: Stage 2가 만든 summary가 성능의 근본. 그 summary를 iterative하게 보는 것(hop)이 한번에 보는 것(Flat/Phase0)보다 못함

### 6.7 카테고리별 시사점: Hierarchical Structure의 장단점

**Best > Flat (hop loop의 tree navigation이 유리한 영역):**

| 카테고리 | N | Best | Flat | Δ | 이유 |
|---------|---|------|------|---|------|
| Counting Problem | 48 | **45.8%** | 39.6% | +6.2%p | 특정 시간대 반복 관찰 필요 |
| Temporal Reasoning | 91 | **50.5%** | 48.4% | +2.1%p | 시간 순서 기반 탐색 |
| Action Recognition | 63 | **49.2%** | 47.6% | +1.6%p | 특정 행동 구간 포커싱 |

→ **"어디를 봐야 하는지"가 명확한 문제에서는 tree navigation이 유리**
→ 이 카테고리들은 특정 시간대의 상세 정보가 필요하므로 leaf 10개의 rich context가 효과적

**Flat > Best (전체 맥락이 중요한 영역):**

| 카테고리 | N | Best | Flat | Δ | 이유 |
|---------|---|------|------|---|------|
| Information Synopsis | 163 | 69.9% | **76.7%** | +6.7%p | 전체 흐름 파악 필요 |
| Object Reasoning | 240 | 49.2% | **50.8%** | +1.7%p | 산발적 정보 종합 |
| Action Reasoning | 180 | 48.3% | **49.4%** | +1.1%p | 전체 맥락 기반 추론 |

→ **전체 맥락을 종합해야 하는 문제에서는 flat dump가 압도적**
→ 부분 탐색으로는 전체 스토리라인을 놓침

### 6.8 구조적 시사점

**현재 시스템의 성능 구조:**
- Phase 0 / Flat: 전체 summary 1회 판단 → **54.8%** (성능의 원천)
- Hop loop: tree 기반 iterative 탐색 → **39.4%** (성능의 병목)
- Tree 구축 (Stage 2): summary/KE/caption 생성 → **필수** (이게 없으면 Flat도 못 함)

**핵심 인사이트:**
- Tree를 **만드는 것**은 좋은데, tree를 **타고 다니는 것**이 문제
- Hop loop이 Phase 0보다 못한 이유: "어디를 봐야 하는지" navigation이 불정확
  → 매 hop마다 10개 leaf의 rich context를 주지만, **엉뚱한 곳을 보면 소용없음**
- 다만 Counting/Temporal에서는 tree navigation이 유리 → **targeted search가 필요한 문제에 한해 hierarchical이 효과적**

**개선 방향:**
1. Hop loop navigation 정확도 개선 (현재 39.4% → 50%+ 필요)
   - 더 정확한 target selection / search direction
   - 첫 hop에서 정답 관련 leaf를 못 찾으면 이후 hop도 틀린 방향
2. Hybrid 전략: 문제 유형에 따라 Flat vs Hop 분기
   - Info Synopsis → Flat, Counting/Temporal → Hop
3. Phase 0 confidence tuning: medium에서도 early return → hop loop 진입 비율 ↓

### 6.9 Hop Loop 실패 사례 Tree Reconstruction 분석

**목적**: Hop loop이 실패하는 케이스에서, tree filter가 선택한 leaf 10개가 실제로 정답 근거가 되는 구간인지 시각적으로 검증.

**방법**: Best config에서 hop loop 진입 후 오답인 사례 중, Counting / Temporal Reasoning / Object Reasoning / Action Reasoning 4개 카테고리에서 각 2개씩 총 8개 샘플을 추출. 원본 memory tree (`stage2_30sec_no_window`)에서 전체 leaf를 나열하고, hop1에서 선택된 10개를 `>>>` 마킹.

**분석 대상 8개 샘플:**

| Category | QID | Video | Total Leaves | Duration | Selected Coverage | Scatter (avg gap) |
|----------|-----|-------|:---:|:---:|:---:|:---:|
| Counting | 631-2 | pU_yyadYgG8 | 72 | 35m41s | 300s (14.0%) | 227s |
| Counting | 747-2 | 4qYqPmIO0v0 | 110 | 54m34s | 300s (9.2%) | 253s |
| Temporal | 864-1 | VFntoBRGF1A | 81 | 40m17s | 300s (12.4%) | 217s |
| Temporal | 816-2 | Z2G9bTvffAw | 64 | 31m56s | 300s (15.7%) | 43s |
| Object | 803-3 | eQGSbBANfVg | 87 | 43m19s | 300s (11.5%) | 93s |
| Object | 850-1 | 5Z2_uRf7ckY | 105 | 52m9s | 300s (9.6%) | 77s |
| Action | 691-2 | lMxFbRc3Luk | 88 | 43m55s | 300s (11.4%) | 213s |
| Action | 807-1 | dLjOLXmu68M | 62 | 30m39s | 300s (16.3%) | 177s |

**관찰 결과:**

1. **고정 10개 leaf = 300s 커버리지**: 모든 샘플에서 hop1은 정확히 10개 leaf (30s × 10 = 300s)를 선택. 30~55분 영상에서 5~10분만 보는 셈 → 정답 구간을 놓칠 확률이 구조적으로 높음.

2. **Counting 문제의 산발적 선택** (QID=631-2):
   - "Joeri가 1990년대 상황 설명 시 몇 번 등장?" → Joeri의 반복 등장 구간을 세야 함
   - 선택된 10개 leaf가 [30s, 60s, 180s, 480s, 1050s, 1080s, 1350s, 1380s, 1410s, 2070s]로 **전체 35분에 걸쳐 산발 분포**
   - max_gap=660s → 연속된 출현 패턴을 추적하기 불가능
   - Counting에는 "반복 구간 연속 관찰"이 필요한데 tree filter는 keyword match 기반이라 관련 leaf를 흩어져 선택

3. **Counting 문제의 국소 집중** (QID=747-2):
   - "LIT의 골 수?" → 초반 경기 장면에 집중 (0~210s에 6개), 이후 2220s~2310s에 2개
   - 경기 후반(3274s 중 660s~2220s 구간)을 거의 skip → 후반 골을 놓침
   - max_gap=1560s (26분 공백)

4. **Temporal 문제의 양극화** (QID=816-2 vs 864-1):
   - 816-2: 선택이 초반 0~420s에 몰려 avg_gap=43s → 비교적 집중적이나 전체 32분 중 초반 7분만 봄
   - 864-1: 10개가 전체 40분에 산발 (avg_gap=217s) → 시간순 추론에 필요한 연속성 부재

5. **Object/Action 문제**: avg_gap 77~213s로 중간 정도의 산포. 특정 장면에 대한 reasoning이 필요한데 관련 없는 leaf가 섞여 LLM에 noise 제공.

**핵심 문제 진단:**

| 문제 | 원인 | 영향 |
|------|------|------|
| 고정 10개 leaf 제한 | 30s × 10 = 300s만 관찰 | 30~55분 영상의 9~16%만 커버 |
| Keyword 기반 tree filter | 질문 관련 키워드 매칭으로 leaf 선택 | 의미적으로 중요한 구간 vs 표면적 키워드 일치 괴리 |
| 산발적 선택 패턴 | 관련 leaf가 타임라인 전체에 퍼짐 | Counting/Temporal에서 연속 관찰 불가 |
| 초반 편향 | 초기 장면이 질문 키워드와 매칭되기 쉬움 | 후반부 정답 구간 누락 |

**vs Flat Baseline과의 대비:**
- Flat: 전체 72~110개 leaf summary를 한 번에 입력 → 모든 구간 커버
- Hop: 10개 leaf의 full caption + KE + history → 적은 구간을 자세히 보지만 **어디를 볼지** 결정이 부정확
- **결론: "좁고 깊게" < "넓고 얕게"**, 현재 navigation이 정확하지 않기 때문

**tree reconstruction 원본**: `/home/youngbeom/.claude/projects/-lustre-youngbeom/c40a5b5d-5022-4101-8041-7474d3b67210/tool-results/b255feb.txt` (8개 샘플, 전체 leaf listing)

---

## 7. 약점 분석 & 향후 방향

### 지는 카테고리 (G1 < VideoLucy, N≥10)

| Task Type | N | G1 | VL | Gap | 분석 |
|-----------|---|-----|-----|-----|------|
| **Information Synopsis** | 163 | 69.9% | 74.2% | **-4.3%p** | P0 70.4%로 선방하나 VL Phase2 91% 정확도에 못 미침 |
| **Object Reasoning** | 240 | 49.2% | 51.7% | **-2.5%p** | P0 51.2%, hop 32.0%. Hop loop 개선 여지 |
| Spatial Reasoning | 11 | 72.7% | 81.8% | -9.1% | N 작음 |

### 구조적 약점

| 약점 | 원인 | 현재 대응 | 잔여 gap |
|------|------|-----------|----------|
| **Hop loop 낮은 정확도** | Phase 0 low conf 문제는 어려운 문제인데, hop이 해결력 부족 | 미대응 | 39.4% (94건) |
| **Information Synopsis** | P0로 축소했으나 VL Phase2의 91% 정확도에 미달 | Phase 0 (G1) | -4.3%p vs VL |
| **Object Reasoning** | P0 51.2%, hop 32.0%. tree summary의 한계 | 미대응 | -2.5%p vs VL |
| **Root summary 부실** | Level_4가 마지막 merge만 반영 | Phase 0가 Level_1~3 전체 사용 | 해결됨 |

### R9: D5 Visual Context Enrichment 결과 (130문제 subset)

**설계**: VisualJudge 대신, hop loop에서 C3 text judge가 low/medium일 때 → frame caption → context 추가 → C3 재판단. C3 prompt 유지.

| | G1 (C3+D0+F1+G1) | **D5 (C3+D5+F1+G1)** | VideoLucy |
|---|---|---|---|
| Accuracy | 46.9% | **46.9% (+0.0%p)** | 46.2% |
| Hop loop | 3/8 (37.5%) | 3/8 (37.5%) | — |
| Visual 답변 변경 | — | 4/8 (50%) | — |

**Hop loop 8건 상세:**

| QID | Task Type | G1→D5 | 결과 |
|-----|-----------|-------|------|
| 633-3 | Info Synopsis | D→B(=GT) | X→O (visual이 정답으로 교정) |
| 816-2 | Temporal Reasoning | D→B(=GT) | X→O (visual이 정답으로 교정) |
| 710-2 | Action Reasoning | A(=GT)→D | O→X (visual이 오답으로 변경) |
| 722-3 | Object Reasoning | D(=GT)→A | O→X (visual이 오답으로 변경) |

**분석**: Visual enrichment는 동작하지만 (8/8 적용, 4/8 답변 변경), **+2 -2 = 넷제로**. Subset hop loop이 8건뿐이라 통계적으로 판단 어려움. Full run 결과 → Section 4.4 참조: 94건 hop loop에서 -3.2%p (39.4%→36.2%), overall -0.3%p (53.2%→52.9%). **Selective visual enrichment 필요** 확인됨.

### 향후 실험 후보

| 실험 | 설명 | 기대 효과 |
|------|------|-----------|
| ~~**D5 full run**~~ | ~~C3+D5+F1+G1로 900문제 full 실행~~ | ~~hop loop 94건에서 D5 효과 통계적 검증~~ → **완료** (Section 4.4). -0.3%p, selective enrichment 필요 |
| **Phase 0 confidence tuning** | medium에서도 early return 허용 | early return 비율/정확도 트레이드오프 |
| **VLM caption prompt 개선** | observation 대신 structured caption | visual 정보 질 향상 |

---

## 8. R10: Hierarchical Structure Ablation (900Q)

### 실험 목적

리뷰어 질문 대비: **"성능 향상이 hierarchical tree 구조 덕인가, key_elements 같은 rich representation 덕인가?"**

기존 A-G 축은 solver 파이프라인 컴포넌트 ablation.
R10은 **memory representation 자체**와 **hierarchical navigation의 가치**를 직접 검증.

핵심 인사이트: raw caption 없이 summary만 쓰면 전체 메모리를 한번에 넣을 수 있음 → 트리 탐색 자체가 필요한가?

### R10 실험 3종

| ID | 실험명 | Config | 제거 대상 | 검증 포인트 |
|----|--------|--------|-----------|-------------|
| R10a | Flat Baseline | `r10a_flat_baseline` | 트리 탐색 전체 | hierarchical navigation vs flat dump |
| R10b | No Key Elements | `r10b_no_key_elements` | key_elements (solver level) | structured index의 retrieval 기여 |
| R10c | No Caption | `r10c_no_caption` | raw caption | 상세 caption vs summary만 |

### R10a: Flat Baseline 설계

| | Phase 0 (G1) | Flat Baseline (R10a) |
|--|--|--|
| 입력 | Level_N~Level_1 모든 레벨 summary + KE brief | Level_1 children(leaf) summary만 |
| 계층 정보 | 있음 (Level_3 → Level_2 → Level_1) | 없음 (flat time-ordered list) |
| 대상 | 트리 노드 blended summary | 원본 leaf summary |
| 후속 | early return 실패 시 hop loop | 단독 최종 답 |

Phase 0과 Flat의 차이로 **hierarchical blending의 가치** vs **단순 summary 다량 투입**을 구분 가능.

### R10b/c: Memory Representation Ablation

- **R10b (-KE)**: tree_filter, hierarchical_scorer, context_assembler, semantic_matcher 4곳에서 key_elements 무시. summary/caption text 매칭 fallback 유지.
- **R10c (-Caption)**: context_assembler에서 raw caption 제거. 트리 탐색/필터링은 그대로(KE+summary로 매칭). Judge에 summary만 전달.

### 예상 결과 시나리오

| 결과 | 의미 |
|------|------|
| Flat << Best | **Hierarchical navigation이 핵심 기여자** |
| Flat ≈ Best | 트리 구조 불필요, summary dump로 충분 |
| -KE << Best | **Structured index가 retrieval에 중요** |
| -KE ≈ Best | KE 없어도 summary 매칭으로 충분 |
| -Caption << Best | **상세 caption이 judge 판단에 필수** |
| -Caption ≈ Best | Summary만으로 충분 |

### 수정 파일

- `pipelines/tree_search.py`: flat_baseline 모드 + `_flat_baseline_context()`, `_forced_answer()` 메서드
- `components/context_assembler.py`: `use_captions`, `use_key_elements` 플래그
- `components/tree_filter.py`: `use_key_elements` 플래그
- `components/hierarchical_scorer.py`: `use_key_elements` 플래그
- `components/semantic_matcher.py`: `use_key_elements` 플래그
- `solver.py`: pipeline_params → 컴포넌트 전달
- `config/full_run/r10a_flat_baseline.yaml`, `r10b_no_key_elements.yaml`, `r10c_no_caption.yaml`
- `run_full_r10.sh`: 3종 제출 스크립트

### R10 결과

| Method | N | Accuracy | vs Best | Status |
|--------|---|----------|---------|--------|
| **R10a (Flat Baseline)** | 900 | **54.8%** | **+1.6%p** | ✅ |
| **R10b (No Key Elements)** | 900 | **53.4%** | **+0.2%p** | ✅ |
| **R10c (No Caption)** | 900 | **53.4%** | **+0.2%p** | ✅ |
| Best (A1+B1+C3+D0+F1+G1) | 900 | 53.2% | — | ✅ |

**결론**: 3개 R10 실험 모두 Best 이상. **Hierarchical tree 구조가 성능에 기여하지 않음** 확인.
- Flat(54.8%) > Best(53.2%): 트리 탐색 불필요
- -KE(53.4%) ≥ Best(53.2%): key_elements 불필요
- -Caption(53.4%) ≥ Best(53.2%): raw caption 불필요

---

## 9. A2 Recovery Cue — Hop Loop 개선 실험

### 9.1 문제 진단

Phase 0(G1)이 전체 성능의 핵심 (806/900 = 89.6% early return). 나머지 94개가 hop loop에 진입하지만, hop loop 정확도는 39.4% (37/94)에 불과.

**Hop loop 실패 원인 분석 (Section 6.9 참고)**:
- keyword cues로 exact match → 의미적으로 중요한 구간을 못 찾음
- 고정 10개 leaf = 9~16% 커버리지 → 너무 적음
- cues는 최초 1회 추출 후 hop loop 내에서 갱신/전달 안 됨

### 9.2 인지과학적 근거

| 이론 | 핵심 | 현재 구현 | 갭 |
|------|------|-----------|-----|
| **SAM (Search of Associative Memory)** | 상위 context cue → sampling → recovery(새 cue 생성) → 재탐색 | cue 1회 추출 → match → 끝 | Recovery 단계 없음 |
| **Hippocampal Indexing Theory** | cue → 해마 인덱스 활성화 → 하위 감각 데이터 재구성 | key_elements = 인덱스, caption = 감각 데이터 | 인덱스 매칭이 exact substring → 의미적 매칭 아님 |
| **Spreading Activation** | 상위 개념 활성화 → 연결 강도 따라 하위 전파 | hop에서 direction만으로 이동 | 활성화 강도 개념 없음 |

**핵심 갭**: cue와 retrieval 사이에 **추론(상상) 단계**가 없음. 사람은 키워드 하나로 연상해가며 시간 구간을 특정하지만, 현재 시스템은 keyword → exact match → 끝.

### 9.3 A2 설계: Recovery Cue

**SAM의 "Sampling → Recovery → 새 Cue" 사이클을 1회 구현**

```
A1(cue 추출) → G1(Phase 0, confidence 낮음) → A2(recovery cue) → tree leaf 재선정 → hop loop
```

Phase 0에서 coarse memory를 이미 본 LLM이, "이 질문에 답하려면 어떤 시간대를 더 자세히 봐야 하는지" 추론.

#### A2 vs 기존 A1 비교

| | A1 (현재) | A2 (Recovery Cue) |
|--|--|--|
| **입력** | 질문만 | 질문 + coarse memory + Phase 0 reasoning |
| **출력** | 키워드 3~5개 | 시간대 3개 + focus points |
| **언제** | Pipeline 시작 시 | Phase 0 이후, hop 진입 전 |
| **tree filter** | keyword exact match | 시간대 기반 leaf 선정 |

#### 프롬프트 핵심

```
이전 시도에서 confident하지 못했습니다.
Previous reasoning: {phase0_reasoning}

어떤 시간대를 더 자세히 봐야 이 질문에 답할 수 있을까요?
→ time_periods: [[start1, end1], ...] (최대 3개)
→ focus_points: "해당 구간에서 무엇을 확인해야 하는지"
```

### 9.4 구현 내역

| 파일 | 변경 | 설명 |
|------|------|------|
| `pipelines/tree_search.py` | STAGE 2.7 추가 (Phase 0 이후) | `_recovery_cue()` 메서드 + `RECOVERY_CUE_PROMPT` |
| `pipelines/tree_search.py` | STAGE 3 수정 | recovery_targets가 있으면 그것을 initial targets로 사용 |
| `config/full_run/a2_recovery_cue.yaml` | 신규 | `recovery_cue: true` 추가, `question_path: subset_for_hop_optimize/` |
| `output/a2_subset_qid_list.tsv` | 신규 | 94개 hop loop 질문 목록 |
| `run_a2_subset.sh` | 신규 | subset 실행 스크립트 |

### 9.5 파이프라인 흐름 변화

```
[기존 Best]
질문 → A1(cues) → tree_filter(keyword match) → Phase 0(G1)
  → confidence high → early return (806개)
  → confidence low → hop loop with keyword-matched leaves (94개, 39.4%)

[A2 Recovery Cue]
질문 → A1(cues) → tree_filter(keyword match) → Phase 0(G1)
  → confidence high → early return (806개)
  → confidence low → A2: LLM이 coarse memory 보고 시간대 추론
                   → 시간대 기반 leaf 재선정
                   → hop loop with LLM-guided leaves (94개)
```

### 9.6 실험 설계

- **데이터**: 94개 hop loop 진입 질문 (75 videos)
- **비교 대상**: Best config의 hop loop 결과 (37/94 = 39.4%)
- **측정**: 전체 정확도 + recovery cue 효과 분석
- **비용**: LLM 콜 1회 추가 (Phase 0 이후 recovery cue 생성)

### 9.7 A2 v1 결과 (94/94 완료)

**Overall:**

| Method | N | Correct | Accuracy | vs Best |
|--------|---|---------|----------|---------|
| Best (hop loop) | 94 | 37 | 39.4% | — |
| **A2 v1 Recovery Cue** | 94 | 34 | **36.2%** | **-3.2%p** |

**A2가 Best보다 -3.2%p 하락.** Recovery cue가 오히려 성능을 떨어뜨림.

**변동 분석:**

| | N |
|--|---|
| 둘 다 정답 | 31 |
| 둘 다 오답 | 54 |
| Best✗ → A2✓ (A2 gain) | 3 |
| Best✓ → A2✗ (A2 loss) | 6 |
| **Net** | **-3** |

**Recovery Cue 효과 분석:**

| 구분 | N | Correct | Accuracy |
|------|---|---------|----------|
| Recovery cue 생성됨 | 47 | 16 | 34.0% |
| Recovery cue 없음 (fallback) | 47 | 18 | 38.3% |
| └─ 넓은 범위 (>1000s) | 27 | 8 | 29.6% |
| └─ 좁은 범위 (≤1000s) | 20 | 8 | 40.0% |

**핵심 발견:**
- Recovery cue가 **생성된 경우(34.0%)가 없는 경우(38.3%)보다 오히려 낮음**
- 47/94에서 recovery cue 자체가 안 생김 (Phase 0의 prev_reasoning이 비어있어 LLM이 시간대를 추론 못함)
- 27/47에서 >1000s 범위 선택 (29.6%) → **Level_1 기반 coarse context의 한계**
- 좁은 범위(≤1000s) 선택 시 40.0%로 Best(39.4%)과 유사 → **정밀한 시간대 선택이 되면 가능성 있음**

### 9.8 A2 v1 실패 원인 분석

| 원인 | 상세 | 영향 |
|------|------|------|
| **Coarse context 문제** | A2에 입력되는 coarse_ctx에 Level_1 시간 범위만 있음 (~500s 단위). LLM이 30초 단위 leaf를 선택할 수 없음 | 27/45에서 >1000s 범위 선택 → 사실상 "전부 봐라" |
| **Phase 0 reasoning 부재** | Phase 0에서 `answerable=False`일 때 prev_reasoning이 빈 문자열 | 47/94에서 recovery cue 미생성 |
| **A2 입력 = Phase 0 입력** | A2 프롬프트에 Phase 0과 동일한 coarse_ctx를 넣음. 같은 입력으로 다른 결과를 기대 | 실질적으로 Phase 0의 재실행에 불과 |

### 9.9 A2 v2 결과 (leaf_ctx, 94/94 완료)

**변경**: A2에 Level_0 leaf summary를 투입 (`_flat_baseline_context()` 형태, 30초 단위)

| 실험 | 정답 | 정답률 | vs Best |
|------|------|--------|---------|
| **Best baseline** | 37/94 | 39.4% | — |
| **A2 v1** (coarse, merged time) | 34/94 | 36.2% | -3.2%p |
| **A2 v2** (leaf_ctx) | **40/94** | **42.6%** | **+3.2%p** |

**Recovery Cue 효과:**
| 구분 | 정답 | 정답률 |
|------|------|--------|
| With cue (53건) | 25 | **47.2%** |
| Without cue (41건) | 15 | 36.6% |

**Method 분포:**
| method | 정답/전체 | 비율 |
|--------|----------|------|
| phase0_coarse | 4/10 | 40.0% |
| hop2 | 2/2 | 100.0% |
| hop3 | 0/1 | 0.0% |
| hop4 | 2/6 | 33.3% |
| hop5 | 31/71 | 43.7% |
| fallback | 1/4 | 25.0% |

**v1→v2 변화**: gained +9, lost -3, net **+6문제**

**핵심 발견**:
- leaf_ctx로 바꾸니 recovery cue가 53건에서 생성 (v1: 47건) — 30초 단위로 정확한 시간 지목 가능해짐
- With cue(47.2%) vs Without cue(36.6%) = **+10.6%p** — recovery cue 자체의 효과 확인
- **Best(39.4%) 대비 +3.2%p 개선** — A2가 hop loop 성능을 실제로 올림
- 여전히 hop5 71건 = 대부분 max hop 도달 → navigate 개선 여지 큼

### 9.10 A2 v3 결과 (coarse_ctx + 개별 시간, 94/94 완료)

**변경**: A2 context를 coarse_ctx로 쓰되, 시간 범위를 `[30s-2370s]` 대신 `[30s-60s, 60s-90s, ..., 2340s-2370s]`로 개별 표시

| 버전 | context | 시간 표현 | 정답률 | vs v1 |
|------|---------|-----------|--------|-------|
| **v1** | coarse | 합침 | 36.2% | — |
| **v3** | coarse | 개별 | **41.5%** | +5.3%p |
| **v2** | leaf | 개별(자동) | **42.6%** | +6.4%p |

- v3 recovery cue: 15건 생성 (v2: 53건) — coarse context에선 cue 생성률 낮음
- v3 vs v2 차이 = 1.1%p — **시간 granularity가 핵심**, context 종류 차이는 미미
- v1→v3: gained +8, lost -3, net **+5**

**결론**: v1 실패의 주 원인은 "coarse여서"가 아니라 **"시간을 합쳐서"**. 시간만 개별 표기하면 coarse든 leaf든 유사한 성능

### 9.11 Structured A1 Ablation (STAGE 1만 변경)

**변경**: STAGE 1에서 `analyze_structured()` 호출 → 카테고리별 phrase-level cues 추출 → flatten → 기존 flat `build()`에 사용
나머지 (STAGE 2 flat filter, A2 v3 recovery, Phase 0 등) 전부 동일.

**Config**: `structured_a1: true`, `recovery_cue: true`, `recovery_cue_context: coarse`

| Config | 정답률 | vs A2v3 |
|--------|--------|---------|
| A2 v3 baseline | 39/94 = 41.5% | — |
| **Structured A1** | **40/94 = 42.6%** | **+1.1%p** |

**Flip 분석** (94문제):
- Both correct: 37
- SA1 gained: 3 (691-2, 705-1, 859-1)
- SA1 lost: 2 (745-1, 806-2)
- Both wrong: 52
- **Net flip: +1**

**Active leaves**: SA1 36.1 (42.3%) vs A2v3 48.3 (56.6%) — **structured cues가 활성화 비율 14.3%p 줄임**

**Recovery**: 동일하게 15/94 questions에서 recovery hop 사용 (완전히 동일한 15문제)

**Hop target source 비교**:
| Source | SA1 | A2v3 |
|--------|-----|------|
| priority (filtered) | 58 | 64 |
| navigation | 322 | 322 |
| recovery_cue | 15 | 15 |
| all_fallback | 11 | 5 |

**분석**:
1. Structured cues가 active leaves를 57%→42%로 **의미있게 줄임** (더 정밀한 필터링)
2. 하지만 정답률 차이는 미미 (+1.1%p, net +1문제)
3. 5개 flip 전부 recovery 미사용, max hop(5) 도달 — **hop loop의 navigation 경로가 달라진 결과**
4. Cue가 달라진 39/94 문제 중 5개만 flip → cue 차이가 navigation에 미치는 영향 제한적
5. **priority source가 줄고 all_fallback이 늘음** — stricter filter가 일부 문제에서 over-filtering

**결론**: Structured A1 cue 추출은 필터를 개선하지만, 현재 파이프라인에서 hop loop이 어차피 navigation으로 탐색하므로 최종 정답률 영향은 미미. **STAGE 2 filter 자체를 structured로 바꾸거나, SAM recovery와 결합해야 효과가 나타날 것.**

### 9.12 Structured S2 + SAM Full Cycle (STAGE 2 filter + 전체 변경)

**Structured S2**: STAGE 2에서 `build_structured(min_cat=2)` 사용, A2는 기존 recovery_cue
**SAM Full Cycle**: structured A1 + structured S2 filter + SAM recovery (전부 변경)

| Config | 정답률 | vs A2v3 | Net flip |
|--------|--------|---------|----------|
| A2 v3 baseline | 39/94 = 41.5% | — | — |
| Structured A1 | 40/94 = 42.6% | +1.1%p | +1 |
| **Structured S2** | **39/94 = 41.5%** | **0** | **0** |
| **SAM full cycle** | **36/94 = 38.3%** | **-3.2%p** | **-3** |

**SAM Flip**: gained 3 (660-2, 742-2, 774-2), lost 6 (644-3, 710-2, 722-3, 745-1, 775-1, 800-1)

**SAM Target Source 비교**:
| Source | A2v3 | SAM |
|--------|------|-----|
| priority | 64 | 17 |
| navigation | 322 | 324 |
| recovery_cue | 15 | **66** |
| all_fallback | 5 | 1 |

**핵심 발견**:
1. **Structured S2 = A2v3 와 완전 동일** (0 flips) — filter 변경만으로는 효과 없음
2. **SAM full = -3.2%p 하락** — recovery가 66/94에서 발동 (A2v3: 15/94), 과도한 recovery가 오히려 해로움
3. SAM recovery가 structured cues로 tree를 재필터 → 잘못된 구간으로 유도하는 경우 많음
4. 시간 기반 A2v3 recovery(15건만 발동)가 SAM recovery(66건 발동)보다 정밀

**결론**: Text-based cue 검색 방향의 개선은 한계 도달. Phase 0 + navigation 지배 구조에서 cue/filter/recovery 변경은 모두 미미하거나 역효과. **Visual 정보 활용으로 방향 전환 필요.**

### 9.13 VideoLucy Merged Baseline 업데이트

원래 VideoLucy 53개 에러/실패 재실행 결과:
| | Correct | Total | Accuracy |
|--|---------|-------|----------|
| Original | 467 | 900 | 51.89% |
| **Merged** | **489** | **900** | **54.33%** |

+22개 개선, 0개 degraded. **새 baseline = 54.33%**
→ Best tree search(53.2%)가 baseline보다 **낮아짐**
→ R10a flat(54.8%)도 baseline 대비 +0.47%p (noise 수준)

---

## 10. Tree-Guided Visual Search (TGVS) — Visual 정보 활용 실험

### 10.1 배경 및 핵심 발견

**문제 진단**: 기존 실험에서 leaf raw_caption(평균 3000자)을 답변 context에 거의 사용하지 않았음.
- Phase 0 (`_phase0_coarse_answer`): **summary만** 사용 (주석: "Leaf caption은 제외")
- R10a Flat (`_flat_baseline_context`): **summary만** (`child.get("summary", "")`)
- Hop loop (`context_assembler`): caption 포함 (`use_captions=True` 기본값), 하지만 localization 실패로 효과 없음
- **Summary 160자 vs Caption 3000자** — 가장 정보량 풍부한 데이터를 1/20 압축본만 사용

**핵심 인사이트**:
1. 쉬운 문제: summary만으로 충분 (Phase 0 = 54.8%)
2. 어려운 문제: **정확한 localization + raw caption** 이 필요
3. Hop loop 실패 원인: caption이 없어서가 아니라 **잘못된 구간의 caption**을 보여줌
4. 우리 강점: tree 기반 time localization (key_elements, 계층 구조)

### 10.2 TGVS 파이프라인 설계

```
Phase 1 (Text Coarse)
  └─ Hierarchical/Flat summary → forced answer
  └─ confidence ≥ medium → 종료

Phase 2 (Tree-Guided Visual Search, max N iterations)
  ├─ [Localize] 관련 구간 찾기
  │   ├─ key_elements: TreeFilter 매칭
  │   ├─ llm_select: LLM이 coarse context에서 시간대 선택
  │   └─ combined: key_elements → fallback llm_select
  ├─ [Leaf Caption] 선택 구간 leaf raw_caption 추출 (3000자급)
  ├─ [VLM Observe] TargetedFrameLoader + VisionVLM 프레임 관찰
  ├─ [Re-judge] coarse + leaf captions + VLM observations → 재판단
  └─ confidence high → 종료

Phase 3 (Forced Answer)
  └─ 모든 축적 context → 강제 답변
```

**구현**: `pipelines/tree_visual.py` (신규), `solver.py` tree_visual 분기 추가

### 10.3 버그 발견 및 수정

**Config 전달 버그**: `solver.py`가 `pipe_config = {**pipeline_params, ...}`로 flatten하여 전달하는데, `TreeVisualPipeline.__init__`에서 `config.get("pipeline_params", {})`로 다시 찾아서 항상 빈 dict → **모든 설정이 기본값**으로 실행됨.

결과: E1~E6 전부 동일 설정(key_elements, hierarchical, direct, 1 iteration)으로 실행됨.
수정: `config.get("pipeline_params", {})` → `config.get(...)` 직접 접근으로 변경.

### 10.4 E1 결과 (유일한 유효 결과, key_elements + VLM direct)

| Metric | Value |
|--------|-------|
| **정확도** | **34/94 = 36.2%** |
| Phase 1 (coarse) | 13/35 = 37.1% (35개 confident) |
| Phase 2 (visual) | 21/54 = 38.9% (54개 visual search) |
| Phase 3 (forced) | 0/5 = 0% |

**비교 (94Q hop-hard subset)**:

| 실험 | 정답률 |
|------|--------|
| Flat Baseline (R10a) | 30/94 = 31.9% |
| VLS (VideoLucy-style, text only) | 31/94 = 33.0% |
| VideoLucy (원본) | 34/94 = 36.2% |
| **TGVS E1** | **34/94 = 36.2%** |
| VideoLucy (merged) | 36/94 = 38.3% |
| Best Hop (G1) | 37/94 = 39.4% |

**주의**: E1은 leaf caption 코드 추가 전에 실행되어 caption 미포함 상태의 결과.

### 10.5 E1~E6 최종 결과

Config 버그 수정 + leaf caption 추가 후 전체 결과 (94Q hop-hard subset):

| # | Coarse | Localize | VLM | Frames | Iter | **정답률** |
|---|--------|----------|-----|--------|------|------------|
| E1 | hierarchical | key_elements | direct | 16 | 1 | 34/94 = 36.2% |
| E2 | **flat** | key_elements | direct | 16 | 1 | 31/94 = 33.0% |
| E3 | hierarchical | **llm_select** | direct | 16 | 1 | 14/94 = 14.9% ⚠️ |
| **E4** | hierarchical | key_elements | **caption** | 16 | 1 | **38/94 = 40.4%** ✓ |
| E5 | hierarchical | **combined** | direct | 16 | **2** | 35/94 = 37.2% |
| E6 | hierarchical | **combined** | **caption** | **32** | **3** | 32/94 = 34.0% |

**Phase별 상세**:

| Exp | Phase1 (coarse) | Phase2 (visual) | Phase3 (forced) |
|-----|-----------------|-----------------|-----------------|
| E1 | 13/35 = 37.1% | 21/54 = 38.9% | 0/5 = 0% |
| E2 | 21/59 = 35.6% | 10/30 = 33.3% | 0/5 = 0% |
| E3 | 13/35 = 37.1% | 0/1 = 0% | 1/58 = 1.7% |
| **E4** | 13/35 = 37.1% | **25/54 = 46.3%** | 0/5 = 0% |
| E5 | 13/35 = 37.1% | 22/56 = 39.3% | 0/3 = 0% |
| E6 | 13/35 = 37.1% | 19/59 = 32.2% | 0/1 = 0% |

**비교 (94Q hop-hard subset 전체)**:

| 실험 | 정답률 |
|------|--------|
| E3 llm_select (버그) | 14/94 = 14.9% |
| Flat Baseline (R10a) | 30/94 = 31.9% |
| E2 flat_coarse | 31/94 = 33.0% |
| VLS text-only | 31/94 = 33.0% |
| E6 heavy | 32/94 = 34.0% |
| E1 key_elements+direct | 34/94 = 36.2% |
| VideoLucy (원본) | 34/94 = 36.2% |
| E5 multi_hop | 35/94 = 37.2% |
| VideoLucy (merged) | 36/94 = 38.3% |
| Best Hop (G1) | 37/94 = 39.4% |
| **E4 key_elements+caption** | **38/94 = 40.4%** |

### 10.6 핵심 발견 — VLM B-Bias 문제

**VLM Direct 모드의 치명적 position bias**:
- Qwen3-VL-8B가 프레임을 보고 직접 답변할 때, **88.1% (52/59)가 "B"를 선택**
- GT에서 B 비율은 33% (31/94)
- 원인: `components/vlm.py` line 90의 prompt 예시가 `"answer": "B"`로 하드코딩

| 모드 | A | B | C | D | B% |
|------|---|---|---|---|-----|
| Ground Truth | 28 | 31 | 20 | 15 | 33.0% |
| E1 VLM Direct | 2 | **52** | 2 | 3 | **88.1%** |
| E1 Re-judge | 2 | **45** | 3 | 3 | **84.9%** |
| E4 Re-judge | 5 | 36 | 7 | 5 | **61.0%** |

→ Caption 모드는 VLM이 답변 없이 설명만 → LLM 재판단 시 B-bias 85%→61%로 감소
→ 그래도 여전히 61%로 bias 존재 (GT 33%)

### 10.7 E4 (Caption) 성공 분석

**E4 vs E1 Flip 분석** (동일 Phase1, 차이는 Phase2에서만):

| QID | E1 Pred | E4 Pred | GT | 유형 |
|-----|---------|---------|-----|------|
| 631-2 | B ❌ | **A** ✓ | A | Counting |
| 719-3 | B ❌ | **A** ✓ | A | Information |
| 742-2 | B ❌ | **C** ✓ | C | Object Reasoning |
| 839-1 | B ❌ | **C** ✓ | C | OCR |
| 884-1 | None ❌ | **C** ✓ | C | Counting |
| 894-1 | B ❌ | **C** ✓ | C | Information |

6개 gain 모두 **E1이 B-bias로 틀린 것을 E4 caption이 교정**한 케이스.

| QID | E1 Pred | E4 Pred | GT | 유형 |
|-----|---------|---------|-----|------|
| 676-2 | B ✓ | D ❌ | B | Temporal |
| 779-1 | B ✓ | A ❌ | B | Temporal |

2개 loss는 GT가 B여서 **E1의 B-bias가 우연히 맞았던** 케이스.

**핵심 메커니즘**: Caption 모드는 관찰(VLM)과 판단(LLM)을 분리 → B-bias 차단 → net +4 (+4.3%p)

### 10.8 E3 (llm_select) 실패 분석

**근본 원인: Job 덮어쓰기 버그**
- SLURM Job 1074257 → 34/94 = 36.2% (먼저 완료)
- SLURM Job 1074686 → 14/94 = 14.9% (나중에 완료, 결과 덮어씀)
- `process_question()`에 caching guard 없음 (`run_video()`에는 있음)

**Job2 실패 원인**: 58/94에서 `_localize_llm_select()`가 빈 interval 반환
- LLM `reason()` 응답이 유효한 JSON 미생성 또는 `"periods"` 키 없음
- GPU 하드웨어 차이(4종 A6000)로 `do_sample=False`에도 비결정적 결과
- intervals=[] → VLM 미호출 → Phase 3 forced → 55/58이 pred=null

**수정 필요**: `process_question()`에 caching guard 추가, llm_select에 fallback 로직

### 10.9 E6 (Heavy) 기대 이하 분석

3가지 설계가 복합적으로 해로움:

1. **Multiple iterations 역효과**: 5개 multi-iter 중 1개만 정답 (20%)
   - 반복할수록 무관한 구간 탐색 → 노이즈 축적 → 오답 확신
   - E4가 iter1에서 맞춘 2개를 E6가 추가 iter에서 틀림

2. **32 frames > 16 frames 역효과**: VLM 캡션이 산만해짐
   - 더 많은 프레임 = 초점 분산, 핵심 디테일 누락

3. **Caption 정보 손실 누적**: Visual → Text 변환의 bottleneck이 반복마다 누적

**교훈**: "더 많이 보기"보다 "정확한 곳을 한번 잘 보기"가 중요

### 10.10 결론 및 Next Steps

**성과**:
- **E4 = 40.4%**: 94Q subset 역대 최고 (Best Hop 39.4%, VL merged 38.3% 초과)
- Caption 모드의 관찰/판단 분리가 VLM B-bias 문제를 효과적으로 완화
- Tree 기반 localization (key_elements)이 llm_select보다 안정적

**한계**:
- B-bias가 여전히 61% (GT 33%) — prompt 수정으로 추가 개선 가능
- Phase3 forced가 항상 0% — fallback 메커니즘 개선 필요
- E1은 leaf caption 미포함 상태 — 재실행 시 추가 개선 가능

**즉시 실행 가능 액션**:
1. VLM prompt의 `"answer": "B"` 예시 → 랜덤/제거하여 B-bias 수정
2. `process_question()` caching guard 추가
3. E4 config + B-bias 수정 후 재실험 → 40.4%+ 기대
4. llm_select에 key_elements fallback 추가

### 10.11 Full 900Q 실행 결과 (B-bias 수정 포함)

**코드 변경**: `vlm.py` 예시 `"answer": "B"` → `"answer": "<A, B, C, or D>"` (B-bias 수정)

| 실험 | Memory | 정답률 |
|------|--------|--------|
| TGVS Caption (old memory) | stage2_30sec_no_window | 483/900 = **53.67%** |
| TGVS Caption v9 | stage2_v9 | 468/900 = **52.00%** |

Phase별 (v9):
| Phase | 정답 | 비율 |
|-------|------|------|
| Phase 1 (coarse) | 441/834 | 52.9% |
| Phase 2 (visual) | 27/61 | 44.3% |
| Phase 3 (forced) | 0/5 | 0.0% |

v9 메모리(3레벨, Level_1 노드 많음)가 old memory(4레벨)보다 coarse context 품질이 다소 낮음.

### 10.12 LVBench Full Best 결과 (stage2_v9)

| Phase | 정답 | 비율 |
|-------|------|------|
| Phase 0 (coarse) | 547/1229 | 44.5% |
| Hop 1-2 | 12/14 | 85.7% |
| Hop 3-5 | 85/288 | 29.5% |
| **전체** | **644/1535** | **41.95%** |

14개 missing (GPU error/timeout). Hop 5까지 간 271개가 29.5%로 역시 deep hop 성능 저하.

---

## 11. Visual V2 — 대규모 Visual Search 실험 (Full 900Q, stage2_v9)

### 11.1 배경

E4 (caption) 접근법이 94Q subset에서 40.4%로 최고였으나, full 900Q에서는 52%로 기대 이하.
방향 전환: **VideoLucy/VideoTree 스타일 벤치마킹** — 더 많은 hop, 단일 구간 선택, agentic tool 선택 등 다양한 조합을 대규모로 실험.

### 11.2 코드 변경 (`pipelines/tree_visual.py` v2)

1. **`confidence_threshold: none`** — Phase 1 early return 없이 모든 문제를 Phase 2로
2. **`max_intervals: N`** — hop당 선택 구간 수 (1 = VideoLucy 스타일 단일 구간)
3. **`search_hint` 전달** — Phase 1의 `search_direction`/`missing_info`를 Phase 2 localization에 활용
4. **Agentic mode (`localize_mode: agentic`)** — LLM이 tool 선택:
   - `scene_browse`: 트리 계층 summary 탐색
   - `caption_search`: leaf raw caption 키워드 검색
   - `visual_inspect`: VLM으로 특정 구간 프레임 관찰
   - `answer`: 최종 답변 결정
   - Think-Act-Observe 루프 (최대 N steps)

### 11.3 실험 설계 (12개)

| # | 실험명 | Coarse | Localize | VLM | Hop | Intervals | Threshold | 핵심 가설 |
|---|--------|--------|----------|-----|-----|-----------|-----------|----------|
| V1 | direct_5hop | hier | key_elements | direct | 5 | 5 | low | B-bias 수정된 direct가 어떤지 |
| V2 | caption_5hop | hier | key_elements | caption | 5 | 5 | low | 5 hop caption 누적 효과 |
| V3 | caption_no_early | hier | key_elements | caption | 5 | 5 | none | 전부 Phase2로 보내면? |
| V4 | single_seg_caption | hier | key_elements | caption | 5 | **1** | low | VideoLucy 스타일 단일 구간 |
| V5 | single_seg_direct | hier | key_elements | direct | 5 | **1** | low | 단일 구간 + direct |
| V6 | llm_select_caption | hier | **llm_select** | caption | 5 | 3 | low | LLM 구간 선택 + caption |
| V7 | combined_caption | hier | **combined** | caption | 5 | 3 | low | key_elements→llm fallback |
| V8 | flat_caption_5hop | **flat** | key_elements | caption | 5 | 5 | low | flat coarse + visual |
| V9 | agentic | hier | **agentic** | caption | 5 | 3 | low | LLM tool 선택 |
| V10 | agentic_no_early | hier | **agentic** | caption | 5 | 3 | none | agentic 전체 |
| V11 | caption_32fr | hier | key_elements | caption | 5 | 5 | low | 32 frames |
| V12 | direct_no_early_single | hier | key_elements | direct | 5 | **1** | none | direct 전체 + 단일 구간 |

**공통**: stage2_v9 메모리, query_analyzer 사용, B-bias 수정된 VLM

### 11.4 V1-V12 결과 (Full 900Q)

| # | 실험명 | Accuracy | Phase Distribution | 비고 |
|---|--------|----------|-------------------|------|
| **V8** | flat_caption_5hop | **54.8%** | P1: 900 (100%) | **Best** — flat coarse가 최고 |
| V1 | direct_5hop | 51.4% | P1: 900 (100%) | threshold=low → Phase2 미진입 |
| V2 | caption_5hop | 51.4% | P1: 900 (100%) | V1과 동일 |
| V4 | single_seg_caption | 51.4% | P1: 900 (100%) | V1과 동일 |
| V5 | single_seg_direct | 51.4% | P1: 900 (100%) | V1과 동일 |
| V6 | llm_select_caption | 51.4% | P1: 900 (100%) | V1과 동일 |
| V7 | combined_caption | 51.4% | P1: 900 (100%) | V1과 동일 |
| V9 | agentic | 51.4% | P1: 900 (100%) | V1과 동일 |
| V11 | caption_32fr | 51.4% | P1: 900 (100%) | V1과 동일 |
| V12 | direct_no_early_single | 49.9% | P3: 900 (100%) | Phase2 전체 투입 → 약간 하락 |
| V3 | caption_no_early | 47.4% | P3: 900 (100%) | Phase2 전체 → 큰 하락 |
| V10 | agentic_no_early | 28.6% | Agent: 487, Crash: 413 | **치명적 — 413/900 crash** |

### 11.5 핵심 발견

**1. `confidence_threshold: low`는 사실상 early return 100%**
- 모든 confidence level(high/medium/low)이 threshold를 통과 → Phase 2가 **한 번도** 실행 안 됨
- V1-V7, V9, V11이 완전히 동일한 결과 (463/900 = 51.4%)
- 실험 차이(localize_mode, vlm_mode 등)가 전혀 발현되지 않음

**2. Phase 1 Confidence 분포 (hierarchical coarse)**
- high: 739 (53.9% acc) | medium: 24 (45.8%) | low: 137 (39.4%)
- Flat coarse (V8): high: 778 (56.8%) | medium: 15 | low: 107 (39.3%)

**3. Visual search는 양날의 검 (V3, V12 기반 분석)**
- Low-conf 질문: text 39.4% → visual 41.6% (+2.2%p) — **도움됨**
- High-conf 질문: text 53.9% → caption 49.0% / direct 52.1% — **오히려 악화**
- Net effect: V3은 -36문제, V12는 -14문제

**4. Oracle upper bound = 67.0%** (V8+V12 per-question best)
- Visual search가 text-only와 다른 문제를 맞추므로, 선택적 적용 시 +12.2%p 잠재력

**5. Agentic mode (V10) crash 원인**
- 413/900 crash: `time_range` 파싱 실패 (string/None → TypeError/IndexError)
- 정상 동작한 487건 중 step2-3은 56-62%로 양호, step4-5는 46-49%로 하락
- 원인: LLM이 tool+query+time_range를 동시에 JSON 출력 → 파싱 불안정

### 11.6 V13-V22 실험 추가 설계

| # | 실험명 | Coarse | Localize | VLM | Intervals | Threshold | 핵심 가설 |
|---|--------|--------|----------|-----|-----------|-----------|----------|
| V13 | llm_select_direct | hier | llm_select | direct | 3 | low | llm_select + direct |
| V14 | llm_select_single | hier | llm_select | caption | 1 | low | llm_select + 단일 구간 |
| V15 | llm_select_no_early | hier | llm_select | caption | 3 | none | llm_select 전체 |
| V16 | combined_single | hier | combined | caption | 1 | low | combined + 단일 구간 |
| V17 | combined_direct | hier | combined | direct | 3 | low | combined + direct |
| V18 | agentic_3step | hier | agentic | caption | 3 | low | agentic 3 step |
| V19 | agentic_7step | hier | agentic | caption | 3 | low | agentic 7 step |
| V20 | agentic_no_early_7step | hier | agentic | caption | 3 | none | agentic 7 step 전체 |
| V21 | agentic_flat | flat | agentic | caption | 3 | low | flat + agentic |
| V22 | agentic_32fr | hier | agentic | caption | 3 | low | agentic + 32fr |

- V13-V17: llm_select, combined 변형
- V18-V22: agentic 변형 (step수, flat, 32fr)
- **주의**: V13-V17은 threshold=low이므로 V1과 동일한 결과가 될 가능성이 높음

### 11.7 V23-V34 실험 추가 설계 (핵심: medium/high threshold)

분석 결과, `threshold: medium` 또는 `high`를 써야 Phase 2가 실제로 실행됨. 또한 `llm_index` 모드 추가 — LLM에게 시간 값 대신 segment 인덱스를 골라서 파싱 오류 방지.

| # | 실험명 | Coarse | Localize | VLM | Intervals | Threshold | 핵심 가설 |
|---|--------|--------|----------|-----|-----------|-----------|----------|
| V23 | flat_keyelm_caption_med | flat | key_elements | caption | 3 | **medium** | flat P1 + low만 Phase2 |
| V24 | flat_keyelm_direct_med | flat | key_elements | direct | 3 | **medium** | flat + direct selective |
| V25 | flat_llmidx_caption_med | flat | **llm_index** | caption | 3 | **medium** | index 기반 localize |
| V26 | flat_llmidx_direct_med | flat | **llm_index** | direct | 3 | **medium** | index + direct |
| V27 | hier_keyelm_caption_med | hier | key_elements | caption | 3 | **medium** | hier + medium |
| V28 | hier_llmidx_caption_med | hier | **llm_index** | caption | 3 | **medium** | hier + index + medium |
| V29 | flat_keyelm_caption_high | flat | key_elements | caption | 3 | **high** | low+med → Phase2 |
| V30 | flat_llmidx_caption_high | flat | **llm_index** | caption | 3 | **high** | index + high threshold |
| V31 | hier_keyelm_caption_high | hier | key_elements | caption | 3 | **high** | hier + high |
| V32 | hier_llmidx_caption_high | hier | **llm_index** | caption | 3 | **high** | hier + index + high |
| V33 | flat_llmidx_caption_med_s1 | flat | llm_index | caption | **1** | **medium** | VideoLucy 스타일 |
| V34 | flat_llmidx_caption_high_s1 | flat | llm_index | caption | **1** | **high** | VideoLucy + 더 많은 Phase2 |

**`llm_index` mode**: Leaf segment에 번호를 매겨서 LLM이 인덱스를 선택 → free-form 시간 값보다 파싱 안정적

### 11.8 코드 개선 (v2 → v2.1)

1. **Agentic crash 수정**: `_safe_time_range()` 헬퍼 추가 → time_range 타입 검증 및 정규화
2. **`_agentic_execute()` try/except**: tool 실행 중 crash 시 에러 메시지 반환 (propagation 차단)
3. **`llm_index` 모드 추가**: `_localize_llm_index()` — numbered segment list → LLM이 index 선택

### 11.9 실험 가설 정리

- **threshold=medium**: Phase 1 low-conf (~107-137문제)만 Phase 2로 → 소규모 선택적 visual
- **threshold=high**: low+medium-conf (~122-161문제)도 Phase 2 → 더 많은 질문에 visual 적용
- **flat vs hier coarse**: flat이 Phase 1에서 +3.4%p 유리 (54.8% vs 51.4%)
- **llm_index vs llm_select vs key_elements**: localization 안정성 및 정확도 비교
- **caption vs direct**: caption이 B-bias를 피하지만, re-judge 단계에서 노이즈 추가 가능
- **Oracle gap = 12.2%p**: 선택적 routing만 잘 해도 ~55-60% 도달 가능

### 11.10 V23-V34 최종 결과

| # | 실험명 | Acc | P1 | P2 | P3 | 비고 |
|---|--------|-----|----|----|----|----- |
| **V25** | flat+llmidx+caption+med | **55.8%** (502/900) | 793 | 67 | 40 | **New Best!** |
| V30 | flat+llmidx+caption+high | 55.4% (499/900) | 778 | 81 | 41 | high → 약간 하락 |
| V33 | flat+llmidx+caption+med+s1 | 55.3% (498/900) | 793 | 61 | 46 | 단일 구간, 거의 동일 |
| V23 | flat+keyelm+caption+med | 53.4% (481/900) | 793 | 99 | 8 | key_elements 하락 |
| V29 | flat+keyelm+caption+high | 53.4% (481/900) | 778 | 113 | 9 | key_elements 하락 |

### 11.11 V25 상세 분석 (Best Config)

**V25 vs V8 (text-only baseline) per-question 비교:**
- Both correct: 479
- V25 only (visual 덕분): **23문제** gained
- V8 only (visual 방해): **14문제** lost
- Both wrong: 384
- **Net: +9문제 (+1.0%p)**

**Phase 2 (low-conf 107문제) 정확도:**
- Phase 2 진입: 107/900 (11.9%)
- Phase 2 정답: 51/107 = **47.7%** (vs V8의 같은 질문들 42/107 = 39.3%)
- Phase 2만으로 **+8.4%p** 향상 (이 107문제 한정)

**Phase 2 해결 단계:**
- iter1에서 해결: 63/107 (59%) — 첫 visual observation으로 confident
- iter2-4: 4/107 (4%)
- forced (iter 소진): 40/107 (37%)

### 11.12 핵심 결론

1. **`llm_index` >> `key_elements`**: 동일 조건에서 +2.4%p (V25 55.8% vs V23 53.4%). LLM이 계층 맥락을 보고 번호를 고르는 것이 키워드 매칭보다 정확.
2. **`key_elements` localization은 오히려 해로움**: V23/V29 (53.4%) < V8 text-only (54.8%). 잘못된 구간 선택 → visual noise.
3. **threshold=medium 최적**: low-conf만 Phase 2로 보내는 게 가장 효율적.
4. **Visual search는 선택적으로 쓸 때만 유효**: 전체 투입(V3 47.4%) < text-only(V8 54.8%) < 선택적(V25 55.8%).
5. **Oracle gap 여전히 큼**: V8+V12 oracle = 67.0%. 현재 55.8%이므로 ~11%p 잠재력 잔존.

### 11.13 코드 개선 (v2 → v2.2)

1. **Agentic crash 수정**: `_safe_time_range()` 헬퍼 → time_range 타입 검증/정규화
2. **`_agentic_execute()` try/except**: tool crash 시 에러 메시지 반환
3. **`llm_index` 모드 추가**: numbered segment list + 계층 구조 헤더 → LLM이 index 선택
4. **Interval merge**: 인접/겹치는 구간 자동 병합, 분리 구간은 유지

### 11.14 실험 상태

```
완료:  V1-V12, V23, V25, V28, V29, V30, V33
미제출: V13-V22 (threshold=low → V1 동일 예상), V24, V26-V27, V31-V32, V34
Best:  V25 = 55.8% (flat + llm_index + caption + medium threshold)
```

### 11.15 V28 (hier + llm_index + caption + medium) 결과

V25(flat)와 동일 설정에서 coarse만 hierarchical로 변경.

| 실험 | Coarse | Acc | Phase1 |
|------|--------|-----|--------|
| **V25** | **flat** | **55.8%** (502/900) | 793 |
| V28 | hier | 52.7% (474/900) | — |

→ flat이 **+3.1%p** 우위. Per-question: V25만 정답 91건 vs V28만 정답 63건. net V25 +28.
→ Phase 1에서 flat(56.9%) > hier(53.6%). Hierarchical coarse가 Phase 1 정확도를 깎음.

### 11.16 LVBench V25 결과 (1549Q)

V25 config(flat + llm_index + caption + medium)을 LVBench 전체 1549문제에 적용.

| Phase | 정답 | 비율 |
|-------|------|------|
| Phase 1 (coarse) | 525/1073 | 48.9% |
| Phase 2 iter1 | 108/270 | 40.0% |
| Phase 3 forced | 39/182 | 21.4% |
| **전체** | **686/1549** | **44.3%** |

기존 LVBench best(41.95%) 대비 **+2.3%p** 개선.

**Localization validation** (time_reference 활용):
- Hit (정답 구간 포함): 86/308 = 27.9% → accuracy 48.8%
- Miss (정답 구간 미포함): 222/308 = 72.1% → accuracy 37.4%
- Hit vs Miss gap: **+11.4%p** — localization이 맞으면 정답률 확 오름

---

## 12. Budget-Constrained Leaf Selection Ablation (LVBench 1549Q)

### 12.1 배경

리뷰어 질문: "leaf summary를 flat dump하면 되는데, 왜 multi-level hierarchy를 만드냐?"
→ **Budget constraint 하에서 hierarchy 기반 selection이 uniform보다 효율적이라는 가설 검증**

### 12.2 실험 설계

leaf summary 중 K개만 선택 가능할 때, 3가지 전략 비교:
- **uniform**: 전체 leaf에서 K개 균등 샘플링
- **sequential**: 앞에서부터 K개 순차 선택
- **hierarchy**: Level_3 → Level_2 → Level_1 재귀적 top-down 선택 (LLM이 각 레벨에서 budget 배분)

Settings: `confidence_threshold: low` (Phase 2 미진입, text-only 비교), LVBench 1549Q

### 12.3 B50 결과 (K=50)

| Strategy | Accuracy | Progress |
|----------|----------|----------|
| **b50_uniform** | **580/1549 = 37.4%** | ✅ |
| b50_sequential | 567/1549 = 36.6% | ✅ |
| b50_hierarchy (summary only) | 512/1549 = 33.1% | ✅ |
| b50_hierarchy+KE (key_elements 추가) | ~300/969 ≈ 31.0% | 중단 (경향 확인) |

### 12.4 핵심 발견

1. **Hierarchy 기반 top-down selection이 가장 나쁨** (33.1% < uniform 37.4%)
2. **Key_elements 추가해도 개선 안 됨** (31.0% — 오히려 악화)
3. **Uniform이 가장 안전한 전략**: 전체 영상을 균등하게 커버
4. **Sequential도 uniform보다 살짝 나쁨** (-0.8%p): 앞쪽 편향

### 12.5 실패 원인 분석

- **상위 summary 추상성**: Level_3 "Men bond across urban settings..." → 구체적 질문의 답 위치 판단 불가
- **잘못된 배제의 비가역성**: Level_3에서 한 branch에 budget 0을 주면 그 아래 모든 leaf가 영구 탈락
- **Key_elements 노이즈**: 상위 레벨 KE가 너무 많고 범위가 넓어 오히려 LLM 판단 방해
- **Uniform의 안전성**: 편향 없이 전체 커버 → 정답 구간이 어디에 있든 일정 확률로 포함

### 12.6 Structured Memory 장점이 드러나는 곳 vs 안 드러나는 곳

| 역할 | 효과 | 근거 |
|------|------|------|
| Stage 2가 summary/caption **생성** | **필수** | Flat Baseline = 54.8%, 이것 없이는 아무것도 안 됨 |
| 계층 구조를 **참고 맥락으로 제공** (llm_index) | **+2.4%p** | V25(55.8%) vs V23(53.4%), hier header가 LLM에 맥락 제공 |
| Counting/Temporal **focused search** | **+2~6%p** | Best vs Flat, 특정 카테고리 한정 |
| hierarchy **top-down selection** (budget 실험) | **역효과** | hierarchy 33.1% < uniform 37.4% |
| key_elements **retrieval** | **~0** | R10b(-KE)=53.4% ≈ Best 53.2% |

**결론**: Structured memory의 가치는 **정보 생성**(Stage 2 tree building)과 **맥락 제공**(llm_index의 hier header)에 있음. **Selection/Navigation 도구**로서의 hierarchy는 일관되게 실패.
