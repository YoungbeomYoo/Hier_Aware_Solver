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

## 7. A2 Recovery Cue — Hop Loop 개선 실험

### 7.1 문제 진단

Phase 0(G1)이 전체 성능의 핵심 (806/900 = 89.6% early return). 나머지 94개가 hop loop에 진입하지만, hop loop 정확도는 39.4% (37/94)에 불과.

**Hop loop 실패 원인 분석 (Section 6.9 참고)**:
- keyword cues로 exact match → 의미적으로 중요한 구간을 못 찾음
- 고정 10개 leaf = 9~16% 커버리지 → 너무 적음
- cues는 최초 1회 추출 후 hop loop 내에서 갱신/전달 안 됨

### 7.2 인지과학적 근거

| 이론 | 핵심 | 현재 구현 | 갭 |
|------|------|-----------|-----|
| **SAM (Search of Associative Memory)** | 상위 context cue → sampling → recovery(새 cue 생성) → 재탐색 | cue 1회 추출 → match → 끝 | Recovery 단계 없음 |
| **Hippocampal Indexing Theory** | cue → 해마 인덱스 활성화 → 하위 감각 데이터 재구성 | key_elements = 인덱스, caption = 감각 데이터 | 인덱스 매칭이 exact substring → 의미적 매칭 아님 |
| **Spreading Activation** | 상위 개념 활성화 → 연결 강도 따라 하위 전파 | hop에서 direction만으로 이동 | 활성화 강도 개념 없음 |

**핵심 갭**: cue와 retrieval 사이에 **추론(상상) 단계**가 없음. 사람은 키워드 하나로 연상해가며 시간 구간을 특정하지만, 현재 시스템은 keyword → exact match → 끝.

### 7.3 A2 설계: Recovery Cue

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

### 7.4 구현 내역

| 파일 | 변경 | 설명 |
|------|------|------|
| `pipelines/tree_search.py` | STAGE 2.7 추가 (Phase 0 이후) | `_recovery_cue()` 메서드 + `RECOVERY_CUE_PROMPT` |
| `pipelines/tree_search.py` | STAGE 3 수정 | recovery_targets가 있으면 그것을 initial targets로 사용 |
| `config/full_run/a2_recovery_cue.yaml` | 신규 | `recovery_cue: true` 추가 |
| `output/a2_subset_qid_list.tsv` | 신규 | 94개 hop loop 질문 목록 |
| `run_a2_subset.sh` | 신규 | subset 실행 스크립트 |

### 7.5 파이프라인 흐름 변화

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
                   → hop loop with LLM-guided leaves (94개, ?%)
```

### 7.6 실험 설계

- **데이터**: 94개 hop loop 진입 질문 (75 videos)
- **비교 대상**: Best config의 hop loop 결과 (37/94 = 39.4%)
- **측정**: 전체 정확도 + 카테고리별 (Action Reasoning 30개, Object Reasoning 25개 등)
- **비용**: LLM 콜 1회 추가 (Phase 0 이후 recovery cue 생성)

### 7.7 향후 확장 계획

A2 1회가 효과 있으면:
- **A2 매 hop**: 매 hop마다 "지금까지 본 것 + 남은 의문 → 새 cue → 재탐색" (full SAM cycle)
- **Focus points를 judge에 전달**: recovery cue의 focus_points를 hop loop judge에게 넘겨서 "이것에 집중해서 판단하라" 지시

### 7.8 결과

| Method | N | Accuracy | vs Best Hop | Status |
|--------|---|----------|-------------|--------|
| Best (hop loop only) | 94 | 39.4% (37/94) | — | Baseline |
| **A2 Recovery Cue** | 94 | **TBD** | TBD | 대기 |
