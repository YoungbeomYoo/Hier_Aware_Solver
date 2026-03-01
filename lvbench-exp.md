# LVBench Experiments

## Dataset Overview

- **Total**: 103 videos, 1,549 questions
- **Available subset** (memory ready): 81 videos, 1,193 questions
- **Mini subset**: 60 questions (Q type별 10개, seed=42)
- **Key feature**: 모든 문제에 `time_reference` (정답 시간 구간) annotation 있음
- **Memory**: `/lustre/youngbeom/DyHiStreamMem/poc/results/LVBench/stage2_v9`
- **Questions**: `/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl`

### Question Type Distribution (available subset)

| Type | Count |
|------|-------|
| entity recognition | 532 |
| event understanding | 486 |
| key information retrieval | 213 |
| temporal grounding | 179 |
| reasoning | 158 |
| summarization | 40 |

## Why LVBench?

Video-MME에서 cue/filter 변경이 정답률에 영향이 없었던 이유:
Phase 0 + navigation이 결과를 지배 → 초기 검색 품질이 측정 불가.

LVBench는 `time_reference`가 있으므로 **retrieval quality를 직접 측정** 가능:
- "우리 tree filter가 정답 구간(time_reference)을 포함하는 leaf를 활성화했는가?"
- Recall@K, time overlap IoU 등으로 search quality 독립 평가 가능

---

## Experiment 1: Baseline (Best Config, Mini 60문제)

**Config**: `config/full_run/lvbench_best.yaml`
- A1(LLM) + B1(exact) + C3(VL+AJ) + D0(text) + F1(compact) + G1(coarse_first) + A2v3(recovery_cue coarse)

### Overall

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **21/60 = 35.0%** |
| Phase 0 accuracy | 19/50 = 38.0% |
| Hop loop accuracy | 2/8 = 25.0% |
| Phase 0 비율 | 50/60 = 83.3% |
| Active leaves (mean) | 85.2 / 130.2 |
| Recovery used | 2/60 |

### By Question Type

| Type | Correct | Total | Accuracy |
|------|---------|-------|----------|
| key information retrieval | 8 | 16 | 50.0% |
| event understanding | 8 | 18 | 44.4% |
| temporal grounding | 7 | 19 | 36.8% |
| reasoning | 4 | 14 | 28.6% |
| entity recognition | 4 | 17 | 23.5% |
| summarization | 2 | 11 | 18.2% |

### 문제점

1. **Phase 0가 high confidence로 50개 답했는데 38%만 정답** — coarse overview가 "확신"하고 틀리는 비율이 높음
2. **Entity recognition 23.5%** — 캐릭터/사물 인식이 텍스트 요약만으로 부족
3. **Summarization 18.2%** — 전체 요약 질문도 coarse만으로 부족
4. Hop loop은 8문제만 진입, 2개 정답 (25%)

### 시사점

- LVBench가 Video-MME보다 확실히 어려움 (35% vs 53%)
- Phase 0의 false positive (high confidence + wrong)가 핵심 문제
- `time_reference`로 retrieval quality 측정하면 search 개선의 가치를 직접 보여줄 수 있음

---

## Experiment 2: Segment Selection (LLM-based, Mini 60문제)

**목적**: Hierarchical summary(Level_N~Level_1)만 보고 LLM이 정답 구간을 찾을 수 있는가?
**방법**: Qwen3-VL-8B에게 Level_2+ overview + Level_1 개별 segment(30초 단위)를 보여주고 "가장 관련있는 5개 구간 선택"

### Setup
- **Model**: Qwen3-VL-8B-Instruct (text-only, A5000)
- **Context**: Level_N~Level_2 overview + Level_1 개별 segment (summary + key_elements)
- **Task**: 5개 구간 선택 → time_reference(GT)와 overlap 여부 측정
- **Script**: `eval_segment_selection.py`

### Overall (50/60 questions, 10 failed: 7 parsing failure + 1 OOM + 2 skipped)

| Metric | Value |
|--------|-------|
| **Hit@1** | **11/50 = 22.0%** |
| **Hit@3** | **17/50 = 34.0%** |
| **Hit@5** | **19/50 = 38.0%** |

### By Question Type

| Type | Hit@1 | Hit@3 | Hit@5 | Total |
|------|-------|-------|-------|-------|
| reasoning | 3 (30.0%) | 6 (60.0%) | **6 (60.0%)** | 10 |
| event understanding | 5 (31.2%) | 5 (31.2%) | 5 (31.2%) | 16 |
| entity recognition | 2 (13.3%) | 4 (26.7%) | 5 (33.3%) | 15 |
| key info retrieval | 3 (25.0%) | 4 (33.3%) | 4 (33.3%) | 12 |
| summarization | 2 (22.2%) | 2 (22.2%) | 3 (33.3%) | 9 |
| temporal grounding | 4 (23.5%) | 4 (23.5%) | 5 (29.4%) | 17 |

### 분석

1. **Hit@5 = 38%** — 5개 구간 중 하나라도 GT와 겹치는 비율. 비디오 평균 ~130개 segment 중 5개 선택이므로 random baseline ~3.8%
2. **Reasoning이 60%로 최고** — 추론 질문은 narrative/plot 관련이라 summary로 잡기 쉬움
3. **Temporal grounding이 29.4%로 최저** — "몇 분에 X가 나오는가" 류는 summary에 시간 정보가 부족
4. **Event understanding Hit@1 = 31.2%** — top-1에서 이미 잘 잡는 경우가 많음
5. **파싱 실패 7건** — LLM이 JSON 형식을 못 지킨 경우. 프롬프트 개선 여지 있음

### 시사점

- Hierarchical summary만으로도 random 대비 ~10x Hit@5 달성
- 하지만 **60%+ 질문에서는 5개 구간으로도 GT 못 잡음** → summary 정보의 한계
- Temporal grounding 개선에는 시간 정보 명시적 연결 필요
- 다음 실험: context 형태 변경 (flat vs hierarchical), visual cue 추가 시 개선 여부

---

## Experiment 3: Flat Baseline (Mini 60문제)

**목적**: 모든 leaf summary를 flat dump → 1회 답변. Hierarchy 없이 summary만으로 어디까지 가는가?
**Config**: `config/full_run/lvbench_flat_baseline.yaml`

### Overall

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **18/60 = 30.0%** |

### By Question Type

| Type | Correct | Total | Accuracy |
|------|---------|-------|----------|
| key information retrieval | 7 | 16 | 43.8% |
| event understanding | 6 | 18 | 33.3% |
| reasoning | 4 | 14 | 28.6% |
| temporal grounding | 5 | 19 | 26.3% |
| summarization | 2 | 11 | 18.2% |
| entity recognition | 3 | 17 | 17.6% |

### 분석

- **Flat 30.0% < Best 35.0%** — Video-MME에서는 Flat > Best였지만 LVBench에서는 반대
- LVBench 비디오가 더 길어서 (평균 130+ leaves) flat dump의 context가 너무 길어짐
- Hierarchy의 계층별 요약이 정보 밀도를 높여줌

---

## Experiment 4: Segment → Answer (End-to-End, Mini 60문제)

**목적**: Hierarchy로 구간 선택 → 선택 구간의 raw caption으로 답변. "구간 잡고 상세 정보 투입"의 가치 검증.

### Flow
1. Exp2의 segment selection 결과 (5개 구간) 재활용
2. 선택된 구간의 **leaf raw caption** 또는 **leaf summary** 로드
3. 해당 context로 질문에 답변

### Setup
- **Model**: Qwen3-VL-8B-Instruct (text-only, A5000)
- **Caption context**: 5구간 × ~3K chars = ~15K chars
- **Summary context**: 5구간 × ~150 chars = ~0.8K chars
- **Script**: `eval_segment_answer.py`

### Overall (43/60 questions, 17 skipped: Exp2 파싱 실패 등)

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| **Seg→Caption** | **21** | **43** | **48.8%** |
| Seg→Summary | 14 | 43 | 32.6% |

### By Question Type

| Type | Caption | Summary | Δ |
|------|---------|---------|-------|
| key info retrieval | **7/9 (77.8%)** | 6/9 (66.7%) | +11.1 |
| event understanding | **8/14 (57.1%)** | 6/14 (42.9%) | +14.2 |
| reasoning | **5/10 (50.0%)** | 4/10 (40.0%) | +10.0 |
| temporal grounding | **7/15 (46.7%)** | 6/15 (40.0%) | +6.7 |
| entity recognition | **5/13 (38.5%)** | 2/13 (15.4%) | **+23.1** |
| summarization | 1/7 (14.3%) | 0/7 (0.0%) | +14.3 |

### 전체 비교표

| Method | N | Accuracy | 특징 |
|--------|---|----------|------|
| **Seg→Caption** | **43** | **48.8%** | **hierarchy 구간 선택 + raw caption** |
| Best (tree search) | 60 | 35.0% | A1+B1+C3+D0+F1+G1 |
| Seg→Summary | 43 | 32.6% | hierarchy 구간 선택 + summary만 |
| Flat (all summary) | 60 | 30.0% | 전체 leaf summary dump |

### 핵심 발견

1. **Caption이 Summary보다 +16.2%p** — raw caption의 상세 정보가 답변 정확도에 직결
2. **Entity recognition에서 +23.1%p** — 인물/사물 상세 묘사가 caption에만 있음
3. **Key info retrieval 77.8%** — 구간만 잘 잡으면 상세 정보로 높은 정답률 달성
4. **48.8%는 43문제 기준** — 17문제 skip(Exp2 파싱 실패). 전체 60문제 기준으로는 낮아질 수 있음
5. **Raw caption을 전부 넣으면 OOM** — 비디오당 200K~760K chars. Hierarchy로 구간 축소가 필수

### 시사점

- **Hierarchy의 역할이 명확해짐**: 전체 raw caption은 context에 못 넣음 → hierarchy summary로 구간 선택 → 선택 구간 caption 투입
- 다음 단계: segment selection 품질 개선 (Hit@5 38% → ?) + visual frame 추가

---
