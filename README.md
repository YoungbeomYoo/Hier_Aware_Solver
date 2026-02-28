# find_solver_please — Unified Video QA Solver

Memory Tree를 활용한 Video QA Solver. Video-MME, LVBench, HD-EPIC 벤치마크 지원.

> 아키텍처 상세: [solver-tester.md](solver-tester.md)

---

## Quick Start

```bash
# 단일 실행 (로컬 GPU)
python solver.py --config config/video_mme_tree_search_visual.yaml

# 특정 문제만
python solver.py --config config/lvbench.yaml --question_id 2257 --video_id Cm73ma6Ibcs

# SLURM 대규모 실행
bash 0227_slurm_video_mme.sh all gigabyte_a6000 semantic
bash 0227_slurm_lvbench.sh all tyan_a6000 visual

# 결과 확인
python check_results.py output/video_mme_full_semantic/v2
```

---

## 파일 구조

### Core

| 파일 | 설명 |
|------|------|
| `solver.py` | Entry point. Config YAML 로딩 → Adapter + Pipeline + Components 조립 → 실행 |
| `aggregate.py` | SLURM 병렬 실행 후 `by_qid/*.json` → `summary.json` 합산 |
| `config/*.yaml` | 데이터셋/파이프라인/컴포넌트 설정 (42개) |
| `components/` | Plug-and-play 모듈 (17개) |
| `pipelines/` | Solver 파이프라인 (memory_only, tree_search, cognitive, agentic, routed) |
| `adapters/` | 데이터셋별 I/O (video_mme, lvbench, hd_epic) |
| `prompts/` | LLM prompt 템플릿 (14개, config에서 교체 가능) |

### 분석 도구

| 파일 | 설명 | 사용법 |
|------|------|--------|
| `check_results.py` | 실험 결과 조회 + baseline 비교 | [아래 상세](#check_resultspy) |
| `compare_coverage.py` | 두 실험의 탐색 시간 구간 비교 (IoU) | [아래 상세](#compare_coveragepy) |
| `compare_node_quality.py` | 노드 선택 품질 비교 (semantic vs visual) | [아래 상세](#compare_node_qualitypy) |
| `analyze_time_hit.py` | LVBench target time hit rate 분석 | [아래 상세](#analyze_time_hitpy) |

### SLURM 스크립트

| 파일 | 설명 |
|------|------|
| `0227_slurm_video_mme.sh` | Video-MME full-set SLURM 배열 작업 |
| `0227_slurm_lvbench.sh` | LVBench full-set SLURM 배열 작업 |
| `0227_run_all_full.sh` | 4개 실험 일괄 제출 (Video-MME×2 + LVBench×2) |
| `0227_run_both.sh` | visual + semantic 순차 실행 |
| `0227_my_idea_solver.sh` | 인터랙티브 실행 (멀티GPU 지원) |
| `run_slurm_parallel.sh` | 범용 SLURM 병렬 실행 |
| `run_*.sh` | 데이터셋별 단일 실행 스크립트 |

---

## solver.py

```bash
# Config 기반 실행
python solver.py --config config/lvbench.yaml

# CLI override
python solver.py --config config/lvbench.yaml \
    --pipeline memory_only \
    --max_hops 3 \
    --max_frames 16 \
    --output_dir ./output/my_experiment

# 특정 문제/비디오만
python solver.py --config config/video_mme.yaml \
    --question_id "604-1" \
    --video_id "0RxMZBLeqRI"

# Dry run (모델 로딩 없이 flow 확인)
python solver.py --config config/lvbench.yaml --dry_run

# 이미 완료된 문제 스킵 (SLURM 재실행 시)
python solver.py --config config/lvbench.yaml --cached 1
```

### Output 구조

```
output/{experiment_name}/vN/
├── by_qid/           # 문제별 JSON 결과
│   ├── 604-1.json    #   pred, answer, correct, hop_contexts, semantic_match, ...
│   ├── 604-2.json
│   └── ...
├── logs/             # SLURM 로그 (slurm_{jobid}_{taskid}.out/.err)
├── question_list.tsv # SLURM용 (qid\tvideo_id)
└── summary.json      # aggregate 결과 (accuracy, per_task_type, per_domain)
```

---

## SLURM 실행

### Video-MME

```bash
# semantic 모드 (Qwen3-Embedding 기반 매칭)
bash 0227_slurm_video_mme.sh all gigabyte_a6000 semantic

# visual 모드 (exact match tree filter)
bash 0227_slurm_video_mme.sh all gigabyte_a6000 visual

# 둘 다 한번에
bash 0227_slurm_video_mme.sh both

# 진행 상황
bash 0227_slurm_video_mme.sh status gigabyte_a6000 semantic

# 결과 합치기
bash 0227_slurm_video_mme.sh aggregate gigabyte_a6000 semantic
```

### LVBench

```bash
bash 0227_slurm_lvbench.sh all tyan_a6000 semantic
bash 0227_slurm_lvbench.sh all tyan_a6000 visual
bash 0227_slurm_lvbench.sh both
bash 0227_slurm_lvbench.sh aggregate tyan_a6000 semantic
```

### 일괄 제출

```bash
bash 0227_run_all_full.sh              # 4개 실험 전부
bash 0227_run_all_full.sh video_mme    # Video-MME만
bash 0227_run_all_full.sh lvbench      # LVBench만
bash 0227_run_all_full.sh status       # 진행 확인
bash 0227_run_all_full.sh aggregate    # 전체 결과 합치기
```

### 참고

- Partition: `gigabyte_a6000`, `tyan_a6000`, `suma_a600` 등
- QOS: `big_qos`
- Container: `/scratch2/youngbeom/simg/acl2026.simg` (singularity)
- 1000개 초과 시 자동 배치 분할 (SLURM MaxArraySize=1001)
- `--cached 1` 옵션으로 이미 완료된 문제 자동 스킵

---

## 분석 도구

### check_results.py

실험 결과 조회, baseline 비교, 실험 간 비교.

```bash
# 전체 실험 목록 + 요약
python check_results.py

# 특정 실험 상세 (task_type, domain 별 accuracy)
python check_results.py output/video_mme_full_semantic/v2

# baseline 비교 (task_type/domain별 delta 표시)
python check_results.py output/video_mme_full_semantic/v2 \
    --baseline output/videolucy-videomme-long/merged_summary.json

# 두 실험 비교 (A만 맞춤, B만 맞춤, 둘다 맞춤/틀림)
python check_results.py --compare \
    output/video_mme_full_semantic/v2 \
    output/video_mme_full_visual/v1

# 비디오별 breakdown
python check_results.py output/video_mme_full_semantic/v2 --by_video

# 틀린 문제만
python check_results.py output/video_mme_full_semantic/v2 --wrong
```

### compare_coverage.py

두 실험이 각 문제를 풀 때 **어떤 시간 구간을 참조했는지** 비교. 30초 단위 셀로 IoU 계산.

```bash
# 전체 겹침 통계
python compare_coverage.py \
    output/video_mme_subset_tree_search_visual3/v1 \
    output/video_mme_subset_tree_search_semantic/v2

# 특정 문제 상세 (양쪽이 본 시간 구간 나열)
python compare_coverage.py DIR_A DIR_B --qid 604-1

# 겹침 통계만
python compare_coverage.py DIR_A DIR_B --overlap_only

# 결과가 다른 문제만
python compare_coverage.py DIR_A DIR_B --diff_only

# 모든 문제 개별 비교
python compare_coverage.py DIR_A DIR_B --show_all
```

**출력 예시:**
```
  Coverage Comparison: 30 common questions
  A: 14/30 correct (46.7%)
  B: 14/30 correct (46.7%)

  Exploration overlap:
    Avg IoU: 0.312
    A-only cells:  45 (23 min total)
    B-only cells:  38 (19 min total)
    Shared cells: 28 (14 min total)

  IoU distribution:
    Low (<0.3):   12 questions — 완전히 다른 구간 참조
    Mid (0.3-0.7): 10 questions — 부분 겹침
    High (>=0.7):  8 questions — 비슷한 구간 참조
```

### compare_node_quality.py

**Semantic vs Visual** 노드 선택 품질 비교. cue 키워드와 caption의 단어 겹침(relevance), semantic embedding match의 good/bad ratio 분석.

```bash
# 3개 실험 비교
python compare_node_quality.py \
    output/video_mme_subset_tree_search_semantic/v2 \
    output/video_mme_subset_tree_search_visual3/v1 \
    output/video_mme_subset_tree_search_visual2

# 특정 문제 상세
python compare_node_quality.py DIR_A DIR_B DIR_C --qid 604-1

# 결과가 다른 문제만 상세 출력
python compare_node_quality.py DIR_A DIR_B DIR_C --diff_only

# 모든 문제 상세
python compare_node_quality.py DIR_A DIR_B DIR_C --detail
```

**출력 예시:**
```
  Keyword Relevance (cue words ∩ caption words):
    Experiment                                     AvgRel  %HasRel  AvgSegs
    semantic/v2                                    0.056    37.4%    25.5
    visual3/v1                                     0.068    42.3%    22.2

  Semantic Embedding Match Quality:
    Good matches (word overlap): 122
    Bad matches (high sim, no overlap): 274
    Bad ratio: 69.2%

  Per-question winner (highest avg relevance):
    semantic/v2   wins 12/30
    visual3/v1    wins 18/30
```

### analyze_time_hit.py

**LVBench 전용.** 각 문제의 ground-truth `time_reference`와 솔버가 실제 탐색한 시간 구간의 겹침(hit rate) 분석.

```bash
# 기본 분석
python analyze_time_hit.py output/lvbench_full_semantic/v3

# 상세 (모든 문제 개별 출력)
python analyze_time_hit.py output/lvbench_full_semantic/v3 --detail

# hit 못한 문제만
python analyze_time_hit.py output/lvbench_full_semantic/v3 --missed

# 맞춘 문제만 / 틀린 문제만
python analyze_time_hit.py output/lvbench_full_semantic/v3 --correct_only
python analyze_time_hit.py output/lvbench_full_semantic/v3 --wrong_only
```

**출력 예시:**
```
  Target Time Hit Analysis — 45 questions
  Any overlap (hit):    22/45 (48.9%)
  >=50% coverage:       15/45 (33.3%)
  Avg target coverage:  33.8%

  Correctness × Hit:
    Correct + Hit:     13  (정답 맞춤 + target 봄)
    Correct + Miss:    10  (정답 맞춤 + target 안봄)
    Wrong + Hit:        9  (틀림 + target 봤는데 틀림)
    Wrong + Miss:      13  (틀림 + target 안봄)

  Miss한 문제의 target까지 거리:
    중앙값: 86초 (1.4분)
```

---

## Config 구조

```yaml
dataset: video_mme          # video_mme | lvbench | hd_epic
pipeline: tree_search        # memory_only | tree_search | cognitive | agentic | routed

model:
  path: /scratch2/youngbeom/ckpt/Qwen3-VL-8B-Instruct
  type: qwen3vl             # qwen3vl | qwen25vl
  dtype: bfloat16
  attn_impl: flash_attention_2

paths:
  memory_dir: /lustre/youngbeom/DyHiStreamMem/poc/results/Video-MME/stage2_30sec_no_window
  question_path: /lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/long
  video_root: /scratch2/youngbeom/datasets/Video-MME/all
  output_dir: ./output/video_mme_full_semantic

components:
  query_analyzer:
    prompt: detailed          # prompts/ 폴더에서 선택
  judge: {}
  tree_filter:
    match_threshold: 1        # exact match = 1, fuzzy < 1
  context_assembler:
    max_text_budget: 100000
  semantic_matcher:           # semantic 모드에서만 사용
    model_path: /scratch2/youngbeom/ckpt/Qwen3-Embedding-0.6B
    top_k: 30
    batch_size: 64
    score_mode: sum
  frame_loader:
    max_frames: 30

pipeline_params:
  max_hops: 5
  max_text_budget: 100000
```

### 주요 Config 파일

| Config | Dataset | Pipeline | 특징 |
|--------|---------|----------|------|
| `video_mme_full_tree_search_semantic.yaml` | Video-MME long | tree_search | Embedding 기반 L1 노드 선택 |
| `video_mme_full_tree_search_visual.yaml` | Video-MME long | tree_search | Exact match tree filter |
| `lvbench_full_tree_search_semantic.yaml` | LVBench | tree_search | Embedding 기반 |
| `lvbench_full_tree_search_visual.yaml` | LVBench | tree_search | Exact match |
| `cognitive_*.yaml` | 각 데이터셋 | cognitive | 인지과학 기반 4-stage |
| `video_mme.yaml` | Video-MME | memory_only | Text-only baseline |
| `hd_epic.yaml` | HD-EPIC | routed | Track A/B 시간 라우팅 |

---

## 데이터 경로

### Memory Tree

| Dataset | Path |
|---------|------|
| Video-MME | `/lustre/youngbeom/DyHiStreamMem/poc/results/Video-MME/stage2_30sec_no_window` |
| LVBench | `/lustre/youngbeom/DyHiStreamMem/poc/results/LVBench/stage2_30sec_no_window` |
| HD-EPIC | `/lustre/youngbeom/DyHiStreamMem/poc/results/HD-EPIC/stage2_v8_sync_30sec_no_window-hd-epic-tuned` |

### Questions

| Dataset | Path |
|---------|------|
| Video-MME | `/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/long` |
| LVBench | `/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl` |
| HD-EPIC | `/lustre/youngbeom/DyHiStreamMem/datasets/HD-EPIC/hd-epic-annotations/vqa-benchmark` |

### Videos

| Dataset | Path |
|---------|------|
| Video-MME | `/scratch2/youngbeom/datasets/Video-MME/all` |
| LVBench | `/scratch2/youngbeom/datasets/LVBench/all_videos` |

### Models

| Model | Path |
|-------|------|
| Qwen3-VL-8B-Instruct | `/scratch2/youngbeom/ckpt/Qwen3-VL-8B-Instruct` |
| Qwen3-Embedding-0.6B | `/scratch2/youngbeom/ckpt/Qwen3-Embedding-0.6B` |

### Baseline

| Name | Path | Accuracy |
|------|------|----------|
| VideoLucy (Video-MME long) | `output/videolucy-videomme-long/merged_summary.json` | 51.91% (888/900) |

---

## 두 가지 Solver 모드

### Visual (Exact Match)
- Memory tree를 계층적으로 탐색하며 **cue 키워드가 caption/key_elements에 정확히 포함**되는 노드 선택
- `tree_filter.match_threshold: 1`
- False positive 적음, recall이 낮을 수 있음

### Semantic (Embedding Match)
- Qwen3-Embedding-0.6B로 질문 요소와 L1 노드의 **코사인 유사도** 계산
- `semantic_matcher` 컴포넌트 사용
- Recall 높지만 **false positive 문제** 있음 (예: "guitar" → "giant head sculpture" sim=0.996)
- `score_mode: sum` — 매칭 점수 합산으로 L1 노드 순위 결정
