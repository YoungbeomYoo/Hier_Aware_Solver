# Unified Benchmark Solver-Tester

> **Core Concept**: Memory를 활용한 Video QA — 모든 벤치마크에 동일한 Streaming Memory Tree를 활용하여 문제를 풀고, 데이터셋별로 결과를 출력하는 통합 프레임워크

---

## 1. 대상 벤치마크 요약

| Benchmark | Task | Answer Format | Question Source | Memory 구조 |
|-----------|------|---------------|-----------------|-------------|
| **HD-EPIC** | Egocentric Video QA (action recognition, recipe localization 등) | Index (0-based) → correct_idx | JSON dict (q_id → question) | streaming_memory_tree (Level 1/2/3) |
| **Video-MME** | Multi-modal Video Understanding | Letter (A/B/C/D) | Per-question JSON files | streaming_memory_tree (Level 1/2/3/4) |
| **LVBench** | Long Video Understanding | Letter (A/B/C/D) | JSONL (video별 qa list) | streaming_memory_tree (Level 1/2) |

---

## 2. 공통 Memory 구조 (Streaming Memory Tree)

세 데이터셋 모두 **동일한 메모리 형식**을 사용:

```
{video_id}.json
├── video_id: str
├── total_clips: int
└── streaming_memory_tree
    ├── Level_1: [{level, time_segments, summary, key_elements, children: [leaf]}]
    ├── Level_2: [{level, time_segments, summary, key_elements, children: [Level_1 node]}]
    ├── Level_3: (optional) [{...}]
    └── Level_4: (optional) [{...}]
```

**Leaf node (Level_1의 children)**:
```json
{
  "start_time": 0.0,
  "end_time": 30.0,
  "caption": "Detailed description...",
  "summary": "Condensed description...",
  "key_elements": {
    "persons": [], "actions": [], "objects": [],
    "attributes": [], "locations": [], "text_ocr": []
  }
}
```

---

## 3. 데이터셋별 Question Format

### 3.1 HD-EPIC
- **파일**: `{question_type}.json` — `{q_id: {inputs, question, choices, correct_idx}}`
- **특징**: `inputs`에 video ID + start/end time, 선택지에 `<TIME>` 태그

### 3.2 Video-MME
- **파일**: `{video_num}-{q_num}.json` — `{videoID, question_id, question, options, answer}`
- **특징**: `answer`가 `"B"` 형태 letter

### 3.3 LVBench
- **파일**: `video_info.meta.jsonl` — `{key, type, qa: [{uid, question, answer, question_type, time_reference}]}`
- **특징**: question 안에 `(A)...(B)...` 형태 선택지 포함

---

## 4. Plug-and-Play 아키텍처

### 4.1 디렉토리 구조

```
find_solver_please/
├── solver.py                     # Entry point (CLI + config 로딩)
├── solver-tester.md              # 이 문서
├── config/                       # YAML 설정 파일
│   ├── hd_epic.yaml              # HD-EPIC routed
│   ├── video_mme.yaml            # Video-MME memory_only
│   ├── lvbench.yaml              # LVBench agentic
│   ├── cognitive_hd_epic.yaml    # HD-EPIC cognitive
│   ├── cognitive_video_mme.yaml  # Video-MME cognitive
│   └── cognitive_lvbench.yaml    # LVBench cognitive
├── components/                   # Plug-and-play 모듈 (17개)
│   ├── json_extractor.py         # LLM 출력 JSON 파싱
│   ├── time_utils.py             # 시간 변환 유틸리티
│   ├── answer_parser.py          # Answer 추출 (letter/index)
│   ├── query_decomposer.py       # 질문 분해 + cue 추출 [LLM]
│   ├── memory_ops.py             # Memory tree 조작 (flatten, navigate, format)
│   ├── filters.py                # Rule-based filter + LLM leaf selection
│   ├── time_router.py            # Track A/B 시간 기반 라우팅
│   ├── solvability.py            # Solvability check + forced fallback [LLM]
│   ├── frame_loader.py           # Targeted/Uniform frame loading (decord)
│   ├── vlm.py                    # VLM inference (6 backends)
│   ├── coverage.py               # Time reference coverage 분석
│   ├── query_analyzer.py         # [Cognitive] Stage 1: 질문 유형 분류 + 타겟 필드 매핑
│   ├── metadata_filter.py        # [Cognitive] Stage 2: 구조화된 key_elements 필드 검색
│   ├── spreading_activation.py   # [Cognitive] Bottom-Up 확산 활성화
│   ├── uncertainty_checker.py    # [Cognitive] Stage 3: 불확실성 기반 역추적 결정
│   └── elimination_reasoner.py   # [Cognitive] Stage 4: 소거법 기반 MCQ 해결
├── adapters/                     # 데이터셋별 I/O 어댑터
│   ├── base.py                   # BaseAdapter ABC
│   ├── hd_epic.py
│   ├── video_mme.py
│   └── lvbench.py
├── pipelines/                    # Solver 파이프라인 (4종)
│   ├── base.py                   # BasePipeline ABC
│   ├── memory_only.py            # Mode 1: Text-only
│   ├── routed.py                 # Mode 2: Track A/B routing
│   ├── agentic.py                # Mode 3: Multi-hop agentic
│   └── cognitive.py              # Mode 4: 인지과학 기반 4-stage
├── prompts/                      # LLM prompt 템플릿 (14개, 갈아끼우기 용)
│   ├── decompose_default.py      # Cue extraction (HD-EPIC style)
│   ├── decompose_detailed.py     # Cue extraction (LVBench style)
│   ├── solvability_strict.py     # Evidence-based strict
│   ├── solvability_relaxed.py    # Lenient inference
│   ├── vlm_answer_only.py        # Simple A/B/C/D output
│   ├── vlm_with_confidence.py    # Answer + confidence + observation
│   ├── navigate_single_hop.py    # Hierarchical tree navigation
│   ├── leaf_select_budget.py     # Budgeted leaf selection
│   ├── query_classify_default.py # Query type classification
│   ├── query_classify_detailed.py# Detailed classification + temporal/conditional
│   ├── elimination_default.py    # Evidence-based elimination
│   ├── elimination_strict.py     # Strict contradiction-only elimination
│   ├── uncertainty_default.py    # Confidence assessment
│   └── uncertainty_conservative.py # Conservative (visual-favoring)
├── references/                   # 기존 레퍼런스 코드
└── subsets/                      # 샘플 데이터
```

### 4.2 컴포넌트 의존성 흐름

```
                    ┌─────────────────────┐
                    │   solver.py (CLI)    │
                    │  config/*.yaml 로딩   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐   ┌──────────┐   ┌─────────────┐
         │ Adapter  │   │ Pipeline │   │ Components  │
         │(dataset) │   │ (mode)   │   │ (plug&play) │
         └─────────┘   └──────────┘   └─────────────┘
```

### 4.3 VLM Backend 지원 현황

| Backend | Class | Type | 비고 |
|---------|-------|------|------|
| **Qwen3-VL** | `TextOnlyLLM` / `VisionVLM` | Local GPU | 기본 backend |
| **Qwen2.5-VL** | `TextOnlyLLM` / `VisionVLM` | Local GPU | solver.py에서 type으로 선택 |
| **SimpleVLM** | `SimpleVLM` | Local GPU | Text-only baseline |
| **Gemini** | `GeminiLLM` | API | GEMINI_API_KEY 환경변수 |
| **OpenAI-compatible** | `APIBasedLLM` | API | DeepSeek, vLLM, Ollama 등 |

---

## 5. 컴포넌트별 Plug-and-Play 인터페이스

### 5.1 LLM 기반 컴포넌트 (prompt 교체 가능)

| Component | Class | Input | Output | Prompt 종류 |
|-----------|-------|-------|--------|-------------|
| Query Decomposition | `QueryDecomposer` | question, choices | `{cues, target_action}` | `decompose_default`, `decompose_detailed` |
| Hierarchical Navigation | `HierarchicalNavigator` | memory_tree, cues | `(memory, segment, path)` | `navigate_single_hop` |
| Leaf Selection | `LLMLeafSelector` | candidates, cues, question | `list[int]` | `leaf_select_budget` |
| Solvability Check | `SolvabilityChecker` | context, question, options | `{solvable, answer, needs_depth}` | `solvability_strict`, `solvability_relaxed` |
| Forced Fallback | `ForcedAnswerFallback` | context, question, options | `str (A/B/C/D)` | built-in |
| Vision VLM | `VisionVLM` | frames, memory, question | `{answer, confidence, observation}` | `vlm_answer_only`, `vlm_with_confidence` |

### 5.2 Non-LLM 컴포넌트 (전략 교체 가능)

| Component | Class | 기능 |
|-----------|-------|------|
| Leaf Flattening | `LeafFlattener` | Memory tree → flat leaf list |
| Rule-based Filter | `RuleBasedFilter` | Keyword matching, LLM 비용 0 |
| Time Router | `TimeRouter` | Track A/B 시간 추출 + 구간 해석 |
| Frame Loader | `TargetedFrameLoader` / `UniformFrameLoader` | decord 기반 프레임 추출 |
| Memory Formatter | `MemoryContextFormatter` | 4종 포맷 (flat, bottom_up, leaf_batch, leaf_compact) |
| Coverage Analyzer | `CoverageAnalyzer` | Time reference coverage 계산 |

---

## 6. Prompt 교체 방법

### Config에서 prompt 지정
```yaml
components:
  query_decomposer:
    prompt: detailed              # prompts/decompose_detailed.py 사용
  solvability_checker:
    prompt: relaxed               # prompts/solvability_relaxed.py 사용
```

### 새 prompt 추가
1. `prompts/` 폴더에 `{component}_{variant}.py` 파일 생성
2. `PROMPT = """..."""` 변수 정의 (format placeholder 포함)
3. `prompts/__init__.py`의 `_REGISTRY`에 등록

```python
# prompts/decompose_custom.py
PROMPT = """Your custom prompt here with {question} and {choices_str} placeholders..."""
```

---

## 7. 실행 방법

### Config 기반 실행
```bash
# LVBench - agentic multi-hop
python solver.py --config config/lvbench.yaml

# Video-MME - memory-only baseline
python solver.py --config config/video_mme.yaml

# HD-EPIC - Track A/B routed
python solver.py --config config/hd_epic.yaml
```

### CLI Override
```bash
# Pipeline 변경
python solver.py --config config/lvbench.yaml --pipeline memory_only

# Max hops / frames 변경
python solver.py --config config/lvbench.yaml --max_hops 3 --max_frames 16

# Output 디렉토리 변경
python solver.py --config config/lvbench.yaml --output_dir ./output/exp_v2

# Dry run (모델 로딩 없이 flow 확인)
python solver.py --config config/lvbench.yaml --dry_run
```

### Output 구조
```
output/{dataset}/
├── by_qid/                    # Per-question JSON
│   ├── question_id_1.json
│   ├── question_id_2.json
│   └── ...
└── summary.json               # Accuracy + breakdown
```

---

## 8. 데이터 경로

### Memory
| Dataset | Path |
|---------|------|
| HD-EPIC | `/lustre/youngbeom/DyHiStreamMem/poc/results/HD-EPIC/stage2_v8_sync_30sec_no_window-hd-epic-tuned` |
| Video-MME | `/lustre/youngbeom/DyHiStreamMem/poc/results/Video-MME/stage2_30sec_no_window` |
| LVBench | `/lustre/youngbeom/DyHiStreamMem/poc/results/LVBench/stage2_30sec_no_window` |

### Questions
| Dataset | Path |
|---------|------|
| HD-EPIC | `/lustre/youngbeom/DyHiStreamMem/datasets/HD-EPIC/hd-epic-annotations/vqa-benchmark/*.json` |
| Video-MME | `/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/long/*.json` |
| LVBench | `/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl` |

### Model
| Item | Path |
|------|------|
| Qwen3-VL-8B | `/scratch2/youngbeom/ckpt/Qwen3-VL-8B-Instruct` |

---

## 9. 핵심 설계 원칙

1. **Plug-and-Play**: 모든 LLM 기반 컴포넌트는 prompt template + llm_fn 주입으로 교체 가능
2. **Config-Driven**: YAML 파일 하나로 dataset/pipeline/prompt/params 전부 제어
3. **Zero Coupling**: 컴포넌트 간 직접 import 없음, pipeline이 조합
4. **Memory-First**: 항상 memory로 먼저 시도 → 실패 시 frame fallback
5. **Cost-Aware**: rule-based filter로 LLM 호출 최소화
6. **Incremental Output**: question 단위 JSON 저장 (중단 후 재시작 가능, `--cached`)
7. **Multi-Backend**: Qwen3-VL, Qwen2.5-VL, Gemini API, OpenAI-compatible API 지원

---

## 10. Cognitive Pipeline — 인지과학 기반 4단계 파이프라인

### 10.1 이론적 배경

| 인지과학 모델 | 적용 단계 | 설명 |
|-------------|---------|------|
| **SAM** (Search of Associative Memory) | Stage 1 | 질문에서 '탐색 범위'를 좁히는 컨텍스트 큐 추출 |
| **Hippocampal Indexing Theory** | Stage 2 | Memory Tree를 인덱스로 활용 — raw 데이터가 아닌 포인터 검색 |
| **Schema-based Retrieval** (Bartlett) | Stage 2 | 계층적 스키마(Goal → Step → Substep)로 맥락 재구성 |
| **Spreading Activation** | Stage 2 | Leaf에서 시작하여 부모 노드를 타고 올라가 Global Context 획득 |
| **Prediction Error → Backtracking** | Stage 3 | 텍스트 메모리만으로 부족할 때만 raw video 로딩 |
| **Elimination-by-Aspects** (Tversky) | Stage 4 | 옵션을 한꺼번에 비교하지 않고, 하나씩 소거하여 최종 답변 |

### 10.2 파이프라인 Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Query Decoupling & Scope Parsing (SAM)            │
│  QueryAnalyzer.analyze()                                     │
│  → question_type, target_fields, cues, time_ranges          │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │ has_explicit_time?     │
          │                       │
     ┌────▼────┐            ┌────▼────┐
     │ Track A │            │ Track B │
     │ (Time)  │            │(Semantic)│
     └────┬────┘            └────┬────┘
          │                      │
┌─────────▼──────────────────────▼────────────────────────────┐
│  Stage 2: Metadata Cross-Filtering + Context Assembly        │
│  MetadataTargetedFilter.filter_by_time/fields()              │
│  SpreadingActivation.activate()                              │
│  → activated_context + leaf_context                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  Stage 3: Uncertainty-Driven Visual Backtracking             │
│  UncertaintyChecker.assess()                                 │
│  → certain/likely → Stage 4 (skip visual)                    │
│  → uncertain/insufficient → FrameLoader + VisionVLM         │
│     (VideoLucy-style: 지정된 시간대의 19초 클립만 로딩)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  Stage 4: Aggregation & Multiple Choice Selection            │
│  EliminationReasoner.eliminate()                             │
│  → per-option evidence analysis → elimination → final answer │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 Cognitive 컴포넌트 상세

| Component | Class | Stage | Input | Output |
|-----------|-------|-------|-------|--------|
| Query Analyzer | `QueryAnalyzer` | 1 | question, choices, time_ref | `{question_type, target_fields, cues, time_ranges}` |
| Metadata Filter | `MetadataTargetedFilter` | 2 | leaves, cues, fields | `(marked, unmarked)` — field-targeted |
| Spreading Activation | `SpreadingActivation` | 2 | tree, target_leaves | `{activated_context, hierarchy_chains}` |
| Uncertainty Checker | `UncertaintyChecker` | 3 | context, question, options | `{confidence, needs_visual, visual_time_ranges}` |
| Elimination Reasoner | `EliminationReasoner` | 4 | context, question, options | `{answer, eliminated, remaining, confidence}` |

### 10.4 기존 파이프라인과 차이점

| 특성 | Agentic Pipeline | Cognitive Pipeline |
|------|-----------------|-------------------|
| 검색 방식 | 키워드 매칭 → LLM leaf selection | 질문 유형별 타겟 필드 매핑 |
| 컨텍스트 구성 | Flat leaf batch | Bottom-Up 계층 활성화 (Goal → Step → Substep) |
| Visual 결정 | Solvability 이진 판단 | 4단계 불확실성 (certain/likely/uncertain/insufficient) |
| 답변 방식 | Solvability → 맞으면 끝 | 소거법: 각 선지별 근거 검토 후 탈락 |
| 호출 패턴 | Multi-hop loop | 단일 pass (Stage 1→2→3→4) |

### 10.5 실행 방법

```bash
# Cognitive Pipeline으로 각 데이터셋 실행
python solver.py --config config/cognitive_hd_epic.yaml
python solver.py --config config/cognitive_lvbench.yaml
python solver.py --config config/cognitive_video_mme.yaml

# 기존 pipeline을 cognitive로 override
python solver.py --config config/lvbench.yaml --pipeline cognitive
```

### 10.6 Config 예시 (Cognitive)

```yaml
dataset: hd_epic
pipeline: cognitive

components:
  query_analyzer:
    prompt: detailed              # 질문 유형 정밀 분류
  uncertainty_checker:
    prompt: conservative          # 보수적 (visual 선호)
  elimination_reasoner:
    prompt: strict                # 근거 없는 소거 방지

pipeline_params:
  max_frames: 64
  max_visual_retries: 1
  include_siblings: false         # 같은 부모의 sibling leaf 포함 여부
  leaf_budget: 15                 # 최대 검토 leaf 수
```
