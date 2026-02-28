# Solver Idea — Semantic Matching + Top-K Selection (2026.02.27)

## 배경

기존 tree_search 파이프라인의 필터링은 **exact substring match** 방식:
- query에서 추출한 cues가 memory node의 key_elements에 문자열로 포함되는지 확인
- 문제: "cut onion" cue가 "chopping onions"과 매칭 안 됨 → 관련 정보 누락
- 반대로 semantic matching으로 바꾸면 "다 켜지는" 문제 우려

## 핵심 아이디어

**Qwen3-Embedding-0.6B로 embedding 뽑아서 총합 score → Top-K 선택**

1. Question에서 key_elements 추출 (question 자체 + options + cues)
2. 각 Level_1 노드의 key_elements를 모두 embedding
3. q_element 각각 vs node_element 각각의 cosine similarity 계산
4. 각 q_element의 max similarity → 노드별 **총합** 점수
5. **Top-30 노드 선택** (threshold 없이 상대적 순위로 필터링)

### 왜 총합이 좋은가

- q_elements가 여러 개 (question + 4 options + cues = ~10개 이상)
- 각 q_element별로 노드 내 best match의 similarity를 합산
- 관련 있는 노드: 여러 q_element에 높은 sim → 총합 높음
- 관련 없는 노드: 몇 개만 걸림 → 총합 낮음
- Top-K로 자르면 확실히 필터링됨 (상대 순위이므로)

### 왜 "다 켜지는" 문제가 없나

- threshold 기반이면: 0.3 이상? → 거의 다 통과
- **Top-K 기반이면**: 무조건 30개만 → 나머지는 탈락
- 총합 점수 차이가 자연스럽게 발생 (관련 노드 vs 무관 노드)

## 메모리 구조 참고

메모리 빌더(`stage2_evolving_cluster_v8_sync.py`)에서 key_elements는 **합집합(union)**으로 상위로 전파:
```
Level_3.key_elements = union(Level_2 children의 key_elements)
Level_2.key_elements = union(Level_1 children의 key_elements)
Level_1.key_elements = union(leaf children의 key_elements)
```

각 카테고리: `actions`, `objects`, `persons`, `attributes`, `locations`, `text_ocr`

예시 (LVBench 비디오 하나):
- Level_1: actions 30개, objects 36개, persons 7개, ...
- Level_2: actions 194개, objects 292개 (합집합이라 많음)
- Level_3: actions 19개, objects 36개 (높은 추상도)

→ **Level_1에서 매칭하는 게 granularity 적절** (너무 많지도 적지도 않음)

## 구현

### 새 파일: `components/semantic_matcher.py`

```python
class SemanticMatcher:
    def __init__(self, model_path, top_k=30, batch_size=64, score_mode="sum"):
        # Qwen3-Embedding-0.6B lazy loading

    def encode(self, texts: list[str]) -> np.ndarray:
        # L2-normalized embeddings

    def score_nodes(self, q_elements, nodes, categories=None) -> list[dict]:
        # q_element별 max similarity → 노드별 총합 score
        # 결과: [{"node_idx", "score", "top_matches", "n_elements"}, ...]

    def select_top_nodes(self, q_elements, tree, level="Level_1", top_k=30) -> dict:
        # tree에서 Level_1 노드 top-K 선택

    def extract_question_elements(self, question, options, cues) -> list[str]:
        # question + options + cues → flat element list
```

### 수정 파일

- `components/tree_filter.py`: `build()` 에 `semantic_scores` 파라미터 추가
  - semantic score가 있으면 leaf의 on/off, priority 순서를 semantic score로 재정렬
  - 선택된 L1 노드의 leaves만 활성화

- `pipelines/tree_search.py`: STAGE 1.5로 semantic matching 단계 추가
  - `semantic_matcher`가 components에 있으면 자동 활성화
  - 결과를 `tree_filter.build()`에 전달
  - 출력 JSON에 `semantic_match` 필드 추가 (디버깅용)

- `solver.py`: `create_components()`에서 config에 `semantic_matcher` 있으면 생성

### Config 파일

```yaml
components:
  semantic_matcher:
    model_path: /scratch2/youngbeom/ckpt/Qwen3-Embedding-0.6B
    top_k: 30          # 선택할 L1 노드 수
    batch_size: 64
    score_mode: sum     # sum: q_element별 max sim 합산
```

### 파이프라인 흐름

```
STAGE 1: Query decompose → cues, time info
STAGE 1.5: Semantic matching
  ├─ question + options + cues → q_elements (flat list)
  ├─ Level_1 nodes의 key_elements embedding
  ├─ cosine similarity matrix → 노드별 총합 score
  └─ Top-30 L1 노드 선택
STAGE 2: Build filtered tree (semantic scores로 priority 재정렬)
STAGE 3: Initial target selection
STAGE 4: Hop loop (judge + navigate)
```

## 실험 계획

1. **Video-MME subset**: `config/video_mme_subset_tree_search_semantic.yaml`
   - 기존 visual3 결과 (exact match, 46.7%) vs semantic matching

2. **LVBench subset**: `config/lvbench_tree_search_semantic.yaml`
   - 기존 결과 vs semantic matching

3. 비교 포인트:
   - 활성화된 leaf 수 변화 (exact → semantic)
   - Hop 수 변화 (더 좋은 초기 타겟 → 적은 hop?)
   - accuracy 변화
   - top_matches 분석 (실제 어떤 매칭이 일어나는지)

## 관련 파일

| 파일 | 역할 |
|------|------|
| `components/semantic_matcher.py` | Qwen3-Embedding 기반 시맨틱 매칭 (NEW) |
| `components/tree_filter.py` | Top-down filtered tree + semantic score 지원 (MODIFIED) |
| `pipelines/tree_search.py` | Tree search pipeline + STAGE 1.5 (MODIFIED) |
| `solver.py` | SemanticMatcher 컴포넌트 생성 (MODIFIED) |
| `config/*_semantic.yaml` | 시맨틱 매칭 config 2개 (NEW) |
| `DyHiStreamMem/poc/stage2_evolving_cluster_v8_sync.py` | 메모리 빌더 (참고) |
| `DyHiStreamMem/memory_builder/stage2/stage2_clustering_final.py` | Clustering 기반 빌더 (참고) |

## Multi-GPU 병렬 실행

### 사용법

```bash
# GPU 4개로 병렬 처리 (비디오를 GPU 수로 round-robin 분할)
bash 0227_my_idea_solver.sh lvbench 0,1,2,3

# 특정 문제 하나만 풀기
bash 0227_my_idea_solver.sh question 503

# 특정 비디오만
bash 0227_my_idea_solver.sh video 2sriHX3PbXw
```

### 구조

```
Main process:
  ├─ Load questions → video list
  ├─ Auto-version output dir (공유)
  ├─ Split videos across GPUs (round-robin)
  └─ Spawn N workers (torch.multiprocessing)

GPU Worker (per GPU):
  ├─ Load VLM model on cuda:N (.to(device), NOT device_map="auto")
  ├─ Load Qwen3-Embedding on same GPU
  ├─ Process assigned videos sequentially
  └─ Save results to shared by_qid/ (파일 기반, 락 불필요)

After all workers done:
  └─ Auto-aggregate → summary.json
```

### 핵심 포인트
- `device_map="auto"` 대신 `.to(f"cuda:{gpu_id}")` → GPU 독점
- 각 프로세스가 독립적 모델 인스턴스 → 메모리 격리
- 결과는 `by_qid/` 파일 기반 → lock-free 병렬 쓰기
- Embedding model (0.6B, ~1.2GB)은 VLM과 같은 GPU에 올림 → 추가 메모리 미미

### CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--gpus 0,1,2,3` | Multi-GPU 병렬 (비디오 분할) |
| `--question_id 503` | 단일 문제만 처리 |
| `--video_id vid_name` | 단일 비디오만 처리 |
| `--list_questions` | 전체 문제 목록 출력 (qid + video_id) |
| `--list_videos` | 전체 비디오 목록 출력 |

## 향후 확장

- [ ] Level_2 매칭: Level_1 대신 Level_2에서 먼저 top-K 선택 → 하위 L1 전부 활성화
- [ ] Hybrid scoring: exact match score + semantic score 가중합
- [ ] 카테고리별 가중치: question_type에 따라 actions 우선 vs objects 우선
- [ ] Batch search (10개씩): 현재 initial target을 10개 → 다음 10개 순차 탐색
- [ ] Graduated visual: light scan (3 frames × 10 segments) → deep scan (10 frames × top 3)
