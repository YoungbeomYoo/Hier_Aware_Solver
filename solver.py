#!/usr/bin/env python3
"""
Unified Benchmark Solver — Entry Point

Memory-based Video QA solver for HD-EPIC, Video-MME, LVBench.

Usage:
    # Single dataset
    python solver.py --config config/lvbench.yaml
    python solver.py --config config/video_mme.yaml
    python solver.py --config config/hd_epic.yaml

    # Dry run (no model loading)
    python solver.py --config config/lvbench.yaml --dry_run

    # Override output directory
    python solver.py --config config/lvbench.yaml --output_dir ./output/experiment_1
"""

import sys
import os
import argparse
import json
from importlib import import_module

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str, overrides: dict = None) -> dict:
    """YAML 또는 JSON config 로딩."""
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except ImportError:
            print("PyYAML not installed. Install with: pip install pyyaml")
            print("Falling back to JSON config.")
            config_path = config_path.replace(".yaml", ".json").replace(".yml", ".json")
            with open(config_path, "r") as f:
                config = json.load(f)
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    if overrides:
        for key, val in overrides.items():
            if val is not None:
                # Support nested keys with dots: "paths.output_dir"
                keys = key.split(".")
                d = config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = val

    return config


def _auto_version_dir(base_dir: str) -> str:
    """output_dir에 자동 버전 부여: base_dir/v1, v2, v3, ..."""
    if not os.path.exists(base_dir):
        versioned = os.path.join(base_dir, "v1")
        os.makedirs(versioned, exist_ok=True)
        return versioned

    # 기존 vN 폴더 찾기
    existing = []
    for name in os.listdir(base_dir):
        if name.startswith("v") and name[1:].isdigit():
            existing.append(int(name[1:]))

    next_v = max(existing, default=0) + 1
    versioned = os.path.join(base_dir, f"v{next_v}")
    os.makedirs(versioned, exist_ok=True)
    return versioned


def create_adapter(config: dict, skip_auto_version: bool = False):
    """Config에서 데이터셋 어댑터 생성."""
    dataset = config["dataset"].lower().replace("-", "_")

    adapter_config = config.get("paths", {})
    base_output = config.get("paths", {}).get("output_dir", f"./output/{dataset}")
    if skip_auto_version:
        # SLURM 등 외부에서 이미 output_dir을 결정한 경우
        adapter_config["output_dir"] = base_output
        os.makedirs(base_output, exist_ok=True)
    else:
        adapter_config["output_dir"] = _auto_version_dir(base_output)

    if dataset in ("hd_epic", "hdepic"):
        from adapters.hd_epic import HDEpicAdapter
        return HDEpicAdapter(adapter_config)
    elif dataset in ("video_mme", "videomme"):
        from adapters.video_mme import VideoMMEAdapter
        return VideoMMEAdapter(adapter_config)
    elif dataset in ("lvbench",):
        from adapters.lvbench import LVBenchAdapter
        return LVBenchAdapter(adapter_config)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: hd_epic, video_mme, lvbench")


def load_model(config: dict, dry_run: bool = False):
    """VLM 모델 로딩."""
    if dry_run:
        return None, None

    import torch
    model_config = config.get("model", {})
    model_path = model_config.get("path", "")
    model_type = model_config.get("type", "qwen3vl").lower()
    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))
    attn_impl = model_config.get("attn_impl", "flash_attention_2")

    print(f"Loading model: {model_path} (type={model_type})")

    if model_type in ("qwen3vl", "qwen3_vl"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype,
            attn_implementation=attn_impl, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif model_type in ("qwen25vl", "qwen2_5_vl", "qwen2.5vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype,
            attn_implementation=attn_impl, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)

    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: qwen3vl, qwen25vl")

    model.eval()
    print("Model loaded.")
    return model, processor


def create_components(config: dict, model, processor, dry_run: bool = False) -> dict:
    """Config에서 컴포넌트 인스턴스 생성."""
    from components.vlm import TextOnlyLLM, VisionVLM, SimpleVLM
    from components.memory_ops import LeafFlattener, HierarchicalNavigator, MemoryContextFormatter
    from components.query_decomposer import QueryDecomposer
    from components.filters import RuleBasedFilter, LLMLeafSelector
    from components.solvability import SolvabilityChecker, ForcedAnswerFallback
    from components.time_router import TimeRouter
    from components.frame_loader import TargetedFrameLoader, UniformFrameLoader
    from components.coverage import CoverageAnalyzer
    from components.query_analyzer import QueryAnalyzer
    from components.metadata_filter import MetadataTargetedFilter
    from components.spreading_activation import SpreadingActivation
    from components.uncertainty_checker import UncertaintyChecker
    from components.elimination_reasoner import EliminationReasoner
    from components.hierarchical_scorer import HierarchicalScorer
    from components.visual_observer import HistoryAwareObserver
    from components.tree_filter import FilteredTreeBuilder
    from components.context_assembler import BudgetContextAssembler
    from components.judge import SolvabilityJudge
    from components.judge_visual import VisualJudge
    from components.history_compactor import HistoryCompactor
    from components.history_accumulator import HistoryAccumulator
    from components.tree_planner import TreePlanner
    from components.token_utils import TokenBudget

    comp_config = config.get("components", {})
    pipeline_params = config.get("pipeline_params", {})

    # --- Token budget (tokenizer-based if model loaded, else char fallback) ---
    tokenizer = processor.tokenizer if processor and hasattr(processor, 'tokenizer') else None
    tb = TokenBudget(tokenizer)

    # --- Prompt loading helper ---
    def get_prompt_template(component_name: str) -> str | None:
        prompt_name = comp_config.get(component_name, {}).get("prompt")
        if prompt_name:
            from prompts import get_prompt
            return get_prompt(component_name.replace("_checker", "").replace("_decomposer", ""),
                            prompt_name)
        return None

    components = {}

    # --- LLM backbone ---
    llm = None
    if model and processor:
        llm = TextOnlyLLM(model, processor)
        components["llm"] = llm

        # --- VisionVLM ---
        vlm_config = comp_config.get("vlm_inference", {})
        components["vision_vlm"] = VisionVLM(
            model, processor,
            image_token_size=vlm_config.get("image_token_size", 256),
            memory_budget=vlm_config.get("memory_budget", 20000),
        )

        # --- SimpleVLM ---
        components["simple_vlm"] = SimpleVLM(model, processor)

    # --- LLM-based components (llm=None in dry_run → rule-based fallback) ---
    decompose_prompt = None
    dc = comp_config.get("query_decomposer", {})
    if dc.get("prompt"):
        from prompts import get_prompt
        decompose_prompt = get_prompt("decompose", dc["prompt"])
    components["decomposer"] = QueryDecomposer(llm, decompose_prompt)

    nav_prompt = None
    nc = comp_config.get("navigator", {})
    if nc.get("prompt"):
        from prompts import get_prompt
        nav_prompt = get_prompt("navigate", nc["prompt"])
    components["navigator"] = HierarchicalNavigator(llm, nav_prompt)

    sel_prompt = None
    sc = comp_config.get("leaf_selector", {})
    if sc.get("prompt"):
        from prompts import get_prompt
        sel_prompt = get_prompt("leaf_select", sc["prompt"])
    components["leaf_selector"] = LLMLeafSelector(
        llm, sel_prompt,
        format_leaf_fn=MemoryContextFormatter().format_leaf_compact,
    )

    solv_prompt = None
    svc = comp_config.get("solvability_checker", {})
    if svc.get("prompt"):
        from prompts import get_prompt
        solv_prompt = get_prompt("solvability", svc["prompt"])
    components["solvability"] = SolvabilityChecker(llm, solv_prompt)

    components["fallback"] = ForcedAnswerFallback(llm)

    # --- Cognitive Pipeline Components ---
    qa_prompt = None
    qac = comp_config.get("query_analyzer", {})
    qa_llm = llm if not qac.get("skip_llm") else None  # A0: skip LLM classification
    if qac.get("prompt"):
        from prompts import get_prompt
        qa_prompt = get_prompt("query_classify", qac["prompt"])
    components["query_analyzer"] = QueryAnalyzer(qa_llm, qa_prompt)

    unc_prompt = None
    ucc = comp_config.get("uncertainty_checker", {})
    if ucc.get("prompt"):
        from prompts import get_prompt
        unc_prompt = get_prompt("uncertainty", ucc["prompt"])
    components["uncertainty_checker"] = UncertaintyChecker(llm, unc_prompt)

    elim_prompt = None
    erc = comp_config.get("elimination_reasoner", {})
    if erc.get("prompt"):
        from prompts import get_prompt
        elim_prompt = get_prompt("elimination", erc["prompt"])
    components["elimination_reasoner"] = EliminationReasoner(llm, elim_prompt)

    # --- HistoryAwareObserver (needs llm + vision_vlm + frame_loader) ---
    # Created after frame_loader below, but we set up config here
    observer_llm_fn = None
    if llm:
        observer_llm_fn = lambda prompt, max_tokens=400: llm.reason(prompt, max_tokens=max_tokens)

    # --- Tree Search Pipeline Components ---
    judge_llm_fn = None
    if llm:
        judge_llm_fn = lambda prompt, max_tokens=400: llm.reason(prompt, max_tokens=max_tokens)

    jc = comp_config.get("judge", {})
    judge_prompt = None
    if jc.get("prompt"):
        from prompts import get_prompt
        judge_prompt = get_prompt("judge", jc["prompt"])
    components["judge"] = SolvabilityJudge(
        judge_llm_fn, judge_prompt, token_budget=tb,
        context_budget=jc.get("context_budget", 20000),
        history_budget=jc.get("history_budget", 4000),
        answer_judge=jc.get("answer_judge", False),
    )

    # --- VisualJudge (judge + query-aware captioning, assembled after frame_loader) ---
    # Only create if judge_visual is explicitly in config (not for text-only mode)
    if "judge_visual" in comp_config:
        jv_config = comp_config.get("judge_visual", {})
        components["_judge_visual_config"] = jv_config  # deferred, assembled below

    tf_config = comp_config.get("tree_filter", {})
    components["tree_filter"] = FilteredTreeBuilder(
        match_threshold=tf_config.get("match_threshold", 1),
        use_key_elements=pipeline_params.get("use_key_elements", True),
    )

    ca_config = comp_config.get("context_assembler", {})
    components["context_assembler"] = BudgetContextAssembler(
        max_budget=ca_config.get("max_text_budget",
                                 pipeline_params.get("max_text_budget", 20000)),
        token_budget=tb,
        use_captions=pipeline_params.get("use_captions", True),
        use_key_elements=pipeline_params.get("use_key_elements", True),
    )

    hc_config = comp_config.get("history_compactor", {})
    history_config = comp_config.get("history", {})
    history_mode = history_config.get("mode", "compact")

    if history_mode == "accumulate":
        components["history_accumulator"] = HistoryAccumulator(
            token_budget=tb,
            max_total_tokens=history_config.get("max_total_tokens", 50000),
        )
    else:
        components["history_compactor"] = HistoryCompactor(
            max_observation_tokens=hc_config.get("max_observation_tokens", 8000),
            max_state_tokens=hc_config.get("max_state_tokens", 800),
            token_budget=tb,
        )

    # --- Non-LLM Cognitive/Composable Components ---
    components["metadata_filter"] = MetadataTargetedFilter()
    components["spreading_activation"] = SpreadingActivation()
    hs_config = comp_config.get("hierarchical_scorer", {})
    components["hierarchical_scorer"] = HierarchicalScorer(
        match_threshold=hs_config.get("match_threshold", 1),
        use_key_elements=pipeline_params.get("use_key_elements", True),
    )

    # --- Non-LLM components ---
    components["flattener"] = LeafFlattener()
    components["formatter"] = MemoryContextFormatter()
    components["rule_filter"] = RuleBasedFilter()
    components["time_router"] = TimeRouter()
    components["coverage"] = CoverageAnalyzer()

    # --- Frame loaders ---
    fl_config = comp_config.get("frame_loader", {})
    components["frame_loader"] = TargetedFrameLoader(
        max_frames=fl_config.get("max_frames", pipeline_params.get("max_frames", 32)),
    )
    components["uniform_loader"] = UniformFrameLoader(
        n_frames=fl_config.get("n_frames", 32),
    )

    # --- HistoryAwareObserver (assembled after all deps ready) ---
    vision_vlm = components.get("vision_vlm")
    frame_loader = components.get("frame_loader")
    if observer_llm_fn or vision_vlm:
        components["visual_observer"] = HistoryAwareObserver(
            llm_fn=observer_llm_fn,
            vision_vlm=vision_vlm,
            frame_loader=frame_loader,
        )

    # --- VisualJudge (assembled after vision_vlm + frame_loader ready) ---
    jv_config = components.pop("_judge_visual_config", None)
    if jv_config is not None:
        # Load C-axis aligned prompts for VisualJudge if specified
        jv_judge_prompt = None
        jv_rejudge_prompt = None
        jv_prompt_name = jv_config.get("prompt")
        if jv_prompt_name:
            try:
                jv_mod = import_module(f"prompts.judge_visual_{jv_prompt_name}")
                jv_judge_prompt = jv_mod.JUDGE_VISUAL_PROMPT
                jv_rejudge_prompt = jv_mod.REJUDGE_PROMPT
            except (ImportError, AttributeError):
                print(f"  [warn] judge_visual prompt '{jv_prompt_name}' not found, using default")

        components["judge_visual"] = VisualJudge(
            llm_fn=judge_llm_fn,
            vision_vlm=vision_vlm,
            frame_loader=frame_loader,
            judge_prompt=jv_judge_prompt,
            rejudge_prompt=jv_rejudge_prompt,
            token_budget=tb,
            context_budget=jv_config.get("context_budget", 20000),
            history_budget=jv_config.get("history_budget", 4000),
            caption_budget=jv_config.get("caption_budget", 3000),
            skip_captioning=jv_config.get("skip_captioning", False),
            force_visual=jv_config.get("force_visual", False),
        )

    # --- TreePlanner (optional, for planned tree search) ---
    tp_config = comp_config.get("tree_planner", {})
    if tp_config or config.get("pipeline_params", {}).get("use_planner"):
        planner_llm_fn = None
        if llm:
            planner_llm_fn = lambda prompt, max_tokens=300: llm.reason(
                prompt, max_tokens=max_tokens,
            )
        components["tree_planner"] = TreePlanner(
            llm_fn=planner_llm_fn,
            token_budget=tb,
            max_overview_tokens=tp_config.get("max_overview_tokens", 15000),
            max_regions=tp_config.get("max_regions", 10),
        )

    # --- SemanticMatcher (optional, for embedding-based key_elements matching) ---
    sm_config = comp_config.get("semantic_matcher", {})
    if sm_config:
        from components.semantic_matcher import SemanticMatcher
        components["semantic_matcher"] = SemanticMatcher(
            model_path=sm_config.get("model_path",
                                      "/scratch2/youngbeom/ckpt/Qwen3-Embedding-0.6B"),
            device=sm_config.get("device"),  # None → auto, or "cuda:0" etc.
            top_k=sm_config.get("top_k", 30),
            batch_size=sm_config.get("batch_size", 64),
            score_mode=sm_config.get("score_mode", "sum"),
            use_key_elements=pipeline_params.get("use_key_elements", True),
        )

    return components


def create_pipeline(config: dict, components: dict, adapter):
    """Config에서 파이프라인 생성."""
    pipeline_name = config.get("pipeline", "memory_only").lower()
    pipeline_params = config.get("pipeline_params", {})

    # Merge pipeline_params into a unified config for the pipeline
    pipe_config = {**pipeline_params, "cached": config.get("cached", True)}

    # Pass two-stage visual config to pipeline
    jv_config = config.get("components", {}).get("judge_visual", {})
    if jv_config.get("two_stage"):
        pipe_config["two_stage_visual"] = True
        pipe_config["scout_frames_per_region"] = jv_config.get("scout_frames_per_region", 3)

    # Pass history budget for accumulator mode
    history_config = config.get("components", {}).get("history", {})
    if history_config.get("mode") == "accumulate":
        pipe_config["history_budget"] = history_config.get("history_budget", 20000)

    if pipeline_name == "memory_only":
        from pipelines.memory_only import MemoryOnlyPipeline
        return MemoryOnlyPipeline(components, adapter, pipe_config)
    elif pipeline_name == "routed":
        from pipelines.routed import RoutedPipeline
        return RoutedPipeline(components, adapter, pipe_config)
    elif pipeline_name == "agentic":
        from pipelines.agentic import AgenticPipeline
        return AgenticPipeline(components, adapter, pipe_config)
    elif pipeline_name == "cognitive":
        from pipelines.cognitive import CognitivePipeline
        return CognitivePipeline(components, adapter, pipe_config)
    elif pipeline_name == "composable":
        from pipelines.composable import ComposablePipeline
        return ComposablePipeline(components, adapter, pipe_config)
    elif pipeline_name == "tree_search":
        from pipelines.tree_search import TreeSearchPipeline
        return TreeSearchPipeline(components, adapter, pipe_config)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Supported: memory_only, routed, agentic, cognitive, composable, tree_search")


def _gpu_worker(gpu_id: int, video_ids: list[str], config: dict, output_dir: str):
    """단일 GPU에서 할당된 비디오들을 처리하는 워커.

    각 워커가 독립적으로 모델을 로드하고, 할당된 비디오를 순차 처리.
    결과는 공유 output_dir/by_qid/에 저장 (파일 기반, 락 불필요).
    """
    import torch
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Starting worker with {len(video_ids)} videos on {device}")

    # Force model to specific GPU
    config_copy = json.loads(json.dumps(config))  # deep copy

    # Override device_map to this specific GPU
    if "model" not in config_copy:
        config_copy["model"] = {}

    # Output dir은 공유 (by_qid/ 기반 결과 저장)
    config_copy["paths"]["output_dir"] = output_dir

    # Adapter (공유 output dir 사용, auto-versioning 건너뛰기)
    dataset = config_copy["dataset"].lower().replace("-", "_")
    adapter_config = config_copy.get("paths", {})
    adapter_config["output_dir"] = output_dir

    if dataset in ("hd_epic", "hdepic"):
        from adapters.hd_epic import HDEpicAdapter
        adapter = HDEpicAdapter(adapter_config)
    elif dataset in ("video_mme", "videomme"):
        from adapters.video_mme import VideoMMEAdapter
        adapter = VideoMMEAdapter(adapter_config)
    elif dataset in ("lvbench",):
        from adapters.lvbench import LVBenchAdapter
        adapter = LVBenchAdapter(adapter_config)
    else:
        print(f"[GPU {gpu_id}] Unknown dataset: {dataset}")
        return

    # Load model on specific GPU
    model_config = config_copy.get("model", {})
    model_path = model_config.get("path", "")
    model_type = model_config.get("type", "qwen3vl").lower()
    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))
    attn_impl = model_config.get("attn_impl", "flash_attention_2")

    print(f"[GPU {gpu_id}] Loading model: {model_path}")

    if model_type in ("qwen3vl", "qwen3_vl"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_type in ("qwen25vl", "qwen2_5_vl", "qwen2.5vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        print(f"[GPU {gpu_id}] Unknown model type: {model_type}")
        return

    model.eval()
    print(f"[GPU {gpu_id}] Model loaded on {device}")

    # Create components (semantic_matcher도 같은 GPU에 로드)
    sm_config = config_copy.get("components", {}).get("semantic_matcher", {})
    if sm_config:
        sm_config["device"] = device  # force embedding model to same GPU

    components = create_components(config_copy, model, processor)
    pipeline = create_pipeline(config_copy, components, adapter)

    # Process assigned videos
    for i, video_id in enumerate(video_ids):
        print(f"[GPU {gpu_id}] ({i+1}/{len(video_ids)}) Processing video: {video_id}")
        try:
            pipeline.run_video(video_id)
        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR on video {video_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"[GPU {gpu_id}] Done! Processed {len(video_ids)} videos.")


def _run_multi_gpu(args, config: dict):
    """Multi-GPU 병렬 실행: 비디오를 GPU 수로 나눠서 각각 독립 프로세스."""
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    print(f"Multi-GPU mode: {n_gpus} GPUs {gpu_ids}")
    print(f"Dataset: {config['dataset']}")
    print(f"Pipeline: {config.get('pipeline', 'memory_only')}")

    # Auto-version output dir (한 번만, 공유)
    base_output = config.get("paths", {}).get("output_dir", "./output/default")
    output_dir = _auto_version_dir(base_output)
    print(f"Output: {output_dir}")

    # Load questions to get video list (adapter만 임시 생성)
    adapter_config = config.get("paths", {}).copy()
    adapter_config["output_dir"] = output_dir
    dataset = config["dataset"].lower().replace("-", "_")

    if dataset in ("hd_epic", "hdepic"):
        from adapters.hd_epic import HDEpicAdapter
        adapter = HDEpicAdapter(adapter_config)
    elif dataset in ("video_mme", "videomme"):
        from adapters.video_mme import VideoMMEAdapter
        adapter = VideoMMEAdapter(adapter_config)
    elif dataset in ("lvbench",):
        from adapters.lvbench import LVBenchAdapter
        adapter = LVBenchAdapter(adapter_config)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    questions_by_video = adapter.load_questions()
    all_videos = sorted(questions_by_video.keys())
    total_q = sum(len(qs) for qs in questions_by_video.values())
    print(f"  {len(all_videos)} videos, {total_q} questions total")

    # Split videos across GPUs (round-robin)
    video_chunks = [[] for _ in range(n_gpus)]
    for i, vid in enumerate(all_videos):
        video_chunks[i % n_gpus].append(vid)

    for i, (gid, chunk) in enumerate(zip(gpu_ids, video_chunks)):
        print(f"  GPU {gid}: {len(chunk)} videos")

    # Spawn workers
    processes = []
    for gid, chunk in zip(gpu_ids, video_chunks):
        if not chunk:
            continue
        p = mp.Process(
            target=_gpu_worker,
            args=(gid, chunk, config, output_dir),
        )
        p.start()
        processes.append((gid, p))

    # Wait for all workers
    for gid, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"  [WARNING] GPU {gid} worker exited with code {p.exitcode}")

    print(f"\nAll workers done! Results in: {output_dir}")

    # Auto-aggregate
    from aggregate import aggregate
    aggregate(output_dir)


def _fix_dash_args():
    """video_id, question_id 값이 '-'로 시작할 때 argparse가 플래그로 오해하는 문제 fix.

    --video_id -QuCz7kxBr8 → --video_id=-QuCz7kxBr8 로 변환.
    """
    fix_flags = {"--video_id", "--question_id", "--video_list", "--question_list", "--output_dir"}
    new_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in fix_flags and i + 1 < len(sys.argv):
            # --flag value → --flag=value (argparse가 value를 플래그로 오해 방지)
            new_argv.append(f"{arg}={sys.argv[i + 1]}")
            i += 2
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv


def main():
    _fix_dash_args()

    parser = argparse.ArgumentParser(description="Unified Benchmark Solver")
    parser.add_argument("--config", type=str, required=True, help="Config file path (YAML/JSON)")
    parser.add_argument("--dry_run", action="store_true", help="Skip model loading")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--cached", type=int, default=None, help="Skip cached results (0/1)")
    parser.add_argument("--pipeline", type=str, default=None, help="Override pipeline (memory_only/routed/agentic)")
    parser.add_argument("--max_hops", type=int, default=None, help="Override max hops (agentic)")
    parser.add_argument("--max_frames", type=int, default=None, help="Override max frames")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Process single video only (for parallel SLURM jobs)")
    parser.add_argument("--question_id", type=str, default=None,
                        help="Process single question only (e.g. --question_id 503)")
    parser.add_argument("--video_list", type=str, default=None,
                        help="File with video IDs (one per line) to process")
    parser.add_argument("--question_list", type=str, default=None,
                        help="File with question IDs (one per line) to process subset")
    parser.add_argument("--list_videos", action="store_true",
                        help="List all video IDs and exit (for SLURM array setup)")
    parser.add_argument("--list_questions", action="store_true",
                        help="List all question IDs and exit (for parallel setup)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Multi-GPU parallel: comma-separated GPU IDs (e.g. --gpus 0,1,2,3)")
    args = parser.parse_args()

    # Build overrides
    overrides = {}
    if args.output_dir:
        overrides["paths.output_dir"] = args.output_dir
    if args.cached is not None:
        overrides["cached"] = bool(args.cached)
    if args.pipeline:
        overrides["pipeline"] = args.pipeline
    if args.max_hops:
        overrides["pipeline_params.max_hops"] = args.max_hops
    if args.max_frames:
        overrides["pipeline_params.max_frames"] = args.max_frames

    config = load_config(args.config, overrides)

    # --- Multi-GPU parallel mode: spawn workers and exit ---
    if args.gpus:
        _run_multi_gpu(args, config)
        return [], {}

    print(f"Dataset: {config['dataset']}")
    print(f"Pipeline: {config.get('pipeline', 'memory_only')}")
    print(f"Output: {config.get('paths', {}).get('output_dir', './output')}")
    print()

    # --output_dir override → auto-versioning 건너뛰기
    adapter = create_adapter(config, skip_auto_version=bool(args.output_dir))

    # --- List-only modes (no model loading needed) ---
    if args.list_videos or args.list_questions:
        questions_by_video = adapter.load_questions()
        if args.list_videos:
            for vid in sorted(questions_by_video.keys()):
                print(vid)
        elif args.list_questions:
            for vid in sorted(questions_by_video.keys()):
                for q in questions_by_video[vid]:
                    print(f"{q['question_id']}\t{vid}")
        return [], {}

    model, processor = load_model(config, dry_run=args.dry_run)
    components = create_components(config, model, processor, dry_run=args.dry_run)
    pipeline = create_pipeline(config, components, adapter)

    # --- Dry run: setup 검증만 하고 종료 ---
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — Setup Validation")
        print("=" * 60)

        # Validate questions can be loaded
        questions_by_video = adapter.load_questions()
        total_q = sum(len(qs) for qs in questions_by_video.values())
        print(f"  Videos: {len(questions_by_video)}")
        print(f"  Questions: {total_q}")

        # Check how many have memory
        mem_found = 0
        mem_missing = 0
        for vid in list(questions_by_video.keys())[:20]:  # Sample first 20
            mem = adapter.load_memory(vid)
            if mem:
                mem_found += 1
            else:
                mem_missing += 1
        print(f"  Memory check (sampled {mem_found + mem_missing}): "
              f"{mem_found} found, {mem_missing} missing")

        # Show components
        print(f"\n  Components created: {sorted(components.keys())}")
        llm_components = [k for k, v in components.items()
                          if hasattr(v, 'llm_fn') and getattr(v, 'llm_fn', None) is not None]
        print(f"  LLM-powered: {llm_components if llm_components else '(none — model not loaded)'}")
        print(f"\n  Pipeline: {config.get('pipeline', 'memory_only')}")
        print(f"  Output dir: {adapter.output_dir}")
        print("\nDry run complete. Use without --dry_run to actually solve.")
        return [], {}

    # --- Single question mode ---
    if args.question_id:
        video_id = args.video_id or ""
        print(f"Single question mode: {args.question_id}")
        result = pipeline.run_question(video_id, args.question_id)
        print(f"\nDone! Results saved to: {adapter.output_dir}")
        return [result] if result else [], {}

    # --- Single video mode ---
    if args.video_id:
        print(f"Single video mode: {args.video_id}")
        results = pipeline.run_video(args.video_id)
        print(f"\nDone! Results saved to: {adapter.output_dir}")
        return results, {}

    # --- Video list mode ---
    if args.video_list:
        with open(args.video_list) as f:
            video_ids = [line.strip() for line in f if line.strip()]
        print(f"Video list mode: {len(video_ids)} videos from {args.video_list}")
        results, summary = pipeline.run_all(video_ids=video_ids)
        print(f"\nDone! Results saved to: {adapter.output_dir}")
        return results, summary

    # --- Question list mode (subset ablation) ---
    if args.question_list:
        with open(args.question_list) as f:
            question_ids = set(line.strip() for line in f if line.strip())
        print(f"Question list mode: {len(question_ids)} questions from {args.question_list}")
        results, summary = pipeline.run_all(question_ids=question_ids)
        print(f"\nDone! Results saved to: {adapter.output_dir}")
        return results, summary

    # --- Full run ---
    results, summary = pipeline.run_all()

    print(f"\nDone! Results saved to: {adapter.output_dir}")
    return results, summary


if __name__ == "__main__":
    main()
