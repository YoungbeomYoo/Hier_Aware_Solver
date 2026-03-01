"""
Prompt Templates — Plug-and-play LLM prompt 관리

각 컴포넌트별 교체 가능한 prompt template.
PROMPT 변수를 정의하고, config에서 이름으로 참조하여 사용.

Usage:
    from prompts import get_prompt
    template = get_prompt("decompose", "detailed")
"""

from importlib import import_module

# Registry: (component_name, variant_name) → module path
_REGISTRY = {
    ("decompose", "default"): "prompts.decompose_default",
    ("decompose", "detailed"): "prompts.decompose_detailed",
    ("solvability", "strict"): "prompts.solvability_strict",
    ("solvability", "relaxed"): "prompts.solvability_relaxed",
    ("solvability", "videolucy"): "prompts.solvability_videolucy",
    ("solvability", "videolucy_force"): "prompts.solvability_videolucy_force",
    # judge aliases (solver.py uses get_prompt("judge", ...))
    ("judge", "strict"): "prompts.solvability_strict",
    ("judge", "relaxed"): "prompts.judge_relaxed",
    ("judge", "videolucy"): "prompts.solvability_videolucy",
    ("judge", "videolucy_force"): "prompts.solvability_videolucy_force",
    ("vlm", "answer_only"): "prompts.vlm_answer_only",
    ("vlm", "with_confidence"): "prompts.vlm_with_confidence",
    ("navigate", "single_hop"): "prompts.navigate_single_hop",
    ("leaf_select", "budget"): "prompts.leaf_select_budget",
    # Cognitive pipeline prompts
    ("query_classify", "default"): "prompts.query_classify_default",
    ("query_classify", "detailed"): "prompts.query_classify_detailed",
    ("elimination", "default"): "prompts.elimination_default",
    ("elimination", "strict"): "prompts.elimination_strict",
    ("uncertainty", "default"): "prompts.uncertainty_default",
    ("uncertainty", "conservative"): "prompts.uncertainty_conservative",
    # Two-stage visual prompts
    ("scout", "default"): "prompts.scout_caption",
    ("focus_select", "default"): "prompts.focus_select",
    ("focus_caption", "default"): "prompts.focus_caption",
    ("focus_caption", "videolucy_style"): "prompts.focus_caption",
    # Vgent-style structured reasoning prompts
    ("sub_question", "generate"): "prompts.sub_question_generate",
    ("sub_question", "answer"): "prompts.sub_question_answer",
    ("aggregate", "info"): "prompts.aggregate_info",
}


def get_prompt(component: str, variant: str) -> str:
    """Get prompt template by component name and variant.

    Args:
        component: "decompose", "solvability", "vlm", "navigate", "leaf_select"
        variant: "default", "detailed", "strict", etc.

    Returns:
        Prompt template string with format placeholders.
    """
    key = (component, variant)
    if key not in _REGISTRY:
        raise ValueError(f"Unknown prompt: {component}/{variant}. Available: {list(_REGISTRY.keys())}")
    module = import_module(_REGISTRY[key])
    return module.PROMPT
