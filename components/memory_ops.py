"""
Memory Operations — Memory Tree 조작 모듈

- LeafFlattener: streaming_memory_tree → flat leaf list
- HierarchicalNavigator: top-down tree search (LLM 기반)
- MemoryContextFormatter: memory를 텍스트로 포매팅
"""

import json
from components.time_utils import time_to_secs, secs_to_time_str


# ============================================================
# LeafFlattener
# ============================================================

class LeafFlattener:
    """streaming_memory_tree에서 모든 leaf node를 flat list로 추출.

    Level_1 노드의 children이 leaf (start_time/end_time/caption/summary).
    상위 레벨은 recursive fallback으로 처리.
    """

    def flatten(self, tree: dict) -> list[dict]:
        """Returns list of {"leaf": dict, "leaf_id": (start, end), "parent_summary": str}"""
        seen = set()
        leaves = []

        def _add_leaf(leaf, parent_summary=""):
            if "start_time" not in leaf:
                return
            leaf_id = (float(leaf["start_time"]), float(leaf["end_time"]))
            if leaf_id in seen:
                return
            seen.add(leaf_id)
            leaves.append({
                "leaf": leaf,
                "leaf_id": leaf_id,
                "parent_summary": parent_summary,
            })

        def _extract_recursive(node, parent_summary=""):
            children = node.get("children", [])
            if not children:
                _add_leaf(node, parent_summary)
                return
            node_summary = node.get("summary", "")
            if isinstance(children[0], dict) and "level" in children[0]:
                for child in children:
                    _extract_recursive(child, node_summary)
            else:
                for child in children:
                    _add_leaf(child, node_summary)

        # Primary: Level_1 children are leaves
        if "Level_1" in tree:
            for node in tree["Level_1"]:
                parent_summary = node.get("summary", "")
                for child in node.get("children", []):
                    _add_leaf(child, parent_summary)

        # Fallback: walk higher levels recursively
        for lvl_key in sorted(tree.keys()):
            if lvl_key == "Level_1":
                continue
            for node in tree[lvl_key]:
                _extract_recursive(node)

        leaves.sort(key=lambda x: x["leaf_id"][0])
        return leaves


# ============================================================
# HierarchicalNavigator
# ============================================================

class HierarchicalNavigator:
    """Top-down memory tree search using LLM.

    Cue 기반으로 가장 관련성 높은 branch를 찾아 localized memory를 반환.
    """

    DEFAULT_PROMPT = """You are a Hierarchical Memory Navigator.
Your goal is to find the most relevant video segment branch based on the search cues.

[Search Cues]
{cues}

[Level 1 Memory Nodes]
{node_summaries}

Select the ID of the ONE node that most likely contains the answer. If multiple, select the best one.
Keep "thought" under 20 words.
Output ONLY valid JSON:
{{
    "thought": "One sentence reason.",
    "selected_id": <int>
}}"""

    def __init__(self, llm_fn, prompt_template=None):
        """
        Args:
            llm_fn: callable(prompt_text, max_tokens) -> dict
            prompt_template: str with {cues}, {node_summaries} placeholders
        """
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def navigate(self, memory_tree: dict, cues: list[str]) -> tuple[str, str, dict]:
        """Top-down hierarchical search.

        Args:
            memory_tree: {"video_id": {..., "memory": {"steps": [...]}}} or streaming_memory_tree
            cues: list of search keywords

        Returns:
            (localized_memory_json, target_time_segments, hierarchy_path)
        """
        # Flatten all substeps as level_1 nodes
        level_1_nodes = []
        tree_goal = ""

        for _, mem_content in memory_tree.items():
            if not isinstance(mem_content, dict):
                continue
            memory = mem_content.get("memory", {})
            tree_goal = memory.get("goal", "")
            for step_obj in memory.get("steps", []):
                for _, step_content in step_obj.items():
                    step_name = step_content.get("step", "")
                    for substep in step_content.get("substeps", []):
                        level_1_nodes.append({
                            "time_segments": f"{substep.get('start', '')} ~ {substep.get('end', '')}",
                            "summary": substep.get("substep", ""),
                            "start": substep.get("start", ""),
                            "end": substep.get("end", ""),
                            "step_name": step_name,
                        })

        if not level_1_nodes:
            return json.dumps(memory_tree, ensure_ascii=False), "unknown", {}

        node_summaries = "\n".join([
            f"ID: {i} | Time: {n['time_segments']} | Summary: {n['summary']}"
            for i, n in enumerate(level_1_nodes)
        ])

        prompt = self.prompt_template.format(cues=cues, node_summaries=node_summaries)
        decision = self.llm_fn(prompt, max_tokens=512)
        selected_id = decision.get("selected_id", 0)

        if selected_id < 0 or selected_id >= len(level_1_nodes):
            selected_id = 0

        target_branch = level_1_nodes[selected_id]
        localized_memory = json.dumps(target_branch, ensure_ascii=False, indent=2)
        target_time_segments = target_branch["time_segments"]

        hierarchy_path = {
            "goal": tree_goal,
            "step": target_branch.get("step_name", ""),
            "substep": target_branch.get("summary", ""),
            "time": target_time_segments,
            "llm_thought": decision.get("thought", ""),
        }

        return localized_memory, target_time_segments, hierarchy_path


# ============================================================
# MemoryContextFormatter
# ============================================================

class MemoryContextFormatter:
    """Memory tree/leaves를 텍스트로 포매팅하는 유틸리티."""

    def format_flat(self, memory_obj: dict, max_chars: int = 12000) -> str:
        """Flat memory format (Video-MME style).

        memory_obj: {"video_id":..., "memory": {"goal":..., "steps":[...]}}
        """
        if not memory_obj:
            return ""

        mem = memory_obj.get("memory", {})
        goal = mem.get("goal", "")
        steps = mem.get("steps", [])

        lines = []
        if goal:
            lines.append(f"[GOAL] {goal}")

        for step_block in steps:
            if not isinstance(step_block, dict):
                continue
            for sid, sval in step_block.items():
                step_name = ""
                substeps = []
                if isinstance(sval, dict):
                    step_name = sval.get("step", "")
                    substeps = sval.get("substeps", []) or []

                if step_name:
                    lines.append(f"\n[{sid}] {step_name}")

                for ss in substeps:
                    if not isinstance(ss, dict):
                        continue
                    st = ss.get("start", None)
                    ed = ss.get("end", None)
                    txt = ss.get("substep", "") or ""
                    if "No substep sentences provided" in txt:
                        continue
                    if st is not None and ed is not None:
                        lines.append(f"  - ({st:.2f}-{ed:.2f}s) {txt}")
                    else:
                        lines.append(f"  - {txt}")

        out = "\n".join(lines).strip()
        if len(out) > max_chars:
            out = out[:max_chars] + "\n...[TRUNCATED]"
        return out

    def format_bottom_up(self, raw_memory_dict: dict, target_ranges: list[tuple]) -> str:
        """Track A: Bottom-up context from matched time ranges (HD-EPIC style).

        Args:
            raw_memory_dict: {video_id: {"memory": {"goal":..., "steps":[...]}}}
            target_ranges: list of (start_sec, end_sec) tuples

        Returns:
            Formatted context string with matched substeps grouped by range.
        """
        extracted_context = ""
        found_any = False
        context_by_range = {tr: [] for tr in target_ranges}

        for _, mem_content in raw_memory_dict.items():
            if not isinstance(mem_content, dict) or "memory" not in mem_content:
                continue
            memory = mem_content["memory"]
            goal = memory.get("goal", "Unknown Goal")
            extracted_context += f"=== Global Context ===\n[Overall Goal]: {goal}\n\n"

            for step_dict in memory.get("steps", []):
                for _, step_data in step_dict.items():
                    step_summary = step_data.get("step", "")
                    for sub in step_data.get("substeps", []):
                        sub_start = time_to_secs(sub["start"])
                        sub_end = time_to_secs(sub["end"])
                        for tr in target_ranges:
                            t_start, t_end = tr
                            if max(t_start, sub_start) < min(t_end, sub_end):
                                context_by_range[tr].append(
                                    f"  [Phase: {step_summary}]\n"
                                    f"  -> {secs_to_time_str(sub_start)} ~ {secs_to_time_str(sub_end)}: {sub['substep']}"
                                )
                                found_any = True

        if not found_any:
            return ""

        for tr, evidences in context_by_range.items():
            if evidences:
                extracted_context += f"▶ Target Time {secs_to_time_str(tr[0])} ~ {secs_to_time_str(tr[1])}:\n"
                extracted_context += "\n".join(list(dict.fromkeys(evidences))) + "\n\n"

        return extracted_context

    def format_leaf_batch(self, leaf_entries: list[dict], max_chars: int = 120000) -> str:
        """Leaf batch context (LVBench agentic style).

        Args:
            leaf_entries: list of {"leaf": {...}, "leaf_id": (s,e), ...}

        Returns:
            Detailed text context from leaf entries.
        """
        entries = []
        for entry in leaf_entries:
            leaf = entry["leaf"]
            st = leaf.get("start_time", 0)
            et = leaf.get("end_time", 0)

            summary = leaf.get("summary", "")
            caption = leaf.get("caption", "")
            ke = leaf.get("key_elements", {})

            summary_line = f"[{st:.0f}s-{et:.0f}s] {summary}"
            lines = [summary_line]
            if caption:
                lines.append(f"  Caption: {caption}")

            for cat, label in [("text_ocr", "On-screen text"), ("persons", "Persons"),
                               ("actions", "Actions"), ("objects", "Objects"),
                               ("locations", "Locations"), ("attributes", "Attributes")]:
                items = ke.get(cat, [])
                if items:
                    lines.append(f"  {label}: {', '.join(str(v) for v in items)}")

            matched = entry.get("matched_cues", [])
            if matched:
                lines.append(f"  [Matched cues: {', '.join(matched)}]")

            entries.append((st, "\n".join(lines), summary_line))

        entries.sort(key=lambda x: x[0])

        full_context = "\n\n".join(e[1] for e in entries)
        if len(full_context) <= max_chars:
            return full_context

        summary_context = "\n".join(e[2] for e in entries)
        if len(summary_context) <= max_chars:
            return summary_context

        return summary_context[:max_chars] + "\n... [truncated]"

    def format_leaf_compact(self, idx: int, leaf_entry: dict) -> str:
        """Single leaf를 LLM selection용 1-line으로 포매팅."""
        leaf = leaf_entry["leaf"]
        st, et = leaf.get("start_time", 0), leaf.get("end_time", 0)
        summary = leaf.get("summary", "N/A")

        matched_cues = leaf_entry.get("matched_cues", [])
        match_str = f" | Matched: [{', '.join(matched_cues)}]" if matched_cues else ""

        ke = leaf.get("key_elements", {})
        ke_highlights = []
        for cat in ["persons", "text_ocr", "actions", "objects"]:
            items = ke.get(cat, [])
            if items:
                ke_highlights.extend(items[:3])
        ke_str = ", ".join(str(h) for h in ke_highlights[:8]) if ke_highlights else ""

        return f"ID {idx} | {st:.0f}s-{et:.0f}s | {summary[:200]} | {ke_str}{match_str}"
