from __future__ import annotations

"""
Tree Planner — 상위 레벨 요약으로 탐색 영역을 미리 선정

Flow:
1. 전체 tree의 상위 레벨 (Level_3 → Level_2 → Level_1)의 summary + key_elements만 추출
2. LLM에게 question과 함께 보여주고 어떤 Level_1 영역을 봐야 하는지 물어봄
3. 선택된 Level_1만 full tree (caption 포함)로 살리고 나머지 pruning

장점:
- 전체 caption을 넣으면 ~70K 토큰이지만, overview만 넣으면 ~8K 토큰
- LLM이 전체 구조를 보고 전략적으로 탐색 대상을 선정
- Pruned tree로 기존 tree_search pipeline 그대로 실행 가능
"""

import re
from components.token_utils import TokenBudget


class TreePlanner:
    """Initial planning: show tree overview to LLM, select focus regions.

    Args:
        llm_fn: text-only LLM callable(prompt, max_tokens) -> dict
        token_budget: TokenBudget instance
        max_overview_tokens: overview 텍스트 최대 토큰
        max_regions: 선택할 최대 Level_1 영역 수
    """

    PLAN_PROMPT = """You are planning a search through a video's hierarchical memory to answer a question. Below is a compact overview of the entire video's content organized by time segments.

### Question
{question}

### Options
{options_text}

### Video Memory Overview (high-level → detailed)
{tree_overview}

### Task
Select the {max_regions} most promising Level_1 region indices that are most likely to contain information needed to answer the question.

Consider:
- Temporal relevance (when events might happen)
- Content relevance (actions, objects, persons mentioned)
- Question type (does it need specific visual details? a sequence of events? counting?)

Output ONLY valid JSON:
{{
    "selected_regions": [0, 3, 7, 12, 15],
    "reasoning": "Brief explanation of the selection strategy"
}}"""

    def __init__(
        self,
        llm_fn=None,
        token_budget: TokenBudget | None = None,
        max_overview_tokens: int = 15000,
        max_regions: int = 10,
    ):
        self.llm_fn = llm_fn
        self.tb = token_budget or TokenBudget(None)
        self.max_overview_tokens = max_overview_tokens
        self.max_regions = max_regions

    def plan(
        self,
        tree: dict,
        question: str,
        options: list[str],
        max_regions: int | None = None,
    ) -> tuple[dict, dict]:
        """Plan which regions to focus on.

        Args:
            tree: full streaming_memory_tree
            question: question text
            options: answer options
            max_regions: override max regions to select

        Returns:
            (pruned_tree, plan_info)
            - pruned_tree: tree with only selected Level_1 nodes
            - plan_info: {"selected_indices": [...], "reasoning": str, ...}
        """
        n_regions = max_regions or self.max_regions

        if not self.llm_fn or not tree:
            return tree, {"selected_indices": [], "reasoning": "no LLM or empty tree"}

        # Build compact overview
        overview, l1_count = self._build_overview(tree)

        # If tree is small enough, no planning needed
        if l1_count <= n_regions:
            return tree, {
                "selected_indices": list(range(l1_count)),
                "reasoning": f"tree small enough ({l1_count} regions), using all",
                "skipped_planning": True,
            }

        # Ask LLM
        selected, reasoning = self._ask_llm(
            overview, question, options, n_regions, l1_count,
        )

        if not selected:
            # Fallback: select first N regions
            selected = list(range(min(n_regions, l1_count)))
            reasoning = "LLM selection failed, using first N regions"

        # Prune tree
        pruned = self._prune_tree(tree, selected)

        plan_info = {
            "selected_indices": selected,
            "reasoning": reasoning,
            "total_l1_regions": l1_count,
            "selected_count": len(selected),
        }

        return pruned, plan_info

    def _build_overview(self, tree: dict) -> tuple[str, int]:
        """Build compact overview: upper levels + Level_1 summaries (no leaf captions).

        Returns (overview_text, l1_node_count)
        """
        parts = []
        l1_count = 0

        # Sort levels high → low
        levels = sorted(
            [k for k in tree.keys()],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        )

        for level_name in levels:
            nodes = tree[level_name]
            parts.append(f"\n=== {level_name} ({len(nodes)} nodes) ===")

            for idx, node in enumerate(nodes):
                summary = node.get("summary", "")
                ke = node.get("key_elements", {})

                # Time range
                start_t, end_t = self._node_time_range(node)

                # Key elements brief
                ke_parts = []
                for field in ["actions", "objects", "persons", "locations"]:
                    vals = ke.get(field, [])
                    if vals:
                        ke_parts.append(
                            f"{field}: {', '.join(str(v) for v in vals[:5])}"
                        )
                ke_str = " | ".join(ke_parts)

                if level_name == "Level_1":
                    # Count leaves, show summary only (no captions)
                    children = [c for c in node.get("children", [])
                                if "start_time" in c]
                    n_leaves = len(children)
                    leaf_time = ""
                    if children:
                        leaf_start = min(float(c["start_time"]) for c in children)
                        leaf_end = max(float(c["end_time"]) for c in children)
                        leaf_time = f"[{leaf_start:.0f}s-{leaf_end:.0f}s]"
                    else:
                        leaf_time = f"[{start_t:.0f}s-{end_t:.0f}s]"

                    line = f"  [{idx}] {leaf_time} {summary} ({n_leaves} segments)"
                    if ke_str:
                        line += f"\n       {ke_str}"
                    l1_count += 1
                else:
                    line = (
                        f"  [{level_name}:{idx}] "
                        f"[{start_t:.0f}s-{end_t:.0f}s] {summary}"
                    )
                    if ke_str:
                        line += f"\n       {ke_str}"

                parts.append(line)

        overview = "\n".join(parts)

        # Truncate if needed
        overview = self.tb.truncate(overview, self.max_overview_tokens)

        return overview, l1_count

    def _ask_llm(
        self,
        overview: str,
        question: str,
        options: list[str],
        max_regions: int,
        l1_count: int,
    ) -> tuple[list[int], str]:
        """Ask LLM to select regions."""
        opt_text = "\n".join(
            f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        )

        prompt = self.PLAN_PROMPT.format(
            question=question,
            options_text=opt_text,
            tree_overview=overview,
            max_regions=max_regions,
        )

        try:
            result = self.llm_fn(prompt, max_tokens=300)
        except Exception as e:
            print(f"    [TreePlanner] LLM error: {e}")
            return [], ""

        if not isinstance(result, dict):
            return [], str(result)[:200]

        # Parse selected_regions
        selected = result.get("selected_regions", [])
        reasoning = result.get("reasoning", "")

        # Validate indices
        valid = []
        for idx in selected:
            if isinstance(idx, (int, float)) and 0 <= int(idx) < l1_count:
                valid.append(int(idx))

        # Fallback: try to parse from reasoning
        if not valid and reasoning:
            nums = re.findall(r'\d+', str(selected) + " " + reasoning)
            for n in nums:
                idx = int(n)
                if 0 <= idx < l1_count and idx not in valid:
                    valid.append(idx)
                    if len(valid) >= max_regions:
                        break

        return valid[:max_regions], reasoning

    def _prune_tree(self, tree: dict, selected_l1_indices: list[int]) -> dict:
        """Keep only selected Level_1 nodes with full leaf details.

        Upper levels are kept as-is for hierarchy context.
        """
        pruned = {}
        selected_set = set(selected_l1_indices)

        for key in tree:
            if key == "Level_1":
                pruned["Level_1"] = [
                    tree["Level_1"][i]
                    for i in sorted(selected_set)
                    if i < len(tree["Level_1"])
                ]
            else:
                # Keep upper levels for context
                pruned[key] = tree[key]

        return pruned

    @staticmethod
    def _node_time_range(node):
        """Get (start, end) time from a node."""
        segs = node.get("time_segments", [])
        if segs:
            flat = []
            for s in segs:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    flat.extend([float(s[0]), float(s[1])])
                else:
                    flat.append(float(s))
            if flat:
                return min(flat), max(flat)

        # Level_1 fallback
        children = [c for c in node.get("children", [])
                    if "start_time" in c]
        if children:
            return (min(float(c["start_time"]) for c in children),
                    max(float(c["end_time"]) for c in children))
        return 0.0, 0.0
