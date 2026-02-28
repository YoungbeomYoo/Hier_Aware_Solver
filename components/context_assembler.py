from __future__ import annotations

"""
Budget-Aware Context Assembler — token budget 내에서 최적 context 구성

Packing 전략 (우선순위 높은 순):
1. 상위 레벨 hierarchy summaries (top-down, compact)
2. 타겟 leaf captions + key_elements (rich, detailed)
3. 인접 leaf summaries (neighboring context)

모든 정보가 max_budget (tokens) 내에 들어가도록 조절.
"""

from components.token_utils import TokenBudget


class BudgetContextAssembler:
    """Budget-aware context assembly from tree path.

    Args:
        max_budget: 기본 최대 토큰 수
        token_budget: TokenBudget instance (None → char fallback)
    """

    def __init__(self, max_budget: int = 4000, token_budget: TokenBudget | None = None,
                 use_captions: bool = True, use_key_elements: bool = True):
        self.max_budget = max_budget
        self.tb = token_budget or TokenBudget(None)
        self.use_captions = use_captions
        self.use_key_elements = use_key_elements

    def assemble(
        self,
        targets: list[dict],
        ancestors_map: dict,
        max_budget: int | None = None,
    ) -> dict:
        """Assemble context within budget.

        Args:
            targets: leaf entries to include
            ancestors_map: {(start, end): [ancestor_info, ...]} from TreeFilter
            max_budget: override budget (tokens)

        Returns:
            {
                "context": str,
                "hierarchy_path": str,
                "included_nodes": list,
                "budget_used": int,  # tokens
            }
        """
        budget = max_budget or self.max_budget
        parts = []
        included = []

        # =============================================
        # Part 1: Hierarchy summaries (top-down)
        # =============================================
        hierarchy_lines = []
        seen_summaries = set()

        for target in targets:
            key = (
                target.get("start_time",
                           float(target.get("node", {}).get("start_time", 0))),
                target.get("end_time",
                           float(target.get("node", {}).get("end_time", 0))),
            )
            ancestors = ancestors_map.get(key, [])

            # Top-down: highest level first
            for anc in reversed(ancestors):
                summary = anc.get("summary", "")
                level = anc.get("level", "?")
                if summary and summary not in seen_summaries:
                    seen_summaries.add(summary)
                    line = f"[{level}] {summary}"
                    if self.use_key_elements:
                        ke = anc.get("key_elements", {})
                        ke_brief = ""
                        for field in ["actions", "objects", "persons"]:
                            vals = ke.get(field, [])
                            if vals:
                                ke_brief += f" | {field}: {', '.join(str(v) for v in vals[:5])}"
                        if ke_brief:
                            line += ke_brief
                    hierarchy_lines.append(line)
                    included.append({"level": level, "type": "summary"})

        if hierarchy_lines:
            parts.append(
                "=== Hierarchical Context (top-down) ===\n"
                + "\n".join(hierarchy_lines)
            )

        # =============================================
        # Part 2: Target leaf details (rich)
        # =============================================
        leaf_lines = []
        for target in targets:
            node = target.get("node", target)
            st = float(node.get("start_time", 0))
            et = float(node.get("end_time", 0))
            caption = node.get("caption", "")
            summary = node.get("summary", "")

            text = f"--- [{st:.1f}s - {et:.1f}s] ---"
            if self.use_captions and caption:
                text += f"\nCaption: {caption}"
            if summary and summary != caption:
                text += f"\nSummary: {summary}"

            # key_elements
            if self.use_key_elements:
                ke = node.get("key_elements", {})
                for field in ["actions", "objects", "persons", "attributes",
                              "locations", "text_ocr"]:
                    vals = ke.get(field, [])
                    if vals:
                        text += f"\n  {field}: {', '.join(str(v) for v in vals)}"

            # Parent summary if available
            parent_summary = target.get("parent_l1_summary", "")
            if parent_summary:
                text += f"\n  [Parent context] {parent_summary}"

            leaf_lines.append(text)
            included.append({"time": f"{st:.1f}-{et:.1f}", "type": "leaf"})

        if leaf_lines:
            parts.append(
                "=== Detailed Segments ===\n"
                + "\n\n".join(leaf_lines)
            )

        # Combine and trim by tokens
        full_context = "\n\n".join(parts)
        full_context = self.tb.truncate(full_context, budget)

        # Hierarchy path string for logging
        path_parts = []
        for n in included:
            if n["type"] == "summary":
                path_parts.append(n["level"])
        path_str = " → ".join(dict.fromkeys(path_parts))  # unique, ordered
        if not path_str:
            path_str = "direct"

        return {
            "context": full_context,
            "hierarchy_path": path_str,
            "included_nodes": included,
            "budget_used": self.tb.count(full_context),
        }

    def assemble_with_neighbors(
        self,
        targets: list[dict],
        ancestors_map: dict,
        tree: dict | None = None,
        max_budget: int | None = None,
    ) -> dict:
        """Like assemble but also includes neighboring leaves within budget."""
        result = self.assemble(targets, ancestors_map, max_budget)

        budget = max_budget or self.max_budget
        remaining = budget - result["budget_used"]

        if remaining <= 100 or not tree or "Level_1" not in tree:
            return result

        # Find neighbors: same Level_1 parent, not already in targets
        target_times = set()
        for t in targets:
            st = t.get("start_time",
                        float(t.get("node", {}).get("start_time", 0)))
            et = t.get("end_time",
                        float(t.get("node", {}).get("end_time", 0)))
            target_times.add((st, et))

        neighbors = []
        for l1_node in tree["Level_1"]:
            has_target = False
            for child in l1_node.get("children", []):
                if "start_time" in child:
                    key = (float(child["start_time"]), float(child["end_time"]))
                    if key in target_times:
                        has_target = True
                        break

            if has_target:
                for child in l1_node.get("children", []):
                    if "start_time" in child:
                        key = (float(child["start_time"]),
                               float(child["end_time"]))
                        if key not in target_times:
                            neighbors.append(child)

        if not neighbors:
            return result

        # Sort neighbors by time
        neighbors.sort(key=lambda x: float(x.get("start_time", 0)))

        # Pack neighbors within remaining token budget
        neighbor_lines = []
        used_tokens = 0
        for n in neighbors:
            st = float(n.get("start_time", 0))
            et = float(n.get("end_time", 0))
            summary = n.get("summary", n.get("caption", ""))
            line = f"  [{st:.1f}s-{et:.1f}s] {summary}"
            line_tokens = self.tb.count(line)
            if used_tokens + line_tokens > remaining - 10:
                break
            neighbor_lines.append(line)
            used_tokens += line_tokens

        if neighbor_lines:
            neighbor_text = (
                "\n\n=== Neighboring Segments ===\n"
                + "\n".join(neighbor_lines)
            )
            result["context"] += neighbor_text
            result["budget_used"] = self.tb.count(result["context"])

        return result

    def format_for_hop(
        self,
        new_context: str,
        history_compact: dict | None,
        max_budget: int | None = None,
    ) -> str:
        """Merge new context with compacted history for next hop.

        Structure:
        1. Search state (compact)
        2. New context (current hop, rich)
        3. Previous observations (accumulated)
        """
        budget = max_budget or self.max_budget

        if not history_compact:
            return self.tb.truncate(new_context, budget)

        search_state = history_compact.get("search_state", "")
        prev_obs = history_compact.get("observations_rich", "")

        # Allocate budget: 15% state, 60% new context, 25% prev observations
        state_budget = int(budget * 0.15)
        new_budget = int(budget * 0.60)
        prev_budget = int(budget * 0.25)

        parts = []
        if search_state:
            parts.append(self.tb.truncate(search_state, state_budget))
        parts.append(self.tb.truncate(new_context, new_budget))
        if prev_obs:
            parts.append(
                "=== Previous Observations ===\n"
                + self.tb.truncate(prev_obs, prev_budget)
            )

        return "\n\n".join(parts)
