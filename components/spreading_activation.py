"""
Spreading Activation — Bottom-Up 확산 활성화

인지과학적 근거: Spreading Activation Model + Hippocampal Indexing Theory
- Leaf에서 출발하여 부모 노드를 타고 올라가 Global Context 획득
- 해마의 인덱스처럼 Leaf → Level_1 → Level_2 → ... → Root 경로 추적
- "이 30초 구간은 밥을 준비하는 전체 과정 중 재료를 섞는 단계에서 일어난 일"

기존 memory_ops.py의 format_bottom_up은 특정 memory dict 구조에 의존.
이 모듈은 streaming_memory_tree 구조에서 직접 bottom-up 경로를 추출.
"""

from components.time_utils import secs_to_time_str


class SpreadingActivation:
    """Bottom-Up Spreading Activation.

    Leaf node에서 시작하여 tree를 역방향으로 타고 올라가며
    각 레벨의 컨텍스트를 수집 → LLM에게 완전한 계층 맥락 제공.
    """

    def activate(
        self,
        tree: dict,
        target_leaves: list[dict],
        include_siblings: bool = False,
    ) -> dict:
        """Bottom-up activation: leaf → parent chain → global context.

        Args:
            tree: streaming_memory_tree dict
            target_leaves: matched leaf entries from MetadataTargetedFilter
            include_siblings: 같은 Level_1 부모의 다른 leaf도 포함할지

        Returns:
            {
                "global_context": str,       # Root level summary
                "hierarchy_chains": list,     # Each: [root → ... → leaf] path
                "activated_context": str,     # LLM-ready formatted text
                "activation_depth": int,      # How deep the tree was traversed
            }
        """
        if not target_leaves or not tree:
            return {
                "global_context": "",
                "hierarchy_chains": [],
                "activated_context": "",
                "activation_depth": 0,
            }

        # Build parent index: leaf_time → Level_1 node → Level_2 node → ...
        parent_map = self._build_parent_map(tree)

        # Get the deepest level (root)
        levels = sorted(tree.keys(), key=lambda x: int(x.split("_")[1]))
        root_level = levels[-1] if levels else None
        global_context = ""
        if root_level and tree[root_level]:
            root_node = tree[root_level][0]
            global_context = root_node.get("summary", "")

        # For each target leaf, trace the activation chain
        hierarchy_chains = []
        activated_l1_ids = set()

        for entry in target_leaves:
            leaf = entry["leaf"]
            leaf_start = float(leaf.get("start_time", 0))
            leaf_end = float(leaf.get("end_time", 0))
            leaf_key = (leaf_start, leaf_end)

            chain = {
                "leaf": {
                    "time": f"{secs_to_time_str(leaf_start)} ~ {secs_to_time_str(leaf_end)}",
                    "summary": leaf.get("summary", ""),
                    "caption": leaf.get("caption", ""),
                    "key_elements": leaf.get("key_elements", {}),
                },
                "parents": [],
            }

            # Trace up the tree
            if leaf_key in parent_map:
                for level_name, parent_node in parent_map[leaf_key]:
                    chain["parents"].append({
                        "level": level_name,
                        "summary": parent_node.get("summary", ""),
                        "time_segments": parent_node.get("time_segments", []),
                    })
                    # Track L1 for sibling inclusion
                    if level_name == "Level_1":
                        l1_id = id(parent_node)
                        activated_l1_ids.add(l1_id)

            hierarchy_chains.append(chain)

        # Include sibling leaves if requested
        sibling_context = ""
        if include_siblings and activated_l1_ids:
            siblings = self._get_siblings(tree, activated_l1_ids, target_leaves)
            if siblings:
                sibling_context = self._format_siblings(siblings)

        # Format activated context
        activated_context = self._format_activation(
            global_context, hierarchy_chains, sibling_context
        )

        return {
            "global_context": global_context,
            "hierarchy_chains": hierarchy_chains,
            "activated_context": activated_context,
            "activation_depth": len(levels),
        }

    def _build_parent_map(self, tree: dict) -> dict:
        """leaf (start, end) → [(level_name, parent_node), ...] chain."""
        parent_map = {}  # (start, end) → list of (level_name, node)

        levels = sorted(tree.keys(), key=lambda x: int(x.split("_")[1]))

        # Level_1 → leaf mapping
        if "Level_1" in tree:
            for l1_node in tree["Level_1"]:
                for child in l1_node.get("children", []):
                    if "start_time" in child:
                        key = (float(child["start_time"]), float(child["end_time"]))
                        if key not in parent_map:
                            parent_map[key] = []
                        parent_map[key].append(("Level_1", l1_node))

        # Higher levels: find which Level_N contains which Level_(N-1)
        for i in range(1, len(levels)):
            higher_level = levels[i]
            lower_level = levels[i - 1]

            for higher_node in tree.get(higher_level, []):
                higher_segs = higher_node.get("time_segments", [])
                if not higher_segs:
                    continue

                # Check overlap
                for lower_node in tree.get(lower_level, []):
                    lower_segs = lower_node.get("time_segments", [])
                    if self._segments_overlap(higher_segs, lower_segs):
                        # This higher node covers this lower node
                        # Find all leaves under this lower node
                        for leaf_key in parent_map:
                            if any(
                                ("Level_" + str(i), lower_node) == p
                                for p in parent_map.get(leaf_key, [])
                            ) or self._leaf_in_segments(leaf_key, lower_segs):
                                parent_map.setdefault(leaf_key, []).append(
                                    (higher_level, higher_node)
                                )

        # Deduplicate parent chains
        for key in parent_map:
            seen = set()
            unique = []
            for level_name, node in parent_map[key]:
                node_id = (level_name, id(node))
                if node_id not in seen:
                    seen.add(node_id)
                    unique.append((level_name, node))
            parent_map[key] = unique

        return parent_map

    @staticmethod
    def _segments_overlap(segs_a, segs_b) -> bool:
        """Check if any segment pair overlaps."""
        for sa in segs_a:
            sa_start = float(sa[0]) if isinstance(sa, (list, tuple)) else float(sa)
            sa_end = float(sa[1]) if isinstance(sa, (list, tuple)) else float(sa)
            for sb in segs_b:
                sb_start = float(sb[0]) if isinstance(sb, (list, tuple)) else float(sb)
                sb_end = float(sb[1]) if isinstance(sb, (list, tuple)) else float(sb)
                if max(sa_start, sb_start) < min(sa_end, sb_end):
                    return True
        return False

    @staticmethod
    def _leaf_in_segments(leaf_key, segments) -> bool:
        """Check if leaf overlaps with any segment."""
        ls, le = leaf_key
        for seg in segments:
            ss = float(seg[0]) if isinstance(seg, (list, tuple)) else float(seg)
            se = float(seg[1]) if isinstance(seg, (list, tuple)) else float(seg)
            if max(ls, ss) < min(le, se):
                return True
        return False

    def _get_siblings(self, tree, activated_l1_ids, target_leaves) -> list[dict]:
        """Get sibling leaves from the same Level_1 parents."""
        target_keys = {e["leaf_id"] for e in target_leaves}
        siblings = []

        if "Level_1" not in tree:
            return siblings

        for l1_node in tree["Level_1"]:
            if id(l1_node) in activated_l1_ids:
                for child in l1_node.get("children", []):
                    if "start_time" in child:
                        key = (float(child["start_time"]), float(child["end_time"]))
                        if key not in target_keys:
                            siblings.append({
                                "leaf": child,
                                "leaf_id": key,
                                "parent_summary": l1_node.get("summary", ""),
                            })
        return siblings

    def _format_siblings(self, siblings: list[dict]) -> str:
        """Format sibling leaves as context."""
        lines = ["\n[Temporal Neighbors (same parent segment)]"]
        for s in sorted(siblings, key=lambda x: x["leaf_id"][0]):
            leaf = s["leaf"]
            st, et = leaf.get("start_time", 0), leaf.get("end_time", 0)
            lines.append(f"  [{st:.0f}s-{et:.0f}s] {leaf.get('summary', '')}")
        return "\n".join(lines)

    def _format_activation(self, global_context: str,
                           chains: list[dict], sibling_context: str) -> str:
        """Format activated context for LLM consumption.

        Goal → Step → Substep 계층 구조로 정리.
        """
        lines = []

        # Global context (Root)
        if global_context:
            lines.append(f"=== Global Context (Root) ===")
            lines.append(f"[Overall Goal/Theme]: {global_context}")
            lines.append("")

        # Hierarchy chains
        for i, chain in enumerate(chains):
            leaf = chain["leaf"]
            parents = chain["parents"]

            lines.append(f"▶ Target Segment {i+1}: {leaf['time']}")

            # Parents (top-down order: highest level first)
            if parents:
                sorted_parents = sorted(
                    parents,
                    key=lambda p: int(p["level"].split("_")[1]),
                    reverse=True,
                )
                indent = "  "
                for p in sorted_parents:
                    lines.append(f"{indent}└─ [{p['level']}] {p['summary']}")
                    indent += "  "

            # Leaf detail
            lines.append(f"    └─ [Leaf] {leaf['summary']}")
            if leaf.get("caption"):
                lines.append(f"       Caption: {leaf['caption']}")

            ke = leaf.get("key_elements", {})
            for cat, label in [("text_ocr", "On-screen text"), ("actions", "Actions"),
                               ("objects", "Objects"), ("persons", "Persons"),
                               ("attributes", "Attributes"), ("locations", "Locations")]:
                items = ke.get(cat, [])
                if items:
                    lines.append(f"       {label}: {', '.join(str(v) for v in items)}")

            lines.append("")

        # Sibling context
        if sibling_context:
            lines.append(sibling_context)

        return "\n".join(lines)
