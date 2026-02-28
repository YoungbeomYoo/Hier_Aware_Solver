from __future__ import annotations

"""
Filtered Tree Builder — key_elements 기준 top-down 활성화 트리

핵심 아이디어:
- 최상위 레벨부터 cue 매칭 → 활성 노드 결정
- 활성 노드의 하위 children을 후보로 연결
- 결과: root→leaf 연결 path들 + time index for direct jump

기존 HierarchicalScorer와 다른 점:
- HierarchicalScorer: 각 레벨 독립 점수 산정, flat 우선순위 반환
- FilteredTreeBuilder: 부모→자식 연결 유지, path 단위로 탐색, ancestor 추적
"""


class FilteredTreeBuilder:
    """Top-down key_elements matching → connected filtered subtree.

    Args:
        match_threshold: 활성화에 필요한 최소 cue 매칭 수
    """

    def __init__(self, match_threshold: int = 1, use_key_elements: bool = True):
        self.match_threshold = match_threshold
        self.use_key_elements = use_key_elements

    def build(
        self,
        tree: dict,
        cues: list[str],
        target_fields: list[str] | None = None,
        semantic_scores: dict | None = None,
    ) -> dict:
        """Build filtered tree with activation paths.

        Returns:
            {
                "activated_nodes": {level_name: [node_info, ...]},
                "paths": [{"path": [...], "aggregate_score": int, "leaf": ...}, ...],
                "time_index": {(start, end): leaf_entry},
                "priority_leaves": [leaf_entry, ...],
                "all_leaves": [leaf_entry, ...],
            }
        """
        empty = {
            "activated_nodes": {}, "paths": [], "time_index": {},
            "priority_leaves": [], "all_leaves": [],
        }
        if not tree or not cues:
            return empty

        cues_lower = [c.lower().strip() for c in cues if c.strip()]
        if not cues_lower:
            return empty

        fields = target_fields or [
            "actions", "objects", "persons", "attributes",
            "locations", "text_ocr", "summary",
        ]

        # Discover all levels, sorted highest → lowest
        levels = sorted(
            [k for k in tree.keys()],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        )

        # Score all non-leaf nodes
        activated = {}
        for level_name in levels:
            level_nodes = []
            for idx, node in enumerate(tree[level_name]):
                score, matched = self._match_node(node, cues_lower, fields)
                level_nodes.append({
                    "node": node,
                    "idx": idx,
                    "level": level_name,
                    "score": score,
                    "matched_cues": matched,
                    "on": score >= self.match_threshold,
                    "time_segments": node.get("time_segments", []),
                })
            activated[level_name] = level_nodes

        # Score leaves (Level_1 children)
        leaf_entries = []
        if "Level_1" in tree:
            for l1_idx, l1_node in enumerate(tree["Level_1"]):
                for child_idx, child in enumerate(l1_node.get("children", [])):
                    if "start_time" not in child:
                        continue
                    score, matched = self._match_node(child, cues_lower, fields)
                    leaf_entries.append({
                        "node": child,
                        "idx": child_idx,
                        "level": "Level_0",
                        "score": score,
                        "matched_cues": matched,
                        "on": score >= self.match_threshold,
                        "parent_l1_idx": l1_idx,
                        "parent_l1_summary": l1_node.get("summary", ""),
                        "start_time": float(child["start_time"]),
                        "end_time": float(child["end_time"]),
                    })

        activated["Level_0"] = leaf_entries

        # Build root→leaf paths
        paths = self._build_paths(tree, levels, activated, leaf_entries)

        # Time index
        time_index = {}
        for entry in leaf_entries:
            key = (entry["start_time"], entry["end_time"])
            time_index[key] = entry

        # Priority leaves (activated, sorted by score)
        priority = sorted(
            [e for e in leaf_entries if e["on"]],
            key=lambda x: -x["score"],
        )

        # Semantic score override: Level_1 semantic scores → leaf scoring
        if semantic_scores and "selected_indices" in semantic_scores:
            selected_l1 = set(semantic_scores["selected_indices"])
            # Build L1 idx → semantic score map
            l1_score_map = {}
            for s in semantic_scores.get("scores", []):
                l1_score_map[s["node_idx"]] = s["score"]

            # Re-score leaves based on parent L1's semantic score
            for entry in leaf_entries:
                l1_idx = entry.get("parent_l1_idx")
                if l1_idx is not None and l1_idx in l1_score_map:
                    entry["semantic_score"] = l1_score_map[l1_idx]
                    entry["on"] = l1_idx in selected_l1
                else:
                    entry["semantic_score"] = 0.0

            # Re-build priority with semantic scores
            priority = sorted(
                [e for e in leaf_entries if e["on"]],
                key=lambda x: (-x.get("semantic_score", 0), -x["score"]),
            )

        return {
            "activated_nodes": activated,
            "paths": paths,
            "time_index": time_index,
            "priority_leaves": priority,
            "all_leaves": leaf_entries,
            "semantic_scores": semantic_scores,
        }

    def find_by_time(
        self,
        filtered: dict,
        target_time: float,
        window: float = 30.0,
    ) -> list[dict]:
        """Find leaves covering or near the target time.

        Returns:
            [{"entry": leaf_entry, "distance": float, "type": "direct"|"nearby"}, ...]
        """
        results = []
        for entry in filtered.get("all_leaves", []):
            st, et = entry["start_time"], entry["end_time"]
            if st <= target_time <= et:
                results.append({"entry": entry, "distance": 0, "type": "direct"})
            elif abs(st - target_time) <= window or abs(et - target_time) <= window:
                dist = min(abs(st - target_time), abs(et - target_time))
                results.append({"entry": entry, "distance": dist, "type": "nearby"})

        results.sort(key=lambda x: x["distance"])
        return results

    def find_by_time_range(
        self,
        filtered: dict,
        start: float,
        end: float,
        expand_window: float = 15.0,
    ) -> list[dict]:
        """Find leaves overlapping with a time range + expansion window."""
        results = []
        search_start = start - expand_window
        search_end = end + expand_window

        for entry in filtered.get("all_leaves", []):
            st, et = entry["start_time"], entry["end_time"]
            if st <= search_end and et >= search_start:
                # Overlap amount → closer is better
                overlap = min(et, search_end) - max(st, search_start)
                results.append({"entry": entry, "overlap": max(0, overlap)})

        results.sort(key=lambda x: -x["overlap"])
        return [r for r in results]

    def get_ancestors(
        self,
        tree: dict,
        leaf_entry: dict,
    ) -> list[dict]:
        """Get all ancestor nodes for a leaf, Level_1 up to highest.

        Returns list ordered [Level_1, Level_2, ..., Level_N].
        """
        ancestors = []
        levels = sorted(
            [k for k in tree.keys()],
            key=lambda x: int(x.split("_")[1]),
        )

        leaf_start = leaf_entry.get("start_time", 0)
        leaf_end = leaf_entry.get("end_time", 0)

        for level_name in levels:
            best = None
            best_overlap = -1
            for node in tree[level_name]:
                segs = node.get("time_segments", [])
                overlap = self._overlap_amount(segs, leaf_start, leaf_end)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = node
            if best and best_overlap > 0:
                ancestors.append({
                    "level": level_name,
                    "node": best,
                    "summary": best.get("summary", ""),
                    "key_elements": best.get("key_elements", {}),
                })

        return ancestors

    def get_unseen_leaves(
        self,
        filtered: dict,
        seen_ids: set,
        budget: int = 10,
    ) -> list[dict]:
        """Get next best unseen leaves from priority list."""
        result = []
        for entry in filtered.get("priority_leaves", []):
            key = (entry["start_time"], entry["end_time"])
            if key not in seen_ids:
                result.append(entry)
                if len(result) >= budget:
                    break
        return result

    def get_siblings(
        self,
        tree: dict,
        leaf_entry: dict,
        seen_ids: set,
    ) -> list[dict]:
        """Get sibling leaves (same Level_1 parent)."""
        if "Level_1" not in tree:
            return []

        target_start = leaf_entry.get("start_time", 0)
        target_end = leaf_entry.get("end_time", 0)

        for l1_node in tree["Level_1"]:
            has_target = False
            for child in l1_node.get("children", []):
                if "start_time" in child:
                    if (float(child["start_time"]) == target_start
                            and float(child["end_time"]) == target_end):
                        has_target = True
                        break

            if has_target:
                siblings = []
                for child in l1_node.get("children", []):
                    if "start_time" not in child:
                        continue
                    key = (float(child["start_time"]), float(child["end_time"]))
                    if key not in seen_ids:
                        siblings.append({
                            "node": child,
                            "start_time": key[0],
                            "end_time": key[1],
                            "level": "Level_0",
                            "score": 0,
                            "matched_cues": [],
                            "on": True,
                            "parent_l1_summary": l1_node.get("summary", ""),
                        })
                siblings.sort(key=lambda x: x["start_time"])
                return siblings

        return []

    def get_unexplored_regions(
        self,
        tree: dict,
        seen_ids: set,
        level: str = "Level_1",
    ) -> list[dict]:
        """Get regions at the given level that still have unseen leaves underneath.

        Args:
            level: "Level_1", "Level_2", "Level_3", ...
                   Level_1 → direct children are leaves
                   Level_2+ → leaves found by time overlap

        Returns list of region summaries sorted by unseen count (desc).
        """
        if level not in tree:
            return []

        # Collect all unseen leaves for overlap checking
        all_unseen_leaves = []
        if "Level_1" in tree:
            for l1_node in tree["Level_1"]:
                for child in l1_node.get("children", []):
                    if "start_time" not in child:
                        continue
                    key = (float(child["start_time"]), float(child["end_time"]))
                    if key not in seen_ids:
                        all_unseen_leaves.append(child)

        regions = []
        for idx, node in enumerate(tree[level]):
            # Count unseen leaves under this node
            node_start, node_end = self._node_time_range(node, tree, level)

            if level == "Level_1":
                # Direct children
                children = [c for c in node.get("children", [])
                            if "start_time" in c]
                unseen = sum(
                    1 for c in children
                    if (float(c["start_time"]), float(c["end_time"])) not in seen_ids
                )
                total = len(children)
            else:
                # Higher levels: count leaves by time overlap
                unseen = 0
                total = 0
                for leaf in all_unseen_leaves:
                    ls, le = float(leaf["start_time"]), float(leaf["end_time"])
                    if ls < node_end and le > node_start:
                        unseen += 1
                # Total = all leaves (seen + unseen) in range
                if "Level_1" in tree:
                    for l1_node in tree["Level_1"]:
                        for child in l1_node.get("children", []):
                            if "start_time" not in child:
                                continue
                            cs, ce = float(child["start_time"]), float(child["end_time"])
                            if cs < node_end and ce > node_start:
                                total += 1

            if unseen == 0:
                continue

            # Key elements brief
            ke = node.get("key_elements", {})
            ke_brief = {}
            for field in ["actions", "objects", "persons"]:
                vals = ke.get(field, [])
                if vals:
                    ke_brief[field] = [str(v) for v in vals[:5]]

            regions.append({
                "level": level,
                "idx": idx,
                "summary": node.get("summary", ""),
                "key_elements_brief": ke_brief,
                "unseen_count": unseen,
                "total_count": total,
                "start_time": node_start,
                "end_time": node_end,
            })

        regions.sort(key=lambda x: -x["unseen_count"])
        return regions

    def get_leaves_under_region(
        self,
        tree: dict,
        region: dict,
        seen_ids: set,
        budget: int = 10,
    ) -> list[dict]:
        """Get unseen leaves under a region (any level).

        For Level_1: direct children.
        For Level_2+: all leaves whose time overlaps with the region's time range.
        """
        level = region.get("level", "Level_1")
        idx = region["idx"]
        start_t = region["start_time"]
        end_t = region["end_time"]

        if level == "Level_1" and "Level_1" in tree and idx < len(tree["Level_1"]):
            l1_node = tree["Level_1"][idx]
            leaves = []
            for child in l1_node.get("children", []):
                if "start_time" not in child:
                    continue
                key = (float(child["start_time"]), float(child["end_time"]))
                if key not in seen_ids:
                    leaves.append({
                        "node": child,
                        "start_time": key[0],
                        "end_time": key[1],
                        "level": "Level_0",
                        "score": 0,
                        "matched_cues": [],
                        "on": False,
                        "parent_l1_idx": idx,
                        "parent_l1_summary": l1_node.get("summary", ""),
                    })
            leaves.sort(key=lambda x: x["start_time"])
            return leaves[:budget]

        # Higher levels: collect all leaves in time range
        leaves = []
        if "Level_1" not in tree:
            return []
        for l1_idx, l1_node in enumerate(tree["Level_1"]):
            for child in l1_node.get("children", []):
                if "start_time" not in child:
                    continue
                cs, ce = float(child["start_time"]), float(child["end_time"])
                key = (cs, ce)
                if key in seen_ids:
                    continue
                if cs < end_t and ce > start_t:
                    leaves.append({
                        "node": child,
                        "start_time": cs,
                        "end_time": ce,
                        "level": "Level_0",
                        "score": 0,
                        "matched_cues": [],
                        "on": False,
                        "parent_l1_idx": l1_idx,
                        "parent_l1_summary": l1_node.get("summary", ""),
                    })
        leaves.sort(key=lambda x: x["start_time"])
        return leaves[:budget]

    def _node_time_range(self, node, tree, level):
        """Get (start, end) time for a node at any level."""
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

        # Fallback for Level_1: derive from children
        if level == "Level_1":
            children = [c for c in node.get("children", [])
                        if "start_time" in c]
            if children:
                return (min(float(c["start_time"]) for c in children),
                        max(float(c["end_time"]) for c in children))
        return 0.0, 0.0

    # ========== Internal ==========

    def _build_paths(self, tree, levels, activated, leaf_entries):
        """Build connected paths from top to each activated leaf."""
        paths = []

        for leaf in leaf_entries:
            if not leaf["on"]:
                continue

            path_nodes = [leaf]
            path_score = leaf["score"]

            # Walk up from leaf's time to each higher level
            for level_name in sorted(levels, key=lambda x: int(x.split("_")[1])):
                best_parent = None
                best_score = -1

                for node_info in activated.get(level_name, []):
                    overlap = self._overlap_amount(
                        node_info["time_segments"],
                        leaf["start_time"], leaf["end_time"],
                    )
                    if overlap > 0 and node_info["score"] > best_score:
                        best_parent = node_info
                        best_score = node_info["score"]

                if best_parent:
                    path_nodes.append(best_parent)
                    path_score += best_parent["score"]

            paths.append({
                "path": list(reversed(path_nodes)),  # top-down
                "aggregate_score": path_score,
                "leaf": leaf,
            })

        paths.sort(key=lambda x: -x["aggregate_score"])
        return paths

    def _match_node(self, node, cues_lower, fields):
        """Match cues against node's key_elements and summary."""
        summary = node.get("summary", "").lower()
        caption = node.get("caption", "").lower()
        combined_text = summary + " " + caption

        ke = node.get("key_elements", {}) if self.use_key_elements else {}

        matched = []
        for cue in cues_lower:
            found = False
            for field in fields:
                if field == "summary":
                    if cue in combined_text:
                        found = True
                        break
                else:
                    if self.use_key_elements:
                        for val in ke.get(field, []):
                            if cue in str(val).lower():
                                found = True
                                break
                        if found:
                            break

            if not found and cue in combined_text:
                found = True

            if found:
                matched.append(cue)

        return len(matched), matched

    def _overlap_amount(self, segments, start, end):
        """Calculate overlap between segments and a time range."""
        total = 0
        for seg in segments:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                s, e = float(seg[0]), float(seg[1])
            else:
                s = e = float(seg)
            overlap = min(e, end) - max(s, start)
            if overlap > 0:
                total += overlap
        return total
