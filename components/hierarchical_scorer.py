"""
Hierarchical Scorer — 전체 트리 레벨별 노드 활성화 점수 산정

핵심 아이디어:
- streaming_memory_tree의 모든 레벨(Level_1 ~ Level_N + leaf)을 스캔
- 각 노드의 key_elements를 query cues와 비교하여 match score 산정
- on/off 판정 (threshold 이상 → on)
- 상위 레벨 on 노드 → 하위에서 우선 탐색 (top-down prioritized search)
- match_count 높은 노드부터 탐색 순서 결정

이전 MetadataTargetedFilter와 다른 점:
- MetadataTargetedFilter: leaf만 검색
- HierarchicalScorer: 모든 레벨의 모든 노드를 점수화
"""


class HierarchicalScorer:
    """모든 레벨 노드에 대한 keyword overlap 기반 on/off 점수 산정.

    Args:
        match_threshold: 최소 매칭 cue 수 (이 이상이면 on)
    """

    def __init__(self, match_threshold: int = 1, use_key_elements: bool = True):
        self.match_threshold = match_threshold
        self.use_key_elements = use_key_elements

    def score_tree(
        self,
        tree: dict,
        cues: list[str],
        target_fields: list[str] | None = None,
    ) -> dict:
        """전체 트리 노드별 점수 산정.

        Args:
            tree: streaming_memory_tree dict
            cues: search keywords from QueryAnalyzer
            target_fields: 우선 검색 필드 (None이면 모든 필드)

        Returns:
            {
                "scored_levels": {
                    "Level_3": [{"node": {...}, "score": 5, "on": True, ...}, ...],
                    "Level_2": [...],
                    "Level_1": [...],
                    "Level_0": [...]  # leaf children
                },
                "search_order": [...],  # 점수 순 정렬된 탐색 순서
                "total_on": int,
                "total_nodes": int,
            }
        """
        if not cues or not tree:
            return {
                "scored_levels": {},
                "search_order": [],
                "total_on": 0,
                "total_nodes": 0,
            }

        cues_lower = [c.lower().strip() for c in cues if c.strip()]
        all_fields = target_fields or [
            "actions", "objects", "persons", "attributes",
            "locations", "text_ocr", "summary",
        ]

        scored_levels = {}
        all_scored_nodes = []

        # --- Score each level's nodes ---
        levels = sorted(tree.keys(), key=lambda x: int(x.split("_")[1]), reverse=True)

        for level_name in levels:
            scored_nodes = []
            for node_idx, node in enumerate(tree[level_name]):
                result = self._score_node(
                    node, cues_lower, all_fields, level_name, node_idx
                )
                scored_nodes.append(result)
                all_scored_nodes.append(result)

                # Also score children (Level_0 / leaf)
                for child_idx, child in enumerate(node.get("children", [])):
                    if "start_time" in child:
                        child_result = self._score_node(
                            child, cues_lower, all_fields,
                            "Level_0", child_idx,
                            parent_level=level_name, parent_idx=node_idx,
                        )
                        scored_levels.setdefault("Level_0", []).append(child_result)
                        all_scored_nodes.append(child_result)

            scored_levels[level_name] = scored_nodes

        # Deduplicate Level_0 (same leaf from different parents)
        if "Level_0" in scored_levels:
            seen = set()
            unique = []
            for item in scored_levels["Level_0"]:
                node = item["node"]
                key = (float(node.get("start_time", 0)), float(node.get("end_time", 0)))
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            scored_levels["Level_0"] = unique

        # --- Build search order ---
        # Priority: higher level first, then by score descending
        level_priority = {name: i for i, name in enumerate(levels)}
        level_priority["Level_0"] = len(levels)

        search_order = sorted(
            [n for n in all_scored_nodes if n["on"]],
            key=lambda x: (
                level_priority.get(x["level"], 999),
                -x["score"],
                -x["primary_score"],
            ),
        )

        total_on = sum(1 for n in all_scored_nodes if n["on"])
        total_nodes = len(all_scored_nodes)

        return {
            "scored_levels": scored_levels,
            "search_order": search_order,
            "total_on": total_on,
            "total_nodes": total_nodes,
        }

    def get_priority_leaves(
        self,
        score_result: dict,
        budget: int = 15,
    ) -> list[dict]:
        """점수 높은 순서로 leaf entries 추출 (pipeline에서 바로 쓸 수 있는 형태).

        Returns:
            list of {"leaf": dict, "leaf_id": (start,end), "parent_summary": str,
                     "match_score": int, "matched_cues": list}
        """
        level_0 = score_result.get("scored_levels", {}).get("Level_0", [])
        on_leaves = [n for n in level_0 if n["on"]]
        on_leaves.sort(key=lambda x: (-x["score"], -x["primary_score"]))

        # Cap to budget
        selected = on_leaves[:budget]

        entries = []
        for item in selected:
            node = item["node"]
            entries.append({
                "leaf": node,
                "leaf_id": (float(node.get("start_time", 0)),
                            float(node.get("end_time", 0))),
                "parent_summary": item.get("parent_summary", ""),
                "match_score": item["score"],
                "matched_cues": item["matched_cues"],
                "matched_fields": item["matched_fields"],
            })

        return entries

    def find_gap_leaves(
        self,
        score_result: dict,
        already_seen: set,
        cues: list[str],
        covered_cues: set,
        budget: int = 10,
    ) -> list[dict]:
        """Multi-hop gap filling: 아직 안 본 노드 중에서 놓친 cue를 커버하는 leaf 찾기.

        Args:
            score_result: score_tree() 결과
            already_seen: 이미 본 leaf_id set
            cues: 원래 검색 cues
            covered_cues: 이미 커버된 cue set
            budget: 이번 hop에서 볼 최대 leaf 수

        Returns:
            Gap-filling leaf entries
        """
        missing_cues = set(c.lower() for c in cues) - covered_cues

        if not missing_cues:
            # 모든 cue 커버됨 → 그래도 on인데 안 본 노드 반환
            level_0 = score_result.get("scored_levels", {}).get("Level_0", [])
            unseen = [
                n for n in level_0
                if n["on"] and self._node_id(n["node"]) not in already_seen
            ]
            unseen.sort(key=lambda x: -x["score"])
            return self._to_entries(unseen[:budget])

        # missing cue를 커버하는 노드 우선
        level_0 = score_result.get("scored_levels", {}).get("Level_0", [])
        candidates = []
        for item in level_0:
            node_id = self._node_id(item["node"])
            if node_id in already_seen:
                continue
            if not item["on"]:
                continue

            # 이 노드가 missing cue 중 몇 개를 커버하는지
            node_cues = set(item["matched_cues"])
            gap_coverage = len(node_cues & missing_cues)

            if gap_coverage > 0:
                candidates.append((gap_coverage, item))

        candidates.sort(key=lambda x: (-x[0], -x[1]["score"]))
        selected = [c[1] for c in candidates[:budget]]

        return self._to_entries(selected)

    def get_activated_context(
        self,
        tree: dict,
        score_result: dict,
    ) -> str:
        """on 노드들의 계층적 컨텍스트 생성 (top-down).

        상위 레벨 on 노드의 summary를 포함하여 LLM에게 계층 맥락 제공.
        """
        scored_levels = score_result.get("scored_levels", {})
        lines = []

        # Top-down: 높은 레벨부터
        level_names = sorted(
            [k for k in scored_levels if k != "Level_0"],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        )

        for level_name in level_names:
            on_nodes = [n for n in scored_levels[level_name] if n["on"]]
            if not on_nodes:
                continue

            on_nodes.sort(key=lambda x: -x["score"])
            lines.append(f"=== {level_name} (Activated Nodes) ===")
            for item in on_nodes:
                node = item["node"]
                segs = node.get("time_segments", [])
                time_str = ""
                if segs:
                    s, e = segs[0] if isinstance(segs[0], (list, tuple)) else (segs[0], segs[0])
                    time_str = f" [{float(s):.0f}s-{float(e):.0f}s]"
                lines.append(
                    f"  [{item['score']} matches]{time_str} {node.get('summary', '')}"
                )
            lines.append("")

        # Level_0 (leaves) — top matches only
        if "Level_0" in scored_levels:
            on_leaves = [n for n in scored_levels["Level_0"] if n["on"]]
            on_leaves.sort(key=lambda x: -x["score"])
            if on_leaves:
                lines.append(f"=== Activated Segments ({len(on_leaves)} on) ===")
                for item in on_leaves[:20]:
                    node = item["node"]
                    st = float(node.get("start_time", 0))
                    et = float(node.get("end_time", 0))
                    lines.append(
                        f"  [{item['score']} matches] [{st:.0f}s-{et:.0f}s] "
                        f"{node.get('summary', '')}"
                    )

        return "\n".join(lines)

    def get_sibling_leaves(
        self,
        tree: dict,
        target_leaf_ids: set,
        budget: int = 10,
    ) -> list[dict]:
        """Sibling expansion: 타겟 leaf와 같은 Level_1 부모 아래 다른 leaf 반환.

        Escalation 3단계: "같은 그룹에 있는 다른 구간 보기"

        Args:
            tree: streaming_memory_tree
            target_leaf_ids: 이미 본 leaf의 (start, end) set
            budget: 최대 반환 수

        Returns:
            Sibling leaf entries (타겟과 같은 부모, 시간순 정렬)
        """
        if "Level_1" not in tree:
            return []

        # 어떤 Level_1 노드가 타겟 leaf를 포함하는지 찾기
        activated_parents = []
        for l1_node in tree["Level_1"]:
            has_target_child = False
            for child in l1_node.get("children", []):
                if "start_time" in child:
                    child_id = (float(child["start_time"]), float(child["end_time"]))
                    if child_id in target_leaf_ids:
                        has_target_child = True
                        break
            if has_target_child:
                activated_parents.append(l1_node)

        # 활성화된 부모의 다른 children (siblings) 수집
        siblings = []
        for parent in activated_parents:
            parent_summary = parent.get("summary", "")
            for child in parent.get("children", []):
                if "start_time" not in child:
                    continue
                child_id = (float(child["start_time"]), float(child["end_time"]))
                if child_id not in target_leaf_ids:
                    siblings.append({
                        "leaf": child,
                        "leaf_id": child_id,
                        "parent_summary": parent_summary,
                        "match_score": 0,
                        "matched_cues": [],
                        "matched_fields": [],
                        "source": "sibling_expansion",
                    })

        # 시간순 정렬
        siblings.sort(key=lambda x: x["leaf_id"][0])
        return siblings[:budget]

    def get_broader_leaves(
        self,
        tree: dict,
        target_leaf_ids: set,
        already_seen_ids: set,
        budget: int = 10,
    ) -> list[dict]:
        """Broader expansion: 상위 Level_2 부모 아래 다른 Level_1 그룹의 leaf 반환.

        Sibling expansion으로도 부족할 때, 한 단계 더 넓게 탐색.
        """
        if "Level_2" not in tree or "Level_1" not in tree:
            return []

        # target leaf가 속한 Level_1 → 그 Level_1이 속한 Level_2 찾기
        target_l1_ids = set()
        for l1_node in tree["Level_1"]:
            for child in l1_node.get("children", []):
                if "start_time" in child:
                    child_id = (float(child["start_time"]), float(child["end_time"]))
                    if child_id in target_leaf_ids:
                        target_l1_ids.add(id(l1_node))
                        break

        # Level_2에서 target_l1을 포함하는 노드 찾기
        from components.spreading_activation import SpreadingActivation
        sa = SpreadingActivation()

        activated_l2 = set()
        for l2_node in tree["Level_2"]:
            l2_segs = l2_node.get("time_segments", [])
            for l1_node in tree["Level_1"]:
                if id(l1_node) in target_l1_ids:
                    l1_segs = l1_node.get("time_segments", [])
                    if sa._segments_overlap(l2_segs, l1_segs):
                        activated_l2.add(id(l2_node))

        # activated L2 아래 모든 L1 → 그 children 중 안 본 것
        broader = []
        for l2_node in tree["Level_2"]:
            if id(l2_node) not in activated_l2:
                continue
            l2_segs = l2_node.get("time_segments", [])
            for l1_node in tree["Level_1"]:
                if id(l1_node) in target_l1_ids:
                    continue  # 이미 본 그룹 스킵
                l1_segs = l1_node.get("time_segments", [])
                if sa._segments_overlap(l2_segs, l1_segs):
                    parent_summary = l1_node.get("summary", "")
                    for child in l1_node.get("children", []):
                        if "start_time" not in child:
                            continue
                        child_id = (float(child["start_time"]), float(child["end_time"]))
                        if child_id in already_seen_ids:
                            continue
                        broader.append({
                            "leaf": child,
                            "leaf_id": child_id,
                            "parent_summary": parent_summary,
                            "match_score": 0,
                            "matched_cues": [],
                            "matched_fields": [],
                            "source": "broader_expansion",
                        })

        broader.sort(key=lambda x: x["leaf_id"][0])
        return broader[:budget]

    # ========== Internal ==========

    def _score_node(self, node: dict, cues_lower: list[str],
                    target_fields: list[str], level_name: str, node_idx: int,
                    parent_level: str = "", parent_idx: int = -1) -> dict:
        """단일 노드 점수 산정."""
        ke = node.get("key_elements", {}) if self.use_key_elements else {}
        summary = node.get("summary", "").lower()
        caption = node.get("caption", "").lower()

        matched_cues = []
        matched_fields = []
        primary_score = 0

        for cue in cues_lower:
            found = False
            found_primary = False

            # key_elements fields
            for field in target_fields:
                if field == "summary":
                    if cue in summary or cue in caption:
                        found = True
                        found_primary = True
                        matched_fields.append("summary")
                        break
                elif self.use_key_elements:
                    for val in ke.get(field, []):
                        if cue in str(val).lower():
                            found = True
                            found_primary = True
                            matched_fields.append(field)
                            break
                    if found:
                        break

            # Fallback: check ALL fields if not found in target
            if not found and self.use_key_elements:
                all_ke_fields = ["actions", "objects", "persons",
                                 "attributes", "locations", "text_ocr"]
                for field in all_ke_fields:
                    if field in target_fields:
                        continue
                    for val in ke.get(field, []):
                        if cue in str(val).lower():
                            found = True
                            matched_fields.append(field)
                            break
                    if found:
                        break

            # Final fallback: summary text
            if not found and (cue in summary or cue in caption):
                found = True
                matched_fields.append("text")

            if found:
                matched_cues.append(cue)
                if found_primary:
                    primary_score += 1

        score = len(matched_cues)
        on = score >= self.match_threshold

        # Parent summary for leaves
        parent_summary = ""
        if parent_level and parent_idx >= 0:
            # Will be filled by caller context
            parent_summary = node.get("parent_summary", "")

        return {
            "level": level_name,
            "node_idx": node_idx,
            "node": node,
            "score": score,
            "primary_score": primary_score,
            "on": on,
            "matched_cues": matched_cues,
            "matched_fields": list(set(matched_fields)),
            "parent_level": parent_level,
            "parent_idx": parent_idx,
            "parent_summary": parent_summary,
        }

    @staticmethod
    def _node_id(node: dict) -> tuple:
        if "start_time" in node:
            return (float(node["start_time"]), float(node["end_time"]))
        segs = node.get("time_segments", [])
        if segs:
            s = segs[0] if not isinstance(segs[0], (list, tuple)) else segs[0][0]
            return (float(s),)
        return (id(node),)

    @staticmethod
    def _to_entries(items: list[dict]) -> list[dict]:
        entries = []
        for item in items:
            node = item["node"]
            entries.append({
                "leaf": node,
                "leaf_id": (float(node.get("start_time", 0)),
                            float(node.get("end_time", 0))),
                "parent_summary": item.get("parent_summary", ""),
                "match_score": item["score"],
                "matched_cues": item["matched_cues"],
                "matched_fields": item["matched_fields"],
            })
        return entries
