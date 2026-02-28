"""
Metadata Targeted Filter — 2단계: 메타데이터 교차 필터링

인지과학적 근거: Schema-based Retrieval (Bartlett)
- 질문 유형에 따라 특정 key_elements 필드만 타겟팅
- 순서 추론 (Temporal Reasoning): 시간순 정렬 후 N번째 추출
- 조건부 검색 (Conditional Search): 특정 조건의 노드에서 다른 필드 추출

기존 RuleBasedFilter는 '키워드 매칭'만 수행.
MetadataTargetedFilter는 '구조화된 필드 단위 검색'을 수행.
"""


class MetadataTargetedFilter:
    """key_elements 필드 기반 구조화된 검색.

    RuleBasedFilter와 다른 점:
    - 어떤 필드를 검색할지 QueryAnalyzer가 결정
    - 필드 내 값의 semantic matching (단순 substring 뿐 아니라)
    - 조건부 교차 검색 지원
    """

    def filter_by_fields(
        self,
        all_leaves: list[dict],
        cues: list[str],
        target_fields: list[str],
        secondary_fields: list[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """타겟 필드 우선 검색.

        target_fields에서 먼저 매칭 → match_count 높게
        secondary_fields에서 추가 매칭 → match_count 낮게
        둘 다 미매칭 → unmarked

        Args:
            all_leaves: flattened leaf entries
            cues: search keywords
            target_fields: primary key_elements fields to search
            secondary_fields: fallback fields

        Returns:
            (marked_leaves, unmarked_leaves)
        """
        cues_lower = [c.lower().strip() for c in cues if c.strip()]
        if not cues_lower:
            return [], list(all_leaves)

        secondary_fields = secondary_fields or []
        marked = []
        unmarked = []

        for entry in all_leaves:
            leaf = entry["leaf"]
            ke = leaf.get("key_elements", {})

            # Phase 1: Target fields
            primary_matches = []
            for cue in cues_lower:
                for field in target_fields:
                    if field == "summary":
                        if cue in leaf.get("summary", "").lower():
                            primary_matches.append((cue, field))
                            break
                        if cue in leaf.get("caption", "").lower():
                            primary_matches.append((cue, "caption"))
                            break
                    else:
                        for val in ke.get(field, []):
                            if cue in str(val).lower():
                                primary_matches.append((cue, field))
                                break
                        else:
                            continue
                        break

            # Phase 2: Secondary fields (if primary didn't match all cues)
            secondary_matches = []
            matched_cues_primary = {m[0] for m in primary_matches}
            remaining_cues = [c for c in cues_lower if c not in matched_cues_primary]

            for cue in remaining_cues:
                for field in secondary_fields:
                    if field == "summary":
                        if cue in leaf.get("summary", "").lower():
                            secondary_matches.append((cue, field))
                            break
                    else:
                        for val in ke.get(field, []):
                            if cue in str(val).lower():
                                secondary_matches.append((cue, field))
                                break
                        else:
                            continue
                        break

            # Also check parent_summary
            parent_sum = entry.get("parent_summary", "").lower()
            for cue in remaining_cues:
                if cue not in {m[0] for m in secondary_matches} and cue in parent_sum:
                    secondary_matches.append((cue, "parent_summary"))

            all_matches = primary_matches + secondary_matches
            if all_matches:
                enriched = dict(entry)
                enriched["match_count"] = len(all_matches)
                enriched["primary_match_count"] = len(primary_matches)
                enriched["matched_cues"] = list({m[0] for m in all_matches})
                enriched["matched_fields"] = list({m[1] for m in all_matches})
                marked.append(enriched)
            else:
                unmarked.append(entry)

        # Sort: primary matches first, then total matches, then by time
        marked.sort(key=lambda x: (
            -x["primary_match_count"],
            -x["match_count"],
            x["leaf_id"][0],
        ))

        return marked, unmarked

    def filter_by_time(
        self,
        all_leaves: list[dict],
        time_ranges: list[tuple[float, float]],
    ) -> list[dict]:
        """시간 범위로 leaf 필터링.

        O(N) 단순 교집합 — Track A에서 LLM 호출 없이 즉시 색인.

        Args:
            all_leaves: flattened leaf entries
            time_ranges: [(start_sec, end_sec), ...]

        Returns:
            Matching leaf entries (시간순 정렬)
        """
        if not time_ranges:
            return []

        matched = []
        for entry in all_leaves:
            leaf = entry["leaf"]
            leaf_start = float(leaf.get("start_time", 0))
            leaf_end = float(leaf.get("end_time", 0))

            for t_start, t_end in time_ranges:
                # 교집합 존재 여부
                if max(t_start, leaf_start) < min(t_end, leaf_end):
                    enriched = dict(entry)
                    enriched["matched_time_range"] = (t_start, t_end)
                    matched.append(enriched)
                    break

        matched.sort(key=lambda x: x["leaf_id"][0])
        return matched

    def get_nth_event(
        self,
        all_leaves: list[dict],
        n: int,
        field: str = "summary",
    ) -> dict | None:
        """시간순 N번째 이벤트 추출 (Temporal Reasoning).

        "What is the second news item?" → n=2

        Args:
            all_leaves: flattened leaf entries (시간순 정렬되어 있어야 함)
            n: 1-based index
            field: 추출할 필드

        Returns:
            Leaf entry or None
        """
        sorted_leaves = sorted(all_leaves, key=lambda x: x["leaf_id"][0])
        if 1 <= n <= len(sorted_leaves):
            return sorted_leaves[n - 1]
        return None

    def conditional_search(
        self,
        all_leaves: list[dict],
        condition_field: str,
        condition_value: str,
        extract_field: str,
    ) -> list[dict]:
        """조건부 교차 검색.

        "파란 셔츠 입은 남자가 주문할 때, 브이로거는 뭐해?"
        → condition: persons에 "blue shirt"가 있는 노드
        → extract: 동일 노드의 actions에서 브이로거 행동 추출

        Args:
            condition_field: 조건 필드 (e.g., "persons")
            condition_value: 조건 값 (e.g., "blue shirt")
            extract_field: 추출 필드 (e.g., "actions")

        Returns:
            Matching leaf entries with extracted values
        """
        condition_lower = condition_value.lower()
        results = []

        for entry in all_leaves:
            leaf = entry["leaf"]
            ke = leaf.get("key_elements", {})

            # Check condition
            field_values = ke.get(condition_field, [])
            condition_met = any(condition_lower in str(v).lower() for v in field_values)

            if condition_met:
                enriched = dict(entry)
                enriched["extracted_values"] = ke.get(extract_field, [])
                enriched["condition_matched"] = condition_value
                results.append(enriched)

        return results
