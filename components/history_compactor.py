from __future__ import annotations

"""
History Compactor — hop 간 정보 전달을 위한 압축

두 가지 핵심 구분:
1. Rich (observations): caption/frame에서 얻은 정보 → 최대한 많이, 상세하게
2. Compact (search state): 검색 상태, 실패 이유, 다음 할 일 → 짧고 명확하게

"질문을 풀고있고 기존에는 이런걸 봤었는데 이런걸 못했어서
 내가 지금 이걸 찾으러 다니고 있다"
"""

from components.token_utils import TokenBudget


class HistoryCompactor:
    """Compress hop history into rich observations + compact search state.

    Args:
        max_observation_tokens: observations 영역 최대 토큰 수
        max_state_tokens: search state 영역 최대 토큰 수
        token_budget: TokenBudget instance (None → char fallback)
    """

    STATE_TEMPLATE = """=== Search State (Hop {hop}) ===
Question: {question_brief}
Status: {status}
Visited: {visited_summary}
Key findings so far: {findings}
Reason for continuing: {reason}
Looking for: {next_action}"""

    def __init__(
        self,
        max_observation_tokens: int = 8000,
        max_state_tokens: int = 800,
        token_budget: TokenBudget | None = None,
        # Legacy char-based params (ignored if token_budget provided)
        max_observation_chars: int | None = None,
        max_state_chars: int | None = None,
    ):
        self.tb = token_budget or TokenBudget(None)
        self.max_obs = max_observation_tokens
        self.max_state = max_state_tokens

    def compact(
        self,
        question: str,
        observations_context: str,
        judge_result: dict,
        hop_number: int,
        traversal_log: list[dict],
        prev_compact: dict | None = None,
    ) -> dict:
        """Compress history for next hop.

        Args:
            question: original question
            observations_context: rich context from current hop
            judge_result: judge output
            hop_number: current hop number
            traversal_log: list of traversal entries
            prev_compact: previous compact (to accumulate)

        Returns:
            {
                "compact_text": str,  # full text for LLM
                "observations_rich": str,  # accumulated observations
                "search_state": str,  # compact state summary
                "traversal_summary": str,  # path trace
            }
        """
        # =============================================
        # 1. Accumulate rich observations
        # =============================================
        prev_obs = ""
        if prev_compact:
            prev_obs = prev_compact.get("observations_rich", "")

        # Add new observations with hop marker
        new_section = f"\n--- Hop {hop_number} ---\n{observations_context}"
        all_obs = prev_obs + new_section

        # Trim from the beginning if over budget (keep recent)
        if self.tb.count(all_obs) > self.max_obs:
            all_obs = self.tb.truncate_keep_tail(
                all_obs, self.max_obs,
                prefix="...(earlier observations truncated)...\n",
            )

        # =============================================
        # 2. Build compact search state
        # =============================================
        question_brief = question[:80]
        if len(question) > 80:
            question_brief += "..."

        # Status
        if judge_result.get("answerable") and judge_result.get("answer"):
            status = (
                f"tentative answer: {judge_result['answer']} "
                f"({judge_result.get('confidence', '?')}), looking for confirmation"
            )
        else:
            status = "searching — not enough info yet"

        # Visited paths
        visited_parts = []
        for entry in traversal_log[-4:]:  # Last 4 hops
            path = entry.get("path", "?")
            hop = entry.get("hop", "?")
            judge_info = entry.get("judge", {})
            conf = judge_info.get("confidence", "?")
            visited_parts.append(f"hop{hop}({path}, {conf})")
        visited_summary = " → ".join(visited_parts)

        # Key findings
        findings = judge_result.get("reasoning", "none")[:300]

        # Reason & next action
        reason = "unknown"
        next_action = "continue searching"

        if not judge_result.get("answerable"):
            reason = judge_result.get("reasoning", "insufficient context")[:200]
            if judge_result.get("missing_info"):
                next_action = f"Need: {judge_result['missing_info'][:150]}"
            elif judge_result.get("search_direction"):
                direction_map = {
                    "earlier_time": "Look at earlier segments",
                    "later_time": "Look at later segments",
                    "same_region_detail": "Look more carefully at the same time region",
                    "different_topic": "Search a different topic area",
                    "broader_context": "Look at broader/higher-level context",
                }
                direction = judge_result["search_direction"]
                next_action = direction_map.get(direction, direction)
        else:
            reason = f"Have tentative answer but only {judge_result.get('confidence', 'low')} confidence"
            next_action = "Verify or find stronger evidence"

        search_state = self.STATE_TEMPLATE.format(
            hop=hop_number,
            question_brief=question_brief,
            status=status,
            visited_summary=visited_summary,
            findings=findings,
            reason=reason,
            next_action=next_action,
        )

        # Trim state if needed
        search_state = self.tb.truncate(search_state, self.max_state)

        # =============================================
        # 3. Combined compact text
        # =============================================
        compact_text = (
            search_state
            + "\n\n=== Accumulated Observations ===\n"
            + all_obs
        )

        # Traversal summary (full trace)
        traversal_summary = " → ".join(
            f"Hop{e.get('hop', '?')}({e.get('path', '?')})"
            for e in traversal_log
        )

        return {
            "compact_text": compact_text,
            "observations_rich": all_obs,
            "search_state": search_state,
            "traversal_summary": traversal_summary,
        }
