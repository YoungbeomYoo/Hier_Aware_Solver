from __future__ import annotations

"""
History Accumulator — VideoLucy 스타일 삭제 없는 hop 정보 누적

HistoryCompactor와의 차이:
- Compactor: max_obs 초과 시 앞부분 삭제 + search_state 덮어쓰기
- Accumulator: 모든 hop 정보를 삭제 없이 누적, 이미 탐색한 구간 추적

VideoLucy의 fine_memory_history 패턴:
- 각 hop의 context + verdict + visual caption을 그대로 쌓음
- excluded_time_periods로 이미 본 구간 재방문 방지
- 전체 누적 history를 judge에게 전달
"""

from components.token_utils import TokenBudget


class HistoryAccumulator:
    """Accumulate all hop information without deletion.

    Args:
        token_budget: TokenBudget instance (None → char fallback)
        max_total_tokens: maximum total tokens for accumulated history
    """

    def __init__(
        self,
        token_budget: TokenBudget | None = None,
        max_total_tokens: int = 50000,
    ):
        self.tb = token_budget or TokenBudget(None)
        self.max_total = max_total_tokens
        self.hop_entries: list[dict] = []

    def accumulate(
        self,
        hop_number: int,
        context: str,
        verdict: dict,
        target_time_ranges: list[tuple[float, float]] | None = None,
        visual_captions: str | None = None,
    ):
        """Add hop result to history. Never deletes previous entries.

        Args:
            hop_number: 1-indexed hop number
            context: assembled context text from this hop
            verdict: judge output dict
            target_time_ranges: [(start, end), ...] explored in this hop
            visual_captions: caption text from visual judge (if any)
        """
        entry = {
            "hop": hop_number,
            "time_ranges": target_time_ranges or [],
            "verdict": {
                "answerable": verdict.get("answerable", False),
                "answer": verdict.get("answer"),
                "confidence": verdict.get("confidence", "low"),
                "reasoning": verdict.get("reasoning", ""),
                "missing_info": verdict.get("missing_info"),
                "search_direction": verdict.get("search_direction"),
            },
            "context_summary": self._summarize_context(context),
            "visual_captions": visual_captions,
        }
        self.hop_entries.append(entry)

    def format_for_judge(self, max_tokens: int = 20000) -> str:
        """Format accumulated history for judge consumption.

        Returns full accumulated history, truncating oldest entries
        only if total exceeds max_tokens.
        """
        if not self.hop_entries:
            return ""

        parts = ["=== Search History (all previous hops) ==="]

        for entry in self.hop_entries:
            hop = entry["hop"]
            v = entry["verdict"]
            section = f"\n--- Hop {hop} ---"

            # Time ranges explored
            if entry["time_ranges"]:
                ranges_str = ", ".join(
                    f"{s:.1f}s-{e:.1f}s" for s, e in entry["time_ranges"]
                )
                section += f"\nExplored: {ranges_str}"

            # Context summary
            if entry["context_summary"]:
                section += f"\nContext: {entry['context_summary']}"

            # Visual observations
            if entry["visual_captions"]:
                section += f"\nVisual observations:\n{entry['visual_captions']}"

            # Verdict
            if v["answerable"] and v["answer"]:
                section += (
                    f"\nVerdict: tentative answer {v['answer']} "
                    f"({v['confidence']})"
                )
            else:
                section += f"\nVerdict: not answerable"

            if v["reasoning"]:
                section += f"\nReasoning: {v['reasoning']}"

            if v["missing_info"]:
                section += f"\nMissing: {v['missing_info']}"

            parts.append(section)

        full_text = "\n".join(parts)

        # If over budget, truncate oldest entries (keep recent)
        if self.tb.count(full_text) > max_tokens:
            full_text = self.tb.truncate_keep_tail(
                full_text, max_tokens,
                prefix="...(earlier hops truncated)...\n",
            )

        return full_text

    def get_explored_time_ranges(self) -> list[tuple[float, float]]:
        """Return all time ranges explored across all hops.

        Used for excluded_time_periods pattern (VideoLucy style)
        to avoid revisiting already-examined regions.
        """
        explored = []
        for entry in self.hop_entries:
            explored.extend(entry["time_ranges"])
        return explored

    def get_hop_count(self) -> int:
        """Return number of accumulated hops."""
        return len(self.hop_entries)

    def get_last_verdict(self) -> dict | None:
        """Return the most recent verdict."""
        if self.hop_entries:
            return self.hop_entries[-1]["verdict"]
        return None

    def _summarize_context(self, context: str, max_chars: int = 1000) -> str:
        """Extract key info from context for compact storage."""
        if not context:
            return ""
        if len(context) <= max_chars:
            return context
        # Keep first and last portions
        half = max_chars // 2
        return context[:half] + "\n...\n" + context[-half:]
