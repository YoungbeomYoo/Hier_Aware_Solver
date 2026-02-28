"""
Filters — 필터링 모듈

- RuleBasedFilter: LLM 없이 키워드 매칭 (비용 0)
- LLMLeafSelector: LLM으로 budgeted leaf selection
"""


class RuleBasedFilter:
    """Cue 기반 키워드 매칭 필터.

    leaf의 summary, caption, key_elements에서 cue를 검색하여
    matched/unmatched로 분류. LLM 호출 없음.
    """

    SEARCHABLE_KE_CATEGORIES = [
        "persons", "actions", "objects", "locations", "text_ocr", "attributes"
    ]

    def filter(self, all_leaves: list[dict], cues: list[str]) -> tuple[list[dict], list[dict]]:
        """Rule-based keyword filtering.

        Args:
            all_leaves: list of {"leaf": {...}, "leaf_id": (s,e), "parent_summary": str}
            cues: list of search keywords

        Returns:
            (marked_leaves, unmarked_leaves)
            marked_leaves에는 match_count, matched_cues가 추가됨.
        """
        cues_lower = [c.lower().strip() for c in cues if c.strip()]
        marked = []
        unmarked = []

        for entry in all_leaves:
            leaf = entry["leaf"]

            # Build searchable text fragments
            searchable = []
            searchable.append(leaf.get("summary", "").lower())
            searchable.append(leaf.get("caption", "").lower())
            searchable.append(entry.get("parent_summary", "").lower())

            ke = leaf.get("key_elements", {})
            for category in self.SEARCHABLE_KE_CATEGORIES:
                for val in ke.get(category, []):
                    searchable.append(str(val).lower())

            # Check each cue
            matched_cues = []
            for cue in cues_lower:
                for text in searchable:
                    if cue in text:
                        matched_cues.append(cue)
                        break

            if matched_cues:
                enriched = dict(entry)
                enriched["match_count"] = len(matched_cues)
                enriched["matched_cues"] = matched_cues
                marked.append(enriched)
            else:
                unmarked.append(entry)

        # Sort: most matches first, then by start_time
        marked.sort(key=lambda x: (-x["match_count"], x["leaf_id"][0]))

        return marked, unmarked


class LLMLeafSelector:
    """LLM 기반 leaf selection.

    후보 leaf 중에서 budget 개를 LLM으로 선택.
    prompt_template 교체로 selection 전략 변경 가능.
    """

    DEFAULT_PROMPT = """You are selecting video segments to examine for answering a question.
Pick the {budget} most promising segments based on how well they match the search cues and question.

[Question]: {question}
[Search Cues]: {cues_str}
{history_section}

[Candidate Segments]
{leaf_descriptions}

Select exactly {budget} segment IDs. Prioritize:
1. Segments with more matched cues
2. Segments whose summary/key elements directly relate to the question
3. Segments covering different time ranges (temporal diversity)

Output ONLY valid JSON:
{{"selected_ids": [0, 1, 2], "reasoning": "brief explanation"}}"""

    def __init__(self, llm_fn, prompt_template: str | None = None,
                 format_leaf_fn=None):
        """
        Args:
            llm_fn: callable(prompt_text, max_tokens) -> dict
            prompt_template: str with {budget}, {question}, {cues_str}, {history_section}, {leaf_descriptions}
            format_leaf_fn: callable(idx, leaf_entry) -> str (1-line format)
        """
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.format_leaf_fn = format_leaf_fn or self._default_format

    @staticmethod
    def _default_format(idx: int, leaf_entry: dict) -> str:
        leaf = leaf_entry["leaf"]
        st, et = leaf.get("start_time", 0), leaf.get("end_time", 0)
        summary = leaf.get("summary", "N/A")
        matched = leaf_entry.get("matched_cues", [])
        match_str = f" | Matched: [{', '.join(matched)}]" if matched else ""
        return f"ID {idx} | {st:.0f}s-{et:.0f}s | {summary[:200]}{match_str}"

    def select(self, candidates: list[dict], cues: list[str],
               question: str, options: list[str],
               hop_history: list[dict] | None = None,
               budget: int = 10) -> list[int]:
        """LLM으로 leaf selection.

        Args:
            candidates: leaf entry list
            cues: search cues
            question: question text
            options: choice list
            hop_history: previous hop records
            budget: max leaves to select

        Returns:
            list of selected indices
        """
        if len(candidates) <= budget:
            return list(range(len(candidates)))

        cues_str = ", ".join(cues)
        leaf_descriptions = "\n".join(
            self.format_leaf_fn(i, entry) for i, entry in enumerate(candidates)
        )

        # Build history
        history_text = ""
        if hop_history:
            for h in hop_history:
                if h.get("type") == "frame_inference" and h.get("observation"):
                    history_text += f"- Hop {h['hop']}: {h['observation'][:150]}\n"
                elif h.get("type") == "leaf_solvability" and h.get("reasoning"):
                    history_text += f"- Previous check: {h['reasoning'][:150]}\n"

        history_section = f"\n[Previous Observations]\n{history_text}" if history_text else ""

        prompt = self.prompt_template.format(
            budget=budget,
            question=question,
            cues_str=cues_str,
            history_section=history_section,
            leaf_descriptions=leaf_descriptions,
        )

        result = self.llm_fn(prompt, max_tokens=200)
        selected_ids = result.get("selected_ids", [])
        valid_ids = [i for i in selected_ids if isinstance(i, int) and 0 <= i < len(candidates)]

        if not valid_ids:
            # Fallback: top-N (already sorted by match_count)
            valid_ids = list(range(min(budget, len(candidates))))

        return valid_ids[:budget]
