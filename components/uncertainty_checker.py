"""
Uncertainty Checker — 3단계: 불확실성 기반 시각적 역추적 판단

인지과학적 근거: Prediction Error → Backtracking
- 텍스트 메모리만으로 충분한지 정밀 판단
- 부족하면 "어느 구간의 raw video를 봐야 하는지" 결정
- 기존 SolvabilityChecker보다 세밀: WHY 부족한지 + WHERE를 봐야 하는지

VideoLucy 방식: 이미 알고 있는 시간대의 19초짜리 클립만 호출
"""


class UncertaintyChecker:
    """불확실성 평가 + 시각적 역추적 결정.

    기존 SolvabilityChecker와 다른 점:
    - solvable/not-solvable 이진 판단이 아니라 세분화된 uncertainty level
    - visual backtracking이 필요할 때 정확한 시간 구간 지정
    - 어떤 종류의 visual 정보가 필요한지 명시
    """

    DEFAULT_PROMPT = """### Role
You are an "Uncertainty Assessment Agent". Evaluate whether the memory context is sufficient to answer the question.

### Instructions
1. Read the question, options, and memory context.
2. Assess your confidence:
   - "certain": Context clearly supports ONE specific answer. No visual needed.
   - "likely": Strong evidence for one answer, but minor ambiguity. No visual needed.
   - "uncertain": Partial evidence exists but not enough to distinguish options. Visual may help.
   - "insufficient": Context has NO relevant information. Visual is essential.
3. If uncertain/insufficient, specify WHAT visual information is needed and WHERE (time range).

{history_section}
### Memory Context
{context}

### Question
{question}

### Options
{options_text}

### Output Format (JSON only)
{{
    "confidence": "certain|likely|uncertain|insufficient",
    "reasoning": "Why this confidence level",
    "answer": "A|B|C|D or null if insufficient",
    "needs_visual": true/false,
    "visual_reason": "What specific visual detail is needed (e.g., 'exact hand position', 'text on screen')",
    "visual_time_ranges": [[start_sec, end_sec], ...],
    "elimination": {{
        "eliminated": ["B", "D"],
        "elimination_reasons": ["B contradicts X", "D is about Y not Z"],
        "remaining": ["A", "C"]
    }}
}}"""

    def __init__(self, llm_fn, prompt_template: str | None = None):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def assess(self, context: str, question: str, options: list[str],
               hop_history: list[dict] | None = None) -> dict:
        """불확실성 평가.

        Returns:
            {
                "confidence": str,  # certain/likely/uncertain/insufficient
                "answer": str | None,
                "reasoning": str,
                "needs_visual": bool,
                "visual_reason": str,
                "visual_time_ranges": list,
                "elimination": dict,
            }
        """
        import re

        opt_text = "\n".join(options)

        history_section = ""
        if hop_history:
            lines = []
            for h in hop_history:
                if h.get("type") == "visual_backtrack" and h.get("observation"):
                    lines.append(f"- Visual check: {h['observation'][:200]}")
                elif h.get("reasoning"):
                    lines.append(f"- Previous: {h['reasoning'][:200]}")
            if lines:
                history_section = "\n### Previous Analysis\n" + "\n".join(lines) + "\n"

        prompt = self.prompt_template.format(
            history_section=history_section,
            context=context,
            question=question,
            options_text=opt_text,
        )

        result = self.llm_fn(prompt, max_tokens=512)

        confidence = result.get("confidence", "uncertain")
        if confidence not in ("certain", "likely", "uncertain", "insufficient"):
            confidence = "uncertain"

        answer = result.get("answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            answer = m.group(0) if m else None

        needs_visual = result.get("needs_visual", confidence in ("uncertain", "insufficient"))

        return {
            "confidence": confidence,
            "answer": answer,
            "reasoning": result.get("reasoning", ""),
            "needs_visual": needs_visual,
            "visual_reason": result.get("visual_reason", ""),
            "visual_time_ranges": result.get("visual_time_ranges", []),
            "elimination": result.get("elimination", {}),
        }
