"""
Elimination Reasoner — 4단계: 소거법 기반 MCQ 해결

인지과학적 근거: Elimination-by-Aspects (Tversky, 1972)
- 전체 옵션을 한꺼번에 비교하지 않고, 하나씩 소거
- 각 선지마다 "이 선지가 맞는/틀린 증거"를 메모리에서 찾기
- 남은 선지가 1개면 정답, 2개면 남은 것끼리 최종 비교
- 전혀 소거 못하면 → ForcedAnswerFallback으로 위임

기존 SolvabilityChecker와 다른 점:
- SolvabilityChecker: "풀 수 있는지" 이진 판단
- EliminationReasoner: "어떤 선지를 왜 버리는지" 근거 기반 소거
"""


class EliminationReasoner:
    """소거법 기반 MCQ Solver.

    Step 1: 각 선지별 evidence 검토 (supporting / contradicting)
    Step 2: 명확히 탈락하는 선지 소거
    Step 3: 남은 선지 중 최종 선택
    """

    DEFAULT_PROMPT = """### Role
You are a multiple-choice question solver using elimination reasoning.

### Instructions
1. For EACH option, find supporting or contradicting evidence in the context.
2. Eliminate options that are clearly wrong based on evidence.
3. Among remaining options, choose the best answer.
4. If you cannot eliminate any option, provide your best guess with reasoning.

{history_section}
### Memory Context
{context}

### Question
{question}

### Options
{options_text}

### Output Format (JSON only)
{{
    "option_analysis": [
        {{
            "option": "A",
            "text": "option text",
            "evidence_for": "supporting evidence from context",
            "evidence_against": "contradicting evidence from context",
            "verdict": "keep|eliminate|uncertain"
        }}
    ],
    "eliminated": ["B", "D"],
    "remaining": ["A", "C"],
    "final_answer": "A",
    "confidence": "high|medium|low",
    "reasoning": "Why this answer is correct among remaining options"
}}"""

    COMPARISON_PROMPT = """### Role
You are comparing two remaining options after elimination.

### Context
{context}

### Question
{question}

### Remaining Options (others already eliminated)
{remaining_text}

### Previous Analysis
{analysis_summary}

### Instructions
Compare ONLY these remaining options. Choose the one better supported by the context.

### Output Format (JSON only)
{{
    "comparison": "Direct comparison of the two options",
    "final_answer": "A",
    "confidence": "high|medium|low",
    "reasoning": "Why this option wins over the other"
}}"""

    def __init__(self, llm_fn, prompt_template: str | None = None,
                 comparison_prompt: str | None = None):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.comparison_prompt = comparison_prompt or self.COMPARISON_PROMPT

    def eliminate(self, context: str, question: str, options: list[str],
                  hop_history: list[dict] | None = None,
                  pre_elimination: dict | None = None) -> dict:
        """소거법 기반 답변.

        Args:
            context: memory context string
            question: 질문
            options: ["option A text", "option B text", ...]
            hop_history: 이전 단계 결과들
            pre_elimination: UncertaintyChecker에서 미리 소거한 정보
                {"eliminated": ["B", "D"], "remaining": ["A", "C"]}

        Returns:
            {
                "answer": str,           # "A", "B", "C", "D"
                "confidence": str,       # "high", "medium", "low"
                "reasoning": str,
                "eliminated": list[str],
                "remaining": list[str],
                "option_analysis": list,
                "method": str,           # "full_elimination" | "comparison" | "direct"
            }
        """
        import re

        letters = [chr(65 + i) for i in range(len(options))]

        # If pre-elimination already narrowed to 1 option → direct answer
        if pre_elimination:
            remaining = pre_elimination.get("remaining", letters)
            if len(remaining) == 1:
                return {
                    "answer": remaining[0],
                    "confidence": "high",
                    "reasoning": "All other options eliminated in uncertainty check",
                    "eliminated": pre_elimination.get("eliminated", []),
                    "remaining": remaining,
                    "option_analysis": [],
                    "method": "pre_eliminated",
                }

        # If pre-elimination narrowed to 2 → comparison mode
        if pre_elimination and len(pre_elimination.get("remaining", letters)) == 2:
            return self._compare_remaining(
                context, question, options,
                pre_elimination["remaining"],
                pre_elimination.get("elimination_reasons", []),
            )

        # Full elimination
        return self._full_elimination(context, question, options, hop_history)

    def _full_elimination(self, context: str, question: str,
                          options: list[str],
                          hop_history: list[dict] | None = None) -> dict:
        """전체 소거 프로세스."""
        import re

        opt_text = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
        )

        history_section = ""
        if hop_history:
            lines = []
            for h in hop_history:
                if h.get("type") == "visual_backtrack" and h.get("observation"):
                    lines.append(f"- Visual: {h['observation'][:200]}")
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

        result = self.llm_fn(prompt, max_tokens=800)

        # Parse answer
        answer = result.get("final_answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            answer = m.group(0) if m else None

        eliminated = result.get("eliminated", [])
        remaining = result.get("remaining", [])
        option_analysis = result.get("option_analysis", [])
        confidence = result.get("confidence", "medium")

        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        # If LLM returned 2 remaining → run comparison for higher accuracy
        if len(remaining) == 2 and confidence != "high":
            analysis_summary = result.get("reasoning", "")
            comp_result = self._compare_remaining(
                "", question, options, remaining,
                [f"{e} eliminated" for e in eliminated],
            )
            # Use comparison result if it's more confident
            if comp_result["confidence"] in ("high", "medium"):
                comp_result["option_analysis"] = option_analysis
                comp_result["eliminated"] = eliminated
                return comp_result

        if not answer:
            # Fallback: pick first remaining
            if remaining:
                answer = remaining[0]
            else:
                answer = "A"

        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "eliminated": eliminated,
            "remaining": remaining,
            "option_analysis": option_analysis,
            "method": "full_elimination",
        }

    def _compare_remaining(self, context: str, question: str,
                           options: list[str], remaining: list[str],
                           elimination_reasons: list[str]) -> dict:
        """2개 남은 선지 직접 비교."""
        import re

        remaining_text = "\n".join(
            f"{letter}. {options[ord(letter) - 65]}"
            for letter in remaining
            if ord(letter) - 65 < len(options)
        )

        analysis_summary = "\n".join(
            f"- {r}" for r in elimination_reasons
        ) if elimination_reasons else "No prior elimination info"

        prompt = self.comparison_prompt.format(
            context=context,
            question=question,
            remaining_text=remaining_text,
            analysis_summary=analysis_summary,
        )

        result = self.llm_fn(prompt, max_tokens=400)

        answer = result.get("final_answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            answer = m.group(0) if m else remaining[0]
        else:
            answer = remaining[0]

        eliminated_others = [
            chr(65 + i) for i in range(len(options))
            if chr(65 + i) not in remaining
        ]

        return {
            "answer": answer,
            "confidence": result.get("confidence", "medium"),
            "reasoning": result.get("reasoning", ""),
            "eliminated": eliminated_others,
            "remaining": remaining,
            "option_analysis": [],
            "method": "comparison",
        }
