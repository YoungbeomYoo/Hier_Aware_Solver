from __future__ import annotations

"""
Solvability Judge — "답할 수 있는가?"만 판단하는 전담 컴포넌트

기존 SolvabilityChecker/UncertaintyChecker와 다른 점:
- 답을 고르는 게 아니라, 답할 수 있는지만 판단
- 답할 수 없다면: 정확히 뭐가 부족한지, 어디를 봐야 하는지 구조적으로 출력
- History-aware: 이전 hop에서 뭘 봤는지 고려
"""

import re
from components.token_utils import TokenBudget


class SolvabilityJudge:
    """Dedicated judge: can the question be answered from given context?

    Output structure:
    - answerable=True  → answer, confidence, reasoning
    - answerable=False → missing_info, search_direction

    Args:
        llm_fn: text-only LLM callable
        prompt_template: override prompt
        token_budget: TokenBudget instance (None → char fallback)
        context_budget: max tokens for context in prompt
        history_budget: max tokens for history in prompt
    """

    JUDGE_PROMPT = """You are a solvability judge for video question answering. Your ONLY job is to determine whether the given context contains sufficient information to confidently answer the question.

### Question
{question}

### Options
{options_text}

### Available Context
{context}
{history_section}
### Task
1. Carefully analyze whether the context provides enough information to pick ONE correct answer.
2. If you CAN answer: state the answer and explain WHY the context supports it.
3. If you CANNOT answer: explain EXACTLY what information is missing and suggest where to look.

Output ONLY valid JSON:
{{
    "answerable": true or false,
    "answer": "A" or "B" or "C" or "D" or null,
    "confidence": "high" or "medium" or "low",
    "reasoning": "Specific explanation of why you can or cannot answer",
    "missing_info": "What specific information is still needed to answer" or null,
    "search_direction": "earlier_time|later_time|same_region_detail|different_topic|broader_context" or null
}}"""

    def __init__(
        self,
        llm_fn=None,
        prompt_template: str | None = None,
        token_budget: TokenBudget | None = None,
        context_budget: int = 20000,
        history_budget: int = 4000,
        answer_judge: bool = False,
    ):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.JUDGE_PROMPT
        self.tb = token_budget or TokenBudget(None)
        self.context_budget = context_budget
        self.history_budget = history_budget
        self.answer_judge = answer_judge

    def judge(
        self,
        context: str,
        question: str,
        options: list[str],
        history_compact: str | None = None,
    ) -> dict:
        """Judge solvability.

        Args:
            context: assembled context text
            question: 질문
            options: 선택지 리스트
            history_compact: 이전 hop의 compact history text (optional)

        Returns:
            {
                "answerable": bool,
                "answer": str | None (A/B/C/D),
                "confidence": str (high/medium/low),
                "reasoning": str,
                "missing_info": str | None,
                "search_direction": str | None,
            }
        """
        default = {
            "answerable": False, "answer": None, "confidence": "low",
            "reasoning": "No LLM available",
            "missing_info": "LLM required for judgement",
            "search_direction": None,
        }

        if not self.llm_fn:
            return default

        opt_text = "\n".join(
            f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        )

        history_section = ""
        if history_compact:
            history_section = (
                "\n### Search History (previous hops)\n"
                + self.tb.truncate(history_compact, self.history_budget)
            )

        prompt = self.prompt_template.format(
            question=question,
            options_text=opt_text,
            context=self.tb.truncate(context, self.context_budget),
            history_section=history_section,
        )

        try:
            result = self.llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"      [Judge] Error: {e}")
            return default

        if not isinstance(result, dict):
            return {
                **default,
                "reasoning": str(result)[:500],
            }

        # Validate answer format
        answer = result.get("answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            result["answer"] = m.group(0) if m else None

        # Normalize confidence
        conf = str(result.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        result["confidence"] = conf

        # Ensure boolean
        result["answerable"] = bool(result.get("answerable", False))

        # Answer judge: 2nd pass verification (C3)
        if self.answer_judge and result.get("answerable") and result.get("answer"):
            verified = self._answer_judge_pass(
                context, question, options, result, history_compact,
            )
            if verified:
                result["answer"] = verified
                result["answer_judge_used"] = True

        return result

    # ==================== Answer Judge (C3) ====================

    ANSWER_JUDGE_PROMPT = """A video QA system produced the following analysis. Your job is to independently verify the answer by re-reading the context and reasoning.

### Question
{question}

### Options
{options_text}

### Context
{context}

### System's Analysis
Reasoning: {reasoning}
Proposed answer: {proposed_answer}

### Task
Re-evaluate the reasoning against the context. Select the single best answer option.
- If the system's reasoning is sound and supported by context, confirm the answer.
- If you find a flaw or better evidence for another option, pick the correct one.

Output ONLY valid JSON:
{{"answer": "A" or "B" or "C" or "D"}}"""

    def _answer_judge_pass(self, context, question, options, verdict, history_compact):
        """Second-pass answer verification (VideoLucy answer_judge style)."""
        opt_text = "\n".join(
            f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        )
        ctx = self.tb.truncate(context, self.context_budget)
        prompt = self.ANSWER_JUDGE_PROMPT.format(
            question=question,
            options_text=opt_text,
            context=ctx,
            reasoning=verdict.get("reasoning", "")[:500],
            proposed_answer=verdict.get("answer", ""),
        )
        try:
            result = self.llm_fn(prompt, max_tokens=50)
        except Exception:
            return None
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        m = re.search(r"[ABCD]", answer.upper())
        return m.group(0) if m else None
