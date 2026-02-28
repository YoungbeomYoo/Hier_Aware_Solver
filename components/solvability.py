"""
Solvability — 풀 수 있는지 판단 모듈

- SolvabilityChecker: memory context만으로 답변 가능 여부 판단
- ForcedAnswerFallback: 못 풀었을 때 강제 답변
"""

import re


class SolvabilityChecker:
    """Memory context만으로 질문에 답할 수 있는지 판단.

    prompt_template 교체로 strict/relaxed 전략 전환 가능.
    """

    DEFAULT_PROMPT = """### Role
You are a strict "Evidence-Based QA Agent". Your task is to answer the user's question based ONLY on the provided memory context.

### Instructions
1. Read the question and the memory context carefully.
2. Determine if the memory context contains sufficient specific information to answer the question completely and accurately.
   - Do NOT use outside knowledge.
   - Do NOT guess or hallucinate details not present in the context.
   - If the context lacks specific details the question asks for (e.g., exact values, player numbers, on-screen statistics), you MUST mark it as unsolvable.
   - Pay special attention to "On-screen text" entries — they contain text visible in the video (scores, player names, statistics).
3. Return a JSON object.
{history_section}
### Memory Context{batch_note}
{leaf_context}

### Question
{question}

### Options
{options_text}

### Output Format (JSON only)
{{
    "reasoning": "Step-by-step reasoning on whether the context supports the answer.",
    "solvable": true/false,
    "needs_depth": true/false,
    "answer": "A or B or C or D (letter only). If solvable is false, set to null."
}}
- "needs_depth": true if you found partial evidence in some segments and need more visual detail to confirm."""

    def __init__(self, llm_fn, prompt_template: str | None = None):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def check(self, context: str, question: str, options: list[str],
              hop_history: list[dict] | None = None,
              batch_info: tuple[int, int] | None = None) -> dict:
        """Solvability check.

        Args:
            context: formatted memory context text
            question: question text
            options: choice list
            hop_history: previous hop records
            batch_info: (batch_num, total_batches) or None

        Returns:
            {"solvable": bool, "answer": str|None, "reasoning": str, "needs_depth": bool}
        """
        opt_text = "\n".join(options)

        # History section
        history_section = ""
        if hop_history:
            history_lines = []
            for h in hop_history:
                htype = h.get("type", "unknown")
                if htype == "leaf_solvability" and h.get("reasoning"):
                    history_lines.append(
                        f"- Previous memory check: solvable={h['solvable']}, "
                        f"reasoning: {h['reasoning'][:200]}"
                    )
                elif htype == "frame_inference" and h.get("observation"):
                    history_lines.append(
                        f"- Visual observation: {h['observation'][:200]} "
                        f"(answer={h.get('answer')}, confidence={h.get('confidence')})"
                    )
            if history_lines:
                history_section = "\n### Previous Reasoning History\n" + "\n".join(history_lines) + "\n"

        batch_note = ""
        if batch_info:
            batch_note = f" (Batch {batch_info[0]}/{batch_info[1]} — you are seeing a subset of the video)"

        prompt = self.prompt_template.format(
            history_section=history_section,
            batch_note=batch_note,
            leaf_context=context,
            question=question,
            options_text=opt_text,
        )

        result = self.llm_fn(prompt, max_tokens=512)

        solvable = result.get("solvable", False)
        answer = result.get("answer", None)
        reasoning = result.get("reasoning", "")
        needs_depth = result.get("needs_depth", False)

        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            answer = m.group(0) if m else None

        return {
            "solvable": bool(solvable),
            "answer": answer,
            "reasoning": reasoning,
            "needs_depth": bool(needs_depth),
        }


class ForcedAnswerFallback:
    """모든 방법이 실패했을 때 강제로 답변을 생성.

    prompt_template 교체 가능.
    """

    DEFAULT_PROMPT = """You MUST answer the following multiple choice question. Pick the best option even if uncertain.

[Memory Context]
{context}

[Question]
{question}

[Options]
{options_text}

Output ONLY valid JSON: {{"answer": "A"}}"""

    def __init__(self, llm_fn, prompt_template: str | None = None):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def force_answer(self, context: str, question: str, options: list[str]) -> str | None:
        """강제 답변 생성.

        Returns:
            letter (A/B/C/D) or None
        """
        opt_text = "\n".join(options)
        prompt = self.prompt_template.format(
            context=context, question=question, options_text=opt_text
        )
        result = self.llm_fn(prompt, max_tokens=20)
        raw = result.get("answer", "") or str(result)
        m = re.search(r"[ABCD]", raw.upper())
        return m.group(0) if m else None
