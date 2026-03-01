"""
Sub-Question Generator — Vgent-style Structured Reasoning

Breaks down a multiple-choice question into yes/no sub-questions
for per-segment verification.
"""

import json


class SubQuestionGenerator:
    """Generate sub-questions from a multiple-choice question.

    Args:
        llm_fn: callable(prompt, max_tokens) -> dict
        prompt_template: prompt with {question} and {candidates} placeholders
    """

    def __init__(self, llm_fn, prompt_template: str):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template

    def generate(self, question: str, options: list[str]) -> dict:
        """Generate sub-questions.

        Args:
            question: the video question
            options: list of option strings

        Returns:
            dict like {"Q1": "Is there...?", "Q2": "Does...?", ...}
            Empty dict on failure.
        """
        candidates_str = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )
        prompt = self.prompt_template.format(
            question=question,
            candidates=candidates_str,
        )

        try:
            result = self.llm_fn(prompt, max_tokens=512)
        except Exception as e:
            print(f"    [SubQGen] Error: {e}")
            return {}

        if not isinstance(result, dict):
            print(f"    [SubQGen] Non-dict result: {result}")
            return {}

        # Filter to only Q-keyed entries
        sub_questions = {
            k: v for k, v in result.items()
            if isinstance(k, str) and k.startswith("Q") and isinstance(v, str)
        }

        if not sub_questions:
            print(f"    [SubQGen] No sub-questions extracted from: {result}")
            return {}

        print(f"    [SubQGen] Generated {len(sub_questions)} sub-questions")
        return sub_questions
