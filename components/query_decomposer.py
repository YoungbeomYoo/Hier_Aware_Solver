"""
Query Decomposer — 질문 분해 + Cue 추출

LLM을 사용하여 질문에서 검색용 키워드(cues)를 추출.
prompt_template을 교체하여 다른 전략 적용 가능.
"""

import re


class QueryDecomposer:
    """질문을 분해하고 검색 cue를 추출하는 컴포넌트.

    Args:
        llm_fn: callable(prompt_text, max_tokens) -> dict (TextOnlyLLM.reason)
        prompt_template: str with {question}, {choices_str} placeholders
    """

    DEFAULT_PROMPT = """Analyze the following video question and choices.
Extract 3 to 5 highly specific keywords (cues) that act as search triggers.
Focus on 'Prominent Objects', 'Specific Actions', and 'State Changes'.

[Question]
{question}

[Choices]
{choices_str}

Output ONLY valid JSON:
{{
    "cues": ["keyword1", "keyword2", "keyword3"],
    "target_action": "Brief description of the action to find"
}}"""

    def __init__(self, llm_fn, prompt_template: str | None = None):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def decompose(self, question: str, choices: list[str]) -> dict:
        """질문 분해 → cue 추출.

        Args:
            question: 질문 텍스트
            choices: 선택지 리스트

        Returns:
            {"cues": list[str], "target_action": str}
        """
        choices_str = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
        prompt = self.prompt_template.format(
            question=question, choices_str=choices_str
        )
        result = self.llm_fn(prompt, max_tokens=150)

        cues = result.get("cues", [])
        if not cues:
            # Fallback: extract capitalized/long words from question
            words = re.findall(r"\b[A-Z][a-z]+\b|\b\w{4,}\b", question)
            cues = list(set(words))[:5]

        return {
            "cues": cues,
            "target_action": result.get("target_action", ""),
        }
