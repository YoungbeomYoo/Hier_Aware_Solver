"""
Sub-Question Answer Prompt — Vgent-style Structured Reasoning

Verifies sub-questions against a video segment's caption/description.
Returns yes/no answers for each sub-question.

Based on Vgent's SQL_ANSWER_PROMPT.
"""

PROMPT = """You are given descriptions of a video segment and a list of questions.
Based ONLY on the provided video segment description, answer each question.

Video segment description:
{segment_description}

Questions: {questions}

Generate corresponding answers in JSON format.
The answer must be either "yes" or "no".
Do not provide any additional explanations or responses beyond the required format.

For Example:
Questions: {{
"Q1": "Is there ...",
"Q2": "Does the video show ..."
}}
Your output:
{{
    "Q1": "yes",
    "Q2": "no"
}}

Ensure that each response adheres strictly to the specified answer types.
Output ONLY valid JSON."""


PROMPT_VISUAL = """Given a list of questions related to the video, generate corresponding answers in JSON format.
The answer must be either "yes" or "no".
Do not provide any additional explanations or responses beyond the required format.
Questions: {questions}

For Example:
Questions: {{
"Q1": "Is there ...",
"Q2": "Does the video show ..."
}}
Your output:
{{
    "Q1": "yes",
    "Q2": "no"
}}

Ensure that each response adheres strictly to the specified answer types."""
