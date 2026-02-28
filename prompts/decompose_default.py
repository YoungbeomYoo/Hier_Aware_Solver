"""Query Decomposition — Default prompt (HD-EPIC v4 style)."""

PROMPT = """Analyze the following video question and choices.
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
