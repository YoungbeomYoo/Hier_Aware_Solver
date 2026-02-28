"""Query Decomposition — Detailed prompt (LVBench v3 style).

Adds Specific Persons and Numbers/Statistics as focus areas.
"""

PROMPT = """Analyze the following video question and choices.
Extract 3 to 5 highly specific keywords (cues) that act as search triggers for finding the relevant video segment.
Focus on: Prominent Objects, Specific Persons, Specific Actions, State Changes, Numbers/Statistics.

[Question]
{question}

[Choices]
{choices_str}

Output ONLY valid JSON:
{{
    "cues": ["keyword1", "keyword2", "keyword3"],
    "target_action": "Brief description of what to look for in the video"
}}"""
