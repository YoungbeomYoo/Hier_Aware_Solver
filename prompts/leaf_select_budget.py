"""Leaf Selection — Budgeted prompt (LVBench v3 style).

Selects N most promising leaves from candidates.
"""

PROMPT = """You are selecting video segments to examine for answering a question.
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
