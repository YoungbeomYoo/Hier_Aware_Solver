"""Hierarchical Navigation — Single hop prompt (HD-EPIC v4 style).

Selects the single best matching node from Level 1 memory nodes.
"""

PROMPT = """You are a Hierarchical Memory Navigator.
Your goal is to find the most relevant video segment branch based on the search cues.

[Search Cues]
{cues}

[Level 1 Memory Nodes]
{node_summaries}

Select the ID of the ONE node that most likely contains the answer. If multiple, select the best one.
Keep "thought" under 20 words.
Output ONLY valid JSON:
{{
    "thought": "One sentence reason.",
    "selected_id": <int>
}}"""
