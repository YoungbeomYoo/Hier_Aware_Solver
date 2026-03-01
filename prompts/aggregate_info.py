"""
Information Aggregation Prompt — Vgent-style Structured Reasoning

Aggregates sub-question verification results across video segments
into a concise summary that aids final answer selection.

Based on Vgent's AGGREGATE_PROMPT.
"""

PROMPT = """You are given a multiple-choice question and a set of sub-questions with their corresponding answers for different video segments.
Your task is to aggregate the information over the video segments and cancel out options from the candidates that are contradicted by the information.
The numbers associated with video segments indicate their order in time, arranged in ascending temporal order. For example, segment 0 happens before segment 1.

Question: {question}
Candidates: {candidates}
Information: {information}

Summarize the information gathered from the video segments within 20 words. Avoid giving or ruling out final answers, since some details may be missing.

Output ONLY valid JSON:
{{"summary": "Your 20-word summary here"}}"""
