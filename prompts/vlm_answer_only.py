"""VLM Inference — Answer-only prompt (simple, HD-EPIC style).

Outputs just the answer letter, no confidence or observation.
"""

PROMPT = """Based on the video frames and memory context, answer the following multiple-choice question.
Respond with only the letter (A, B, C, or D) of the correct option.

{memory_context}

Question: {question}
Options:
{options_text}

The best answer is:"""
