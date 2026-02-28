"""VLM Inference — With confidence prompt (LVBench agentic style).

Outputs answer + confidence level + observation for multi-hop reasoning.
"""

PROMPT = """{history_text}The following {n_frames} frames are from the relevant video segments:

[Video Memory Context]
{memory_context}

Question: {question}
Options:
{options_text}

Select the best answer. Output ONLY valid JSON:
{{"answer": "B", "confidence": "high", "observation": "Key visual details that support the answer"}}
- "high": Clear visual/textual evidence supports your answer.
- "low": Uncertain, describe what additional information would help."""
