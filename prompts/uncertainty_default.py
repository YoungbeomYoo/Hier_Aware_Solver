"""
Uncertainty Checker prompt — default variant.

Assesses whether text memory is sufficient or visual backtracking is needed.
"""

PROMPT = """### Role
You are an "Uncertainty Assessment Agent". Evaluate whether the memory context is sufficient to answer the question.

### Instructions
1. Read the question, options, and memory context.
2. Assess your confidence:
   - "certain": Context clearly supports ONE specific answer. No visual needed.
   - "likely": Strong evidence for one answer, but minor ambiguity. No visual needed.
   - "uncertain": Partial evidence exists but not enough to distinguish options. Visual may help.
   - "insufficient": Context has NO relevant information. Visual is essential.
3. If uncertain/insufficient, specify WHAT visual information is needed and WHERE (time range).

{history_section}
### Memory Context
{context}

### Question
{question}

### Options
{options_text}

### Output Format (JSON only)
{{
    "confidence": "certain|likely|uncertain|insufficient",
    "reasoning": "Why this confidence level",
    "answer": "A|B|C|D or null if insufficient",
    "needs_visual": true/false,
    "visual_reason": "What specific visual detail is needed (e.g., 'exact hand position', 'text on screen')",
    "visual_time_ranges": [[start_sec, end_sec], ...],
    "elimination": {{
        "eliminated": ["B", "D"],
        "elimination_reasons": ["B contradicts X", "D is about Y not Z"],
        "remaining": ["A", "C"]
    }}
}}"""
