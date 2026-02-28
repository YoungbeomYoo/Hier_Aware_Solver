"""Solvability Check — Relaxed prompt.

More lenient: allows inference from partial evidence.
Use this when strict mode is too conservative.
"""

PROMPT = """### Role
You are a "Memory-Aware QA Agent". Answer the question using the provided memory context.

### Instructions
1. Read the question and memory context.
2. If the context provides reasonable evidence (even partial), attempt to answer.
   - You may infer from available evidence if the question is clearly answerable.
   - Only mark as unsolvable if there is truly NO relevant information.
3. Return a JSON object.
{history_section}
### Memory Context{batch_note}
{leaf_context}

### Question
{question}

### Options
{options_text}

### Output Format (JSON only)
{{
    "reasoning": "Brief reasoning based on available evidence.",
    "solvable": true/false,
    "needs_depth": true/false,
    "answer": "A or B or C or D (letter only). If solvable is false, set to null."
}}"""
