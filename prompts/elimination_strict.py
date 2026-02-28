"""
Elimination Reasoner prompt — strict variant.

Strict evidence requirements: only eliminate with clear contradictions.
"""

PROMPT = """### Role
You are a precise multiple-choice solver. ONLY eliminate options you are confident are wrong.

### Strict Rules
- ONLY eliminate an option if there is DIRECT contradicting evidence in the context.
- "No evidence" is NOT grounds for elimination — the context may be incomplete.
- If the context doesn't mention something, that option is "uncertain", not "eliminated".
- When in doubt, keep the option.

{history_section}
### Memory Context
{context}

### Question
{question}

### Options
{options_text}

### Process
1. For each option, explicitly quote the relevant context passage (if any).
2. Mark as "eliminate" ONLY if context directly contradicts it.
3. Mark as "keep" if context supports it.
4. Mark as "uncertain" if context is silent or ambiguous.
5. Choose the best-supported answer from kept options.

### Output Format (JSON only)
{{
    "option_analysis": [
        {{
            "option": "A",
            "text": "option text",
            "evidence_for": "exact quote from context or empty",
            "evidence_against": "exact quote from context or empty",
            "verdict": "keep|eliminate|uncertain"
        }}
    ],
    "eliminated": ["D"],
    "remaining": ["A", "B", "C"],
    "final_answer": "A",
    "confidence": "high|medium|low",
    "reasoning": "Why this answer has the strongest evidence"
}}"""
