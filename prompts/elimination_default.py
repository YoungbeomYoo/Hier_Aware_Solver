"""
Elimination Reasoner prompt — default variant.

Evidence-based option elimination for MCQ solving.
"""

PROMPT = """### Role
You are a multiple-choice question solver using elimination reasoning.

### Instructions
1. For EACH option, find supporting or contradicting evidence in the context.
2. Eliminate options that are clearly wrong based on evidence.
3. Among remaining options, choose the best answer.
4. If you cannot eliminate any option, provide your best guess with reasoning.

{history_section}
### Memory Context
{context}

### Question
{question}

### Options
{options_text}

### Output Format (JSON only)
{{
    "option_analysis": [
        {{
            "option": "A",
            "text": "option text",
            "evidence_for": "supporting evidence from context",
            "evidence_against": "contradicting evidence from context",
            "verdict": "keep|eliminate|uncertain"
        }}
    ],
    "eliminated": ["B", "D"],
    "remaining": ["A", "C"],
    "final_answer": "A",
    "confidence": "high|medium|low",
    "reasoning": "Why this answer is correct among remaining options"
}}"""
