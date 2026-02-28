"""
Query Classification prompt — detailed variant.

More thorough analysis with temporal reasoning and conditional search detection.
"""

PROMPT = """You are a question analysis expert for video QA. Analyze the question below with maximum precision.

### Analysis Tasks
1. **Question Type**: Classify from: action, object, person, attribute, text_ocr, location, temporal, global
2. **Search Cues**: Extract 3-7 search keywords from BOTH question and choices.
3. **Target Fields**: Which key_elements fields to search (actions, objects, persons, attributes, text_ocr, locations, summary)
4. **Temporal Pattern**: Does the question ask about ordering, sequence, or Nth event?
5. **Conditional Pattern**: Does the question reference a condition (e.g., "when X happens, what does Y do?")?
6. **Visual Necessity**: Would raw video frames be required, or is text memory sufficient?

[Question]
{question}

[Choices]
{choices_str}

Output ONLY valid JSON:
{{
    "question_type": "action",
    "cues": ["keyword1", "keyword2", "keyword3"],
    "target_fields": ["actions", "summary"],
    "secondary_fields": ["objects"],
    "target_action": "Brief description of what to find",
    "temporal_pattern": null,
    "conditional_pattern": null,
    "requires_visual": false,
    "reasoning": "Brief explanation of classification"
}}

Field descriptions:
- "temporal_pattern": null, or {{"type": "nth", "n": 2}} for "second thing", or {{"type": "sequence"}} for ordering
- "conditional_pattern": null, or {{"condition_field": "persons", "condition_value": "blue shirt", "extract_field": "actions"}}
- "requires_visual": true ONLY if text descriptions cannot capture the needed detail (spatial layout, precise motion, color)"""
