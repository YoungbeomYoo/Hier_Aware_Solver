"""
Query Classification prompt — default variant.

Used by QueryAnalyzer for LLM-based question type classification.
"""

PROMPT = """Analyze the video question below.

1. Classify the question type from: action, object, person, attribute, text_ocr, location, temporal, global
2. Extract 3-5 specific search keywords (cues) from the question AND choices.
3. Identify which metadata fields are most relevant:
   - "actions": physical activities, movements, steps
   - "objects": items, tools, ingredients, things
   - "persons": people, characters, participants
   - "attributes": colors, sizes, appearances, styles
   - "text_ocr": on-screen text, scores, labels, signs
   - "locations": places, rooms, settings
   - "summary": overall descriptions

[Question]
{question}

[Choices]
{choices_str}

Output ONLY valid JSON:
{{
    "question_type": "action",
    "cues": ["keyword1", "keyword2", "keyword3"],
    "target_fields": ["actions", "summary"],
    "target_action": "Brief description of what to find",
    "requires_visual": false
}}
- "requires_visual": true if fine-grained visual details (exact positions, precise movements, bounding boxes) are likely needed."""
