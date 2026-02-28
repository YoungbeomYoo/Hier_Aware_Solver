"""VideoLucy-style prompts for VisualJudge — relaxed confidence, inference allowed."""

JUDGE_VISUAL_PROMPT = """You are a video question answering agent. Use the provided context and your reasoning to determine the best answer.

### Question
{question}

### Options
{options_text}

### Available Context (text-based memory)
{context}
{history_section}
### Task
Analyze the context and determine if you can answer the question.
- If the context provides reasonable evidence (even partial), select the best answer.
- Be willing to infer from available clues rather than requiring explicit statements.
- Only say unanswerable if the context has absolutely no relevant information.

Output ONLY valid JSON:
{{
    "answerable": true or false,
    "answer": "A" or "B" or "C" or "D" or null,
    "confidence": "high" or "medium" or "low",
    "reasoning": "Brief explanation of your answer",
    "needs_visual": true or false,
    "visual_query": "What to look for in video frames" or null,
    "visual_focus": ["specific detail to check"],
    "missing_info": "What information is missing" or null,
    "search_direction": "earlier_time|later_time|same_region_detail|different_topic|broader_context" or null
}}

Confidence guidelines:
- "high": The context clearly supports one answer over others.
- "medium": Partial evidence points to an answer but some uncertainty remains.
- "low": Very little relevant information in the context.

Guidelines for needs_visual:
- TRUE if: visual details (colors, spatial layout, actions, OCR) would help resolve uncertainty
- FALSE if: text context is sufficient to answer confidently"""


REJUDGE_PROMPT = """You are a video question answering agent. You now have BOTH text context AND visual descriptions from actual video frames.

### Question
{question}

### Options
{options_text}

### Text Context (from memory)
{context}

### Visual Observations (from actual video frames)
{visual_captions}
{history_section}
### Task
Using both the text context and visual observations, determine the best answer.
- Consider all available evidence from both sources.
- When text and visual information conflict, give MORE WEIGHT to visual observations.
- Be willing to infer the answer from partial evidence if one option is clearly more likely.

Output ONLY valid JSON:
{{
    "answerable": true or false,
    "answer": "A" or "B" or "C" or "D" or null,
    "confidence": "high" or "medium" or "low",
    "reasoning": "How text + visual evidence together support the answer",
    "visual_helped": true or false,
    "visual_contribution": "What new information the visual frames provided"
}}

Confidence guidelines:
- "high": Combined evidence clearly supports one answer.
- "medium": Evidence favors one answer but alternatives cannot be fully ruled out.
- "low": Neither text nor visual clearly addresses the question."""
