"""Focus Region Select — LLM이 유망 구간 선택 + VLM instruction 생성.

Two-stage visual의 Scout → Focus 전환 단계에서 사용.
VideoLucy의 get_single_related_time + Instruction 생성 패턴 차용.

Config:
  judge_visual:
    focus_select_prompt: default
"""

PROMPT = """You are selecting the most promising video regions to examine in detail for answering a question.

### Question
{question}

### Options
{options_text}

### Scout Observations (brief captions from each region)
{scout_observations}

### Task
1. Which regions are most likely to contain information needed to answer the question?
2. What specific visual details should we focus on when examining these regions closely?

Output ONLY valid JSON:
{{
    "selected_regions": [0, 2, 5],
    "instruction": "Describe in detail the specific actions and objects. Pay special attention to [what matters for the question]. Note any text, subtitles, or voice-over content visible.",
    "reasoning": "Why these regions were selected"
}}

Select 1-3 regions maximum. Provide a specific, targeted instruction for the visual model."""
