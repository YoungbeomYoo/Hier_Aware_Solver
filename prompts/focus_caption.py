"""Focus Caption — VideoLucy 스타일 상세 VLM captioning prompt.

Two-stage visual의 Stage 2에서 사용.
LLM이 생성한 instruction을 VLM에게 전달하여 타겟팅된 상세 묘사 수행.

Config:
  judge_visual:
    focus_prompt: videolucy_style
"""

PROMPT = """Observe these video frames carefully and provide a detailed and objective description of what is shown.
You should pay special attention to any visible texts, subtitles, text overlays, or voice-overs in the video.
In addition: {instruction}

For reference:
- Question: {question}
- Options: {options_text}

Describe what you see in detail. Do NOT try to answer the question directly — just provide thorough visual descriptions."""
