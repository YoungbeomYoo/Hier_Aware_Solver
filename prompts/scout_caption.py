"""Scout Caption — 구간별 간단 VLM captioning prompt.

Two-stage visual의 Stage 1에서 사용.
각 구간에서 3장씩 프레임을 뽑아 간단하게 묘사.

Config:
  judge_visual:
    scout_prompt: default
"""

PROMPT = """Briefly describe what you see in these video frames.
Focus on: actions happening, objects visible, persons and their appearances, any text/subtitles on screen.
Keep it concise (2-3 sentences). Be specific and factual.

For reference, the question being investigated is: {question}"""
