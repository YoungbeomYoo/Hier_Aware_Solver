"""Solvability Check — VideoLucy forced answer style prompt.

VideoLucy 원본 must_answer_with_coarse_and_fine_memory_prompt 충실 재현.
- 반드시 하나의 최선 답을 선택해야 함 (거부 불가)
- Time Period 참조 포함
- ordinal number 주의사항
- reasoning이 answer를 절대적으로 지지해야 함

Config:
  judge:
    prompt: videolucy_force
"""

PROMPT = """The following provides descriptions of what's shown in the video during different time periods:

{context}
{history_section}
Now, a question has been raised regarding the content descriptions of this video.
{question}

{options_text}

Please read and understand the given video content and question in depth. Strictly based on the video content, select the single best option. You must choose an option from these provided options. The answer you provide must include the English letters of the options [A, B, C, D].

Please note that if an ordinal number appears in the provided question, in most cases, the meaning of this ordinal number is not related to the ordinal of the provided time period. You need to focus on analyzing the meaning of this ordinal number.

Please output ONLY valid JSON in a strictly standardized format:
{{
    "answerable": true,
    "answer": "A" or "B" or "C" or "D",
    "confidence": "high" or "medium" or "low",
    "Time Period": [["start_time", "end_time"], ...],
    "reasoning": "Your reasoning about your judgment. You need to ensure and check that your reasoning must be able to absolutely support your answer.",
    "missing_info": null,
    "search_direction": null
}}
- You MUST provide an answer. Do not refuse.
- "Time Period": Fill in the time periods corresponding to the best answer."""
