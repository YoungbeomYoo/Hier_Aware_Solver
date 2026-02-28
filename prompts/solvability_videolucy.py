"""Solvability Check — VideoLucy style prompt.

VideoLucy 원본 answer_with_coarse_memory_prompt 충실 재현.
- Confidence: True/False (binary) — 절대적 확신 있을 때만 True
- Time Period: 답변 근거가 되는 시간 구간 명시
- 보수적 추론: "임의 추론 금지", "절대적 객관적 근거" 요구
- ordinal number 주의사항
- 부족하면 답하지 않고 이유만 반환

Config:
  judge:
    prompt: videolucy
"""

PROMPT = """The following provides descriptions of what's shown in the video during different time periods:

{context}

Note that since these descriptions are not very complete and detailed, some key information in the video segments of each time period may not all appear in these content descriptions.

Now, a question has been raised regarding the content descriptions of this video.
{question}

{options_text}
{history_section}
Please read the given video content descriptions and the question in depth, and determine whether you can accurately answer the given question solely based on the currently provided descriptions.

If you can answer it with absolute confidence, please answer this question and provide the time periods you are referring to. The answer you provide must have completely and absolutely objective support in the video descriptions. Do not make inferences arbitrarily.

Please note that there is only one option that can answer this question. The answer you provide must include the English letters of the options [A, B, C, D].

If you think the current content descriptions of the video is still insufficient to accurately answer the question, please do not answer it and give me your reason.

You must note that if an ordinal number appears in the provided question, in the vast majority of cases, you should not simply assume that this ordinal number represents the ordinal of the provided time period. You need to focus on understanding the specific meaning of this ordinal number within the question based on all the content descriptions.

Please output ONLY valid JSON in a strictly standardized format containing the following key-value pairs:
{{
    "Confidence": true or false,
    "answerable": true or false,
    "answer": "A" or "B" or "C" or "D" or null,
    "confidence": "high" or "low",
    "Time Period": [["start_time", "end_time"], ...] or null,
    "reasoning": "Your reasoning about your judgment. You need to ensure and check that your reasoning must be able to absolutely support your answer.",
    "missing_info": "What specific information is still needed to answer" or null,
    "search_direction": "earlier_time" or "later_time" or "same_region_detail" or "different_topic" or null
}}
- "Confidence": Set to true ONLY if you are certain about the answer. Set to false if not.
- "answerable": Same as Confidence. true if you can answer, false if not.
- "answer": When Confidence is true, fill in the answer (A/B/C/D). When Confidence is false, set to null.
- "confidence": "high" when Confidence is true, "low" when false.
- "Time Period": When Confidence is true, fill in the time periods you are referring to. When false, set to null.
- "reasoning": Show your reasoning. Ensure your reasoning absolutely supports your answer. Do NOT make unfounded inferences without evidence.
- Do NOT guess. If the descriptions are insufficient, set Confidence to false and explain what is missing."""
