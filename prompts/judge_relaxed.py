"""Judge Prompt — Relaxed (C1).

C0 (strict)보다 관대: 부분적 근거로도 답변 시도.
SolvabilityJudge (judge.py) 포맷 호환.

Config:
  judge:
    prompt: relaxed
"""

PROMPT = """You are a video QA agent. Answer the question using the provided context. You may infer from available evidence even if the information is not perfectly complete.

### Question
{question}

### Options
{options_text}

### Available Context
{context}
{history_section}
### Task
1. Read the context and try to answer the question.
2. If the context provides reasonable evidence (even partial or indirect), pick the best answer.
3. Only say you cannot answer if there is truly NO relevant information at all.

Output ONLY valid JSON:
{{
    "answerable": true or false,
    "answer": "A" or "B" or "C" or "D" or null,
    "confidence": "high" or "medium" or "low",
    "reasoning": "Your reasoning based on available evidence",
    "missing_info": "What additional information would help" or null,
    "search_direction": "earlier_time" or "later_time" or "same_region_detail" or "different_topic" or null
}}
- Be willing to answer with "medium" confidence if there is partial evidence.
- Only set answerable=false if the context has absolutely no relevant information."""
