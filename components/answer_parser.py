"""
Answer Parser — 모델 출력에서 정답 추출

- extract_choice_letter: A/B/C/D letter 추출 (Video-MME, LVBench)
- parse_response_index: 0-based index 변환 (HD-EPIC)
"""

import re


def extract_choice_letter(text: str) -> str | None:
    """모델 출력에서 A/B/C/D 중 첫 번째 매칭 letter 추출."""
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def parse_response_index(text: str, n_choices: int) -> int:
    """모델 출력에서 letter를 0-based index로 변환.

    Returns:
        0-based index. 파싱 실패 시 -1.
    """
    if not text:
        return -1
    m = re.search(r"[A-Z]", text.strip().upper())
    if m:
        idx = ord(m.group(0)) - ord("A")
        return idx if 0 <= idx < n_choices else -1
    return -1
