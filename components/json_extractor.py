"""
JSON Extractor — LLM 출력에서 JSON 파싱

depth-tracking bracket matching + fallback regex extraction.
세 솔버(HD-EPIC v4, LVBench v3, Video-MME) 모두 동일 로직 통합.
"""

import json
import re


def extract_json(response: str) -> dict:
    """LLM 응답 문자열에서 첫 번째 JSON object를 추출.

    1차: bracket depth tracking으로 완전한 JSON 추출
    2차: 잘린 JSON일 경우 regex fallback으로 key-value 추출

    Args:
        response: LLM raw response string

    Returns:
        Parsed dict. 파싱 실패 시 빈 dict.
    """
    start = response.find("{")
    if start == -1:
        return {}

    # Phase 1: depth-tracking bracket matching
    depth, in_string, escape = 0, False, False
    end = -1
    for i, c in enumerate(response[start:], start):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        try:
            return json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Phase 2: regex fallback — key-value extraction
    result = {}
    fragment = response[start:]

    # string values
    for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', fragment):
        result[m.group(1)] = m.group(2)

    # integer values
    for m in re.finditer(r'"(\w+)"\s*:\s*(-?\d+)', fragment):
        if m.group(1) not in result:
            result[m.group(1)] = int(m.group(2))

    # array values
    for m in re.finditer(r'"(\w+)"\s*:\s*\[([^\]]*)\]', fragment):
        key = m.group(1)
        if key not in result:
            items = m.group(2).strip()
            if items:
                try:
                    result[key] = json.loads(f"[{items}]")
                except json.JSONDecodeError:
                    result[key] = [x.strip().strip('"') for x in items.split(",")]

    # boolean values
    for m in re.finditer(r'"(\w+)"\s*:\s*(true|false)', fragment):
        key = m.group(1)
        if key not in result:
            result[key] = m.group(2) == "true"

    return result
