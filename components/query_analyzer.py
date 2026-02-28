"""
Query Analyzer — 1단계: 질의 분해 및 시간 범위 획득

인지과학적 근거: SAM (Search of Associative Memory)
- 질문의 '성격'과 '타겟 필드'를 분류
- 명시적 시간 존재 여부 판단 → Track A/B 분기
- 타겟 메타데이터 매핑 (Video-RAG 방식)

기능:
1. 시간 추출 (question + choices 모두에서)
2. 질문 유형 분류 (action / object / attribute / temporal / global)
3. 타겟 key_elements 필드 매핑
4. Cue 추출 (기존 QueryDecomposer 통합)
"""

import re
from components.time_router import TimeRouter


# ============================================================
# Rule-based question type → target field mapping
# ============================================================

# 질문 패턴 → 우선 탐색할 key_elements 필드
QUESTION_TYPE_PATTERNS = {
    "action": {
        "patterns": [
            r"what (?:step|action|is .+ doing|did .+ do|happens|occurred)",
            r"how (?:does|did|is) .+ (?:doing|done|performed|made)",
            r"describe the (?:action|activity|movement)",
            r"what (?:is|was) (?:he|she|they|the participant) (?:doing|performing)",
        ],
        "target_fields": ["actions", "summary"],
        "secondary_fields": ["objects", "persons"],
    },
    "object": {
        "patterns": [
            r"what (?:object|item|thing|tool|ingredient)",
            r"what is (?:on|in|near|beside|under|above)",
            r"identify the",
            r"what (?:is|are) (?:being )?(?:used|held|placed|shown)",
        ],
        "target_fields": ["objects", "summary"],
        "secondary_fields": ["actions", "attributes"],
    },
    "person": {
        "patterns": [
            r"who (?:is|are|was|were)",
            r"how many (?:people|persons|players|participants)",
            r"what (?:is|does) the (?:man|woman|person|player|host|guest)",
        ],
        "target_fields": ["persons", "summary"],
        "secondary_fields": ["actions", "attributes"],
    },
    "attribute": {
        "patterns": [
            r"what (?:color|colour|size|shape|style|type|kind|brand)",
            r"what is .+ wearing",
            r"describe the (?:appearance|look|style)",
        ],
        "target_fields": ["attributes", "objects"],
        "secondary_fields": ["persons", "summary"],
    },
    "text_ocr": {
        "patterns": [
            r"what (?:text|number|score|title|name|word|label|sign)",
            r"what (?:is|was) (?:written|displayed|shown) on",
            r"what (?:does|did) the (?:screen|board|sign|text|subtitle) (?:say|show|read|display)",
            r"read the",
        ],
        "target_fields": ["text_ocr", "summary"],
        "secondary_fields": ["objects", "attributes"],
    },
    "location": {
        "patterns": [
            r"where (?:is|are|was|were|does|did)",
            r"what (?:place|location|room|area|setting|scene|environment)",
            r"in what (?:room|setting|location)",
        ],
        "target_fields": ["locations", "summary"],
        "secondary_fields": ["objects", "attributes"],
    },
    "temporal": {
        "patterns": [
            r"when (?:does|did|is|was)",
            r"(?:before|after|between|during|while) .+(?:what|who|how)",
            r"what (?:order|sequence)",
            r"(?:first|second|third|last|next) (?:thing|step|action|event)",
            r"what is the (?:second|third|next|final) ",
        ],
        "target_fields": ["summary", "actions"],
        "secondary_fields": ["objects", "persons"],
    },
    "global": {
        "patterns": [
            r"what is the (?:overall|main|general|primary|central) ",
            r"summarize|summary|overview|theme|topic|subject|purpose",
            r"what is this video about",
            r"what was not (?:reported|mentioned|discussed)",
        ],
        "target_fields": ["summary"],
        "secondary_fields": ["actions", "objects"],
    },
}


class QueryAnalyzer:
    """1단계: 질문 분석 및 탐색 범위 결정.

    SAM 모델의 '컨텍스트 큐' 역할 — 질문에서 탐색 공간을 좁히는 모든 단서를 추출.

    Args:
        llm_fn: callable(prompt, max_tokens) -> dict (optional, for LLM-based classification)
        prompt_template: LLM 기반 분류 prompt (optional)
    """

    DEFAULT_CLASSIFY_PROMPT = """Analyze the video question below.

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

    def __init__(self, llm_fn=None, prompt_template: str | None = None):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_CLASSIFY_PROMPT
        self._time_router = TimeRouter()

    def analyze(self, question: str, choices: list[str],
                time_reference: str = "") -> dict:
        """질문 전체 분석.

        Returns:
            {
                "question_type": str,
                "target_fields": list[str],
                "secondary_fields": list[str],
                "cues": list[str],
                "target_action": str,
                "has_explicit_time": bool,
                "time_ranges_from_question": list[tuple],
                "time_ranges_from_choices": list[list[tuple]],
                "all_time_ranges": list[tuple],
                "time_reference": tuple | None,
                "requires_visual": bool,
            }
        """
        full_text = question + " " + " ".join(choices)

        # ============================================================
        # 1. 시간 추출 — question에서, choices 개별에서, time_reference에서
        # ============================================================
        q_time_ranges = self._time_router.extract_time_ranges(question)

        # 각 choice별 시간 추출 (선지에 시간이 있으면 전부 긁어옴)
        choice_time_ranges = []
        for c in choices:
            choice_time_ranges.append(self._time_router.extract_time_ranges(c))

        # 모든 시간 합치기
        all_times = list(q_time_ranges)
        for ct in choice_time_ranges:
            all_times.extend(ct)

        # time_reference 파싱
        time_ref_parsed = self._time_router.parse_time_reference(time_reference)

        has_explicit_time = bool(all_times) or bool(time_ref_parsed)

        # ============================================================
        # 2. 질문 유형 분류 + 타겟 필드 매핑
        # ============================================================
        q_type, target_fields, secondary_fields = self._classify_question_type(question)

        # ============================================================
        # 3. Cue 추출
        # ============================================================
        cues = []
        target_action = ""
        requires_visual = False

        if self.llm_fn:
            # LLM 기반 정밀 분류
            result = self._llm_classify(question, choices)
            if result.get("cues"):
                cues = result["cues"]
            if result.get("target_action"):
                target_action = result["target_action"]
            if result.get("question_type"):
                q_type = result["question_type"]
            if result.get("target_fields"):
                target_fields = result["target_fields"]
            requires_visual = result.get("requires_visual", False)

        if not cues:
            cues = self._extract_cues_rule_based(question, choices)

        return {
            "question_type": q_type,
            "target_fields": target_fields,
            "secondary_fields": secondary_fields,
            "cues": cues,
            "target_action": target_action,
            "has_explicit_time": has_explicit_time,
            "time_ranges_from_question": q_time_ranges,
            "time_ranges_from_choices": choice_time_ranges,
            "all_time_ranges": all_times,
            "time_reference": time_ref_parsed,
            "requires_visual": requires_visual,
        }

    def _classify_question_type(self, question: str) -> tuple[str, list[str], list[str]]:
        """Rule-based question type classification (LLM 비용 0)."""
        q_lower = question.lower().strip()
        for q_type, config in QUESTION_TYPE_PATTERNS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, q_lower):
                    return q_type, config["target_fields"], config["secondary_fields"]
        return "global", ["summary"], ["actions", "objects"]

    def _llm_classify(self, question: str, choices: list[str]) -> dict:
        """LLM 기반 정밀 분류."""
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = self.prompt_template.format(
            question=question, choices_str=choices_str
        )
        return self.llm_fn(prompt, max_tokens=200)

    def _extract_cues_rule_based(self, question: str, choices: list[str]) -> list[str]:
        """Rule-based cue extraction (fallback)."""
        # Named entities, capitalized words, long words
        text = question + " " + " ".join(choices[:2])
        words = re.findall(r"\b[A-Z][a-z]{2,}\b|\b\w{5,}\b", text)
        # Remove common stop-like words
        stop = {"which", "these", "those", "about", "video", "following",
                "question", "answer", "option", "choose", "select",
                "between", "during", "before", "after", "shown"}
        cues = [w for w in words if w.lower() not in stop]
        return list(dict.fromkeys(cues))[:5]  # unique, max 5
