"""
History-Aware Visual Observer — 실패 이유 기반 targeted frame observation

기존 VisionVLM과의 차이:
- VisionVLM: "프레임 보고 답 골라라" (generic)
- HistoryAwareObserver: "이전에 X 때문에 답을 못했다. 프레임에서 Y를 집중적으로 관찰하라"

Escalation pattern:
1. Memory context만으로 판단 시도 → 실패
2. 실패 이유(ambiguity_reason) + 관찰 포인트(observation_focus) 추출
3. 해당 시간대 프레임 로딩
4. Observer가 failure-aware prompt로 fine-detail 정보 추출
5. 추출된 observation으로 다시 답변 시도
"""

import gc
import re
import os


class HistoryAwareObserver:
    """실패 이유를 반영한 targeted visual observation.

    Step 1: 왜 못 맞췄는지 분석 (ambiguity_reason)
    Step 2: 프레임에서 뭘 봐야 하는지 결정 (observation_focus)
    Step 3: focus-guided prompt로 VLM에게 프레임 관찰 지시
    Step 4: 관찰 결과를 structured output으로 반환
    """

    ANALYZE_FAILURE_PROMPT = """You tried to answer a video question but couldn't be confident.

### Question
{question}

### Options
{options_text}

### Memory Context (what you already know)
{context}

### Your Previous Attempt
{previous_reasoning}

### Task
Analyze WHY you couldn't answer confidently. Then specify EXACTLY what visual details from the video frames would resolve the ambiguity.

Output ONLY valid JSON:
{{
    "ambiguity_reason": "Specific reason why memory context is insufficient",
    "observation_focus": [
        "Exact visual detail to look for (e.g., 'color of the object in person's right hand')",
        "Second visual detail if needed"
    ],
    "distinguish_between": {{
        "option_a": "What visual evidence would support this option",
        "option_b": "What visual evidence would support this option"
    }},
    "time_priority": "which part of the time range to focus on (start/middle/end/all)"
}}"""

    OBSERVE_FRAMES_PROMPT = """You are a detail-focused visual observer. Your job is NOT to answer the question directly, but to EXTRACT specific visual information from the frames.

### Background
A previous attempt to answer this question using text memory failed because:
**{ambiguity_reason}**

### Observation Instructions
Focus on these specific visual details:
{observation_focus_text}

### What to look for per option:
{distinguish_text}

### Question (for context only)
{question}

### Options (for context only)
{options_text}

### IMPORTANT
Do NOT answer the question. Instead, describe EXACTLY what you see in the frames that is relevant to the observation instructions above. Be specific about:
- Physical appearances, colors, sizes
- Spatial relationships (left/right, above/below)
- Actions and movements
- Text, labels, numbers visible on screen
- Temporal sequence (what happens first, then, finally)

Output ONLY valid JSON:
{{
    "observations": [
        "Detailed observation 1",
        "Detailed observation 2"
    ],
    "key_finding": "The most important visual detail that resolves the ambiguity",
    "evidence_for_options": {{
        "A": "Visual evidence for/against option A",
        "B": "Visual evidence for/against option B",
        "C": "Visual evidence for/against option C",
        "D": "Visual evidence for/against option D"
    }},
    "suggested_answer": "A|B|C|D based on visual evidence",
    "confidence": "high|medium|low"
}}"""

    ANSWER_WITH_OBSERVATION_PROMPT = """Now answer the question using both memory context AND visual observations.

### Memory Context
{context}

### Visual Observations (from actual video frames)
{observations_text}

### Question
{question}

### Options
{options_text}

### Instructions
The visual observations were specifically collected to resolve ambiguity.
Trust the visual evidence over text descriptions when they conflict.

Output ONLY valid JSON:
{{
    "answer": "A|B|C|D",
    "confidence": "high|medium|low",
    "reasoning": "How the visual observations resolved the ambiguity"
}}"""

    def __init__(self, llm_fn, vision_vlm, frame_loader,
                 analyze_prompt: str | None = None,
                 observe_prompt: str | None = None,
                 answer_prompt: str | None = None):
        """
        Args:
            llm_fn: text-only LLM (for failure analysis + final answer)
            vision_vlm: VisionVLM instance (for frame observation)
            frame_loader: TargetedFrameLoader instance
        """
        self.llm_fn = llm_fn
        self.vision_vlm = vision_vlm
        self.frame_loader = frame_loader
        self.analyze_prompt = analyze_prompt or self.ANALYZE_FAILURE_PROMPT
        self.observe_prompt = observe_prompt or self.OBSERVE_FRAMES_PROMPT
        self.answer_prompt = answer_prompt or self.ANSWER_WITH_OBSERVATION_PROMPT

    def observe_and_answer(
        self,
        context: str,
        question: str,
        options: list[str],
        previous_reasoning: str,
        time_ranges: list[tuple[float, float]],
        video_path: str,
        max_frames: int = 30,
        hop_history: list[dict] | None = None,
    ) -> dict:
        """Full escalation: analyze failure → observe frames → answer.

        Args:
            context: memory context text
            question: 질문
            options: 선택지 리스트
            previous_reasoning: 이전 시도의 추론 (왜 실패했는지)
            time_ranges: 관찰할 시간 구간 [(start, end), ...]
            video_path: 비디오 파일 경로
            max_frames: 최대 프레임 수
            hop_history: 이전 단계 히스토리

        Returns:
            {
                "answer": str | None,
                "confidence": str,
                "observations": list[str],
                "key_finding": str,
                "ambiguity_reason": str,
                "observation_focus": list[str],
                "used_visual": bool,
            }
        """
        opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))

        # ========================================
        # Step 1: Analyze why answer failed
        # ========================================
        failure_analysis = self._analyze_failure(
            context, question, opt_text, previous_reasoning,
        )
        ambiguity_reason = failure_analysis.get("ambiguity_reason", "Insufficient context")
        observation_focus = failure_analysis.get("observation_focus", [])
        distinguish = failure_analysis.get("distinguish_between", {})

        print(f"      [Observer] Reason: {ambiguity_reason[:80]}")
        print(f"      [Observer] Focus: {observation_focus[:2]}")

        # ========================================
        # Step 2: Load and observe frames
        # ========================================
        if not video_path or not os.path.exists(video_path) or not time_ranges:
            return {
                "answer": None,
                "confidence": "low",
                "observations": [],
                "key_finding": "",
                "ambiguity_reason": ambiguity_reason,
                "observation_focus": observation_focus,
                "used_visual": False,
            }

        try:
            frames_np, frame_secs = self.frame_loader.load(
                video_path, time_ranges, max_frames=max_frames,
            )
        except Exception as e:
            print(f"      [Observer] Frame load error: {e}")
            return {
                "answer": None, "confidence": "low", "observations": [],
                "key_finding": "", "ambiguity_reason": ambiguity_reason,
                "observation_focus": observation_focus, "used_visual": False,
            }

        if frames_np is None or len(frame_secs) == 0:
            return {
                "answer": None, "confidence": "low", "observations": [],
                "key_finding": "", "ambiguity_reason": ambiguity_reason,
                "observation_focus": observation_focus, "used_visual": False,
            }

        print(f"      [Observer] Loaded {len(frame_secs)} frames, observing...")

        observation_result = self._observe_frames(
            frames_np, question, opt_text,
            ambiguity_reason, observation_focus, distinguish,
            context, hop_history,
        )

        del frames_np
        gc.collect()

        observations = observation_result.get("observations", [])
        key_finding = observation_result.get("key_finding", "")
        evidence_map = observation_result.get("evidence_for_options", {})

        # Quick answer from observer
        observer_answer = observation_result.get("suggested_answer")
        observer_confidence = observation_result.get("confidence", "low")

        if observer_answer and isinstance(observer_answer, str):
            m = re.search(r"[ABCD]", observer_answer.upper())
            observer_answer = m.group(0) if m else None

        # If high confidence from observer → use directly
        if observer_confidence == "high" and observer_answer:
            return {
                "answer": observer_answer,
                "confidence": "high",
                "observations": observations,
                "key_finding": key_finding,
                "ambiguity_reason": ambiguity_reason,
                "observation_focus": observation_focus,
                "used_visual": True,
            }

        # ========================================
        # Step 3: Final answer with observation
        # ========================================
        observations_text = "\n".join(f"- {obs}" for obs in observations)
        if key_finding:
            observations_text += f"\n\nKey finding: {key_finding}"
        if evidence_map:
            observations_text += "\n\nPer-option evidence:"
            for opt_key, evidence in evidence_map.items():
                observations_text += f"\n  {opt_key}: {evidence}"

        final_result = self._answer_with_observation(
            context, question, opt_text, observations_text,
        )

        final_answer = final_result.get("answer")
        if final_answer and isinstance(final_answer, str):
            m = re.search(r"[ABCD]", final_answer.upper())
            final_answer = m.group(0) if m else observer_answer

        return {
            "answer": final_answer or observer_answer,
            "confidence": final_result.get("confidence", observer_confidence),
            "observations": observations,
            "key_finding": key_finding,
            "ambiguity_reason": ambiguity_reason,
            "observation_focus": observation_focus,
            "used_visual": True,
            "reasoning": final_result.get("reasoning", ""),
        }

    # ========== Internal Steps ==========

    def _analyze_failure(self, context: str, question: str,
                         opt_text: str, previous_reasoning: str) -> dict:
        """Step 1: 왜 못 맞췄는지 분석."""
        if not self.llm_fn:
            return {
                "ambiguity_reason": "No LLM available",
                "observation_focus": ["general visual details"],
                "distinguish_between": {},
            }

        prompt = self.analyze_prompt.format(
            question=question,
            options_text=opt_text,
            context=context[:6000],
            previous_reasoning=previous_reasoning or "No previous attempt",
        )
        return self.llm_fn(prompt, max_tokens=400)

    def _observe_frames(self, frames_np, question: str, opt_text: str,
                        ambiguity_reason: str, observation_focus: list,
                        distinguish: dict, context: str,
                        hop_history: list | None) -> dict:
        """Step 2: Focus-guided frame observation."""
        focus_text = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(observation_focus))
        if not focus_text:
            focus_text = "  1. General visual details relevant to the question"

        distinguish_text = ""
        if distinguish:
            for opt_key, desc in distinguish.items():
                distinguish_text += f"  {opt_key}: {desc}\n"
        if not distinguish_text:
            distinguish_text = "  (No specific per-option guidance)"

        # Build observation prompt as memory_context for VisionVLM
        observation_instruction = self.observe_prompt.format(
            ambiguity_reason=ambiguity_reason,
            observation_focus_text=focus_text,
            distinguish_text=distinguish_text,
            question=question,
            options_text=opt_text,
        )

        # Use VisionVLM with the observation instruction as context
        options_list = [line.strip() for line in opt_text.split("\n") if line.strip()]
        result = self.vision_vlm.infer(
            frames_np, observation_instruction, question, options_list,
            hop_history,
        )

        # Parse the VLM output — it might return structured or unstructured
        if isinstance(result, dict):
            return result
        return {"observations": [str(result)], "confidence": "low"}

    def _answer_with_observation(self, context: str, question: str,
                                 opt_text: str, observations_text: str) -> dict:
        """Step 3: observation 결과 반영하여 최종 답변."""
        if not self.llm_fn:
            return {"answer": None, "confidence": "low"}

        prompt = self.answer_prompt.format(
            context=context[:6000],
            observations_text=observations_text,
            question=question,
            options_text=opt_text,
        )
        return self.llm_fn(prompt, max_tokens=300)
