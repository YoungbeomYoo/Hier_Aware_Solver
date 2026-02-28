from __future__ import annotations

"""
Visual-Aware Judge — Judge가 visual 필요 여부를 판단 + query-aware frame captioning

기존 SolvabilityJudge와의 차이:
- SolvabilityJudge: text context만 보고 answerable/not 판단
- VisualJudge: 판단 시 "이 질문은 visual 확인이 필요하다" + "프레임에서 뭘 봐야 하는지" 출력
  → 필요 시 query-aware captioning 수행 → caption을 context에 추가 → 재판단

Flow:
1. judge_with_visual_need(): text context 기반 판단 + needs_visual + visual_query 출력
2. caption_frames(): visual_query 기반 targeted frame captioning (VLM)
3. rejudge(): caption-enriched context로 최종 판단

Pipeline에서의 사용 예시:
    verdict = judge_visual.judge_with_visual_need(context, question, options, history)
    if verdict["needs_visual"] and video_path:
        captions = judge_visual.caption_frames(
            verdict["visual_query"], video_path, time_ranges, max_frames,
        )
        verdict = judge_visual.rejudge(context, captions, question, options, history)
"""

import gc
import os
import re
from components.token_utils import TokenBudget


class VisualJudge:
    """Judge that decides if visual information is needed + query-aware captioning.

    Combines:
    - Solvability judgement (can we answer from text?)
    - Visual need assessment (do we need frames? what to look for?)
    - Query-aware frame captioning (focused captions guided by visual_query)
    - Re-judgement with enriched context

    Args:
        llm_fn: text-only LLM callable(prompt, max_tokens) -> dict
        vision_vlm: VisionVLM instance for frame captioning
        frame_loader: TargetedFrameLoader instance
        judge_prompt: override for judge + visual need assessment prompt
        caption_prompt: override for query-aware captioning prompt
        rejudge_prompt: override for re-judgement prompt
    """

    # ======================================================================
    # Prompt 1: Judge + Visual Need Assessment
    # ======================================================================

    # # v1 judge visual prompt
    # JUDGE_VISUAL_PROMPT = """You are a solvability judge for video question answering. Determine whether the given text context is sufficient to answer the question. Also decide whether visual information (video frames) would help.
    #
    # ### Question
    # {question}
    #
    # ### Options
    # {options_text}
    #
    # ### Available Context (text-based memory)
    # {context}
    # {history_section}
    # ### Task
    # 1. Can you confidently answer from the text context alone?
    # 2. Would seeing actual video frames help resolve any ambiguity?
    # 3. If visual info would help, specify EXACTLY what to look for in the frames.
    #
    # Output ONLY valid JSON:
    # {{
    #     "answerable": true or false,
    #     "answer": "A" or "B" or "C" or "D" or null,
    #     "confidence": "high" or "medium" or "low",
    #     "reasoning": "Why you can or cannot answer from text context",
    #     "needs_visual": true or false,
    #     "visual_query": "Specific description of what to look for in video frames" or null,
    #     "visual_focus": [
    #         "Exact visual detail 1 (e.g., 'color of the object held in right hand')",
    #         "Exact visual detail 2 (e.g., 'whether person turns left or right at the intersection')"
    #     ],
    #     "missing_info": "What information is missing" or null,
    #     "search_direction": "earlier_time|later_time|same_region_detail|different_topic|broader_context" or null
    # }}
    #
    # Guidelines for needs_visual:
    # - TRUE if: spatial details, colors, physical appearances, motion direction, OCR text, scene layout, counting objects, verifying actions not clearly described in captions
    # - FALSE if: the text captions/summaries already contain specific enough detail, the question is about narrative/plot/dialogue, temporal ordering is clear from text"""

    # v2 judge visual prompt — stricter confidence, evidence-based
    JUDGE_VISUAL_PROMPT = """You are a STRICT solvability judge for video question answering. You must provide concrete evidence from the context for any answer you give.

### Question
{question}

### Options
{options_text}

### Available Context (text-based memory)
{context}
{history_section}
### Task
1. Search the context for SPECIFIC evidence that directly supports or eliminates each option.
2. Quote or cite the exact segment (with timestamp) where you found the evidence.
3. If you cannot find a direct quote/reference for your answer, you are NOT confident.

Output ONLY valid JSON:
{{
    "answerable": true or false,
    "answer": "A" or "B" or "C" or "D" or null,
    "confidence": "high" or "medium" or "low",
    "evidence": "Direct quote or specific reference from context that supports your answer (include timestamp)",
    "reasoning": "Step-by-step explanation of how the evidence leads to the answer",
    "eliminates": {{"B": "reason B is wrong", "C": "reason C is wrong"}},
    "needs_visual": true or false,
    "visual_query": "Specific description of what to look for in video frames" or null,
    "visual_focus": [
        "Exact visual detail 1",
        "Exact visual detail 2"
    ],
    "missing_info": "What information is missing" or null,
    "search_direction": "earlier_time|later_time|same_region_detail|different_topic|broader_context" or null
}}

### STRICT Confidence Rules (follow these exactly):
- "high": You found an EXPLICIT statement or description in the context that directly answers the question. You can quote the exact words. All other options can be clearly eliminated.
- "medium": The context strongly implies an answer but does not state it explicitly. OR you can support your answer but cannot eliminate all alternatives.
- "low": You are guessing based on partial information. The context does not clearly address the question.

IMPORTANT: If the context only contains general descriptions without specific details matching the question, confidence MUST be "low" or "medium", NOT "high". Do NOT infer answers from vague or tangential information.

### Guidelines for needs_visual:
- TRUE if: spatial details, colors, physical appearances, motion direction, OCR text, counting, verifying actions, specific visual sequences, or when confidence is "low"/"medium" and frames might help
- FALSE if: text already contains explicit specific detail that directly answers the question"""

    # ======================================================================
    # Prompt 2: Query-Aware Frame Captioning
    # ======================================================================
    CAPTION_PROMPT = """You are a query-aware visual captioner. Your job is to describe the video frames with SPECIFIC FOCUS on the aspects requested below.

### Visual Query
{visual_query}

### Focus Points
{visual_focus_text}

### Question Context (for reference)
{question}

### Options Context (for reference)
{options_text}

### Instructions
Describe what you see in the frames, focusing SPECIFICALLY on the visual query and focus points above.
Do NOT try to answer the question. Just provide detailed, factual visual descriptions.

Be specific about:
- Physical appearances (colors, sizes, shapes, materials)
- Spatial relationships (positions, distances, arrangements)
- Actions and movements (direction, speed, manner)
- Text/labels/numbers visible on screen
- Scene details (environment, lighting, objects present)
- Temporal sequence (what happens across frames)

Output ONLY valid JSON:
{{
    "frame_captions": [
        "Detailed caption for what is observed in the frames, focused on the query"
    ],
    "query_relevant_details": [
        "Specific detail directly relevant to the visual query",
        "Another relevant detail"
    ],
    "key_observation": "The single most important visual finding related to the query",
    "scene_description": "Brief overall scene description for context"
}}"""

    # ======================================================================
    # Prompt 3: Re-Judge with Visual Captions
    # ======================================================================

    # # v1 rejudge prompt
    # REJUDGE_PROMPT = """You are a solvability judge for video question answering. You now have BOTH text context AND visual descriptions from actual video frames. Make your final judgement.
    #
    # ### Question
    # {question}
    #
    # ### Options
    # {options_text}
    #
    # ### Text Context (from memory)
    # {context}
    #
    # ### Visual Observations (from actual video frames)
    # {visual_captions}
    # {history_section}
    # ### Task
    # Using both the text context and visual observations, determine the answer.
    # When text and visual information conflict, give MORE WEIGHT to visual observations (they come from actual frames).
    #
    # Output ONLY valid JSON:
    # {{
    #     "answerable": true or false,
    #     "answer": "A" or "B" or "C" or "D" or null,
    #     "confidence": "high" or "medium" or "low",
    #     "reasoning": "How text + visual evidence together support the answer",
    #     "visual_helped": true or false,
    #     "visual_contribution": "What the visual info added that text alone didn't provide"
    # }}"""

    # v2 rejudge prompt — evidence-based, strict confidence
    REJUDGE_PROMPT = """You are a STRICT solvability judge for video question answering. You now have BOTH text context AND visual descriptions from actual video frames.

### Question
{question}

### Options
{options_text}

### Text Context (from memory)
{context}

### Visual Observations (from actual video frames)
{visual_captions}
{history_section}
### Task
1. For each option, check if either text or visual evidence supports or contradicts it.
2. Cite specific evidence (quote text segments or visual observations) for your answer.
3. When text and visual conflict, give MORE WEIGHT to visual observations (they come from actual frames).

Output ONLY valid JSON:
{{
    "answerable": true or false,
    "answer": "A" or "B" or "C" or "D" or null,
    "confidence": "high" or "medium" or "low",
    "evidence": "Direct quote or observation that supports the answer",
    "reasoning": "Step-by-step: how text + visual evidence together lead to this answer",
    "eliminates": {{"B": "evidence against B", "C": "evidence against C"}},
    "visual_helped": true or false,
    "visual_contribution": "What new information the visual frames provided that text lacked"
}}

### STRICT Confidence Rules (same as initial judge):
- "high": EXPLICIT evidence (quoted text or observed visual detail) directly answers the question AND all other options are eliminated with evidence.
- "medium": Strong evidence for the answer but cannot eliminate all alternatives, OR evidence is indirect/implied.
- "low": Guessing from partial evidence. Neither text nor visual clearly addresses the question."""

    def __init__(
        self,
        llm_fn=None,
        vision_vlm=None,
        frame_loader=None,
        judge_prompt: str | None = None,
        caption_prompt: str | None = None,
        rejudge_prompt: str | None = None,
        token_budget: TokenBudget | None = None,
        context_budget: int = 20000,
        history_budget: int = 4000,
        caption_budget: int = 3000,
        skip_captioning: bool = False,
        force_visual: bool = False,
    ):
        self.llm_fn = llm_fn
        self.vision_vlm = vision_vlm
        self.frame_loader = frame_loader
        self.judge_prompt = judge_prompt or self.JUDGE_VISUAL_PROMPT
        self.caption_prompt = caption_prompt or self.CAPTION_PROMPT
        self.rejudge_prompt = rejudge_prompt or self.REJUDGE_PROMPT
        self.tb = token_budget or TokenBudget(None)
        self.context_budget = context_budget
        self.history_budget = history_budget
        self.caption_budget = caption_budget
        self.skip_captioning = skip_captioning
        self.force_visual = force_visual

    # ==================================================================
    # Step 1: Judge + Visual Need Assessment
    # ==================================================================
    def judge_with_visual_need(
        self,
        context: str,
        question: str,
        options: list[str],
        history_compact: str | None = None,
    ) -> dict:
        """Judge solvability and determine if visual info is needed.

        Returns:
            {
                "answerable": bool,
                "answer": str | None,
                "confidence": str,
                "reasoning": str,
                "needs_visual": bool,
                "visual_query": str | None,
                "visual_focus": list[str],
                "missing_info": str | None,
                "search_direction": str | None,
            }
        """
        default = {
            "answerable": False, "answer": None, "confidence": "low",
            "reasoning": "No LLM available",
            "needs_visual": False, "visual_query": None, "visual_focus": [],
            "missing_info": "LLM required", "search_direction": None,
        }

        if not self.llm_fn:
            return default

        opt_text = "\n".join(
            f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        )

        history_section = ""
        if history_compact:
            history_section = (
                "\n### Search History\n"
                + self.tb.truncate(history_compact, self.history_budget)
            )

        prompt = self.judge_prompt.format(
            question=question,
            options_text=opt_text,
            context=self.tb.truncate(context, self.context_budget),
            history_section=history_section,
        )

        try:
            result = self.llm_fn(prompt, max_tokens=500)
        except Exception as e:
            print(f"      [VisualJudge] Error: {e}")
            return default

        if not isinstance(result, dict):
            return {**default, "reasoning": str(result)[:500]}

        # Normalize fields
        result = self._normalize_judge_output(result)
        return result

    # ==================================================================
    # Step 2: Query-Aware Frame Captioning
    # ==================================================================
    def caption_frames(
        self,
        visual_query: str,
        video_path: str,
        time_ranges: list[tuple[float, float]],
        question: str = "",
        options: list[str] | None = None,
        visual_focus: list[str] | None = None,
        max_frames: int = 30,
    ) -> dict:
        """Load frames and generate query-aware captions.

        Args:
            visual_query: what to look for (from judge output)
            video_path: path to video file
            time_ranges: [(start, end), ...] to sample frames from
            question: question text for reference
            options: options list for reference
            visual_focus: specific focus points from judge
            max_frames: max frames to load

        Returns:
            {
                "frame_captions": list[str],
                "query_relevant_details": list[str],
                "key_observation": str,
                "scene_description": str,
                "caption_text": str,  # formatted text for context injection
                "n_frames": int,
                "success": bool,
            }
        """
        empty = {
            "frame_captions": [], "query_relevant_details": [],
            "key_observation": "", "scene_description": "",
            "caption_text": "", "n_frames": 0, "success": False,
        }

        if not self.vision_vlm or not self.frame_loader:
            print("      [VisualJudge] No vision_vlm or frame_loader")
            return empty

        if not video_path or not os.path.exists(video_path):
            print(f"      [VisualJudge] Video not found: {video_path}")
            return empty

        if not time_ranges:
            print("      [VisualJudge] No time ranges")
            return empty

        # Load frames
        try:
            frames_np, frame_secs = self.frame_loader.load(
                video_path, time_ranges, max_frames=max_frames,
            )
        except Exception as e:
            print(f"      [VisualJudge] Frame load error: {e}")
            return empty

        if frames_np is None or len(frame_secs) == 0:
            return empty

        print(f"      [VisualJudge] Loaded {len(frame_secs)} frames for captioning")

        # Build focus text
        focus_list = visual_focus or []
        focus_text = "\n".join(
            f"  {i + 1}. {f}" for i, f in enumerate(focus_list)
        )
        if not focus_text:
            focus_text = f"  1. {visual_query}"

        opt_text = ""
        if options:
            opt_text = "\n".join(
                f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
            )

        # Build captioning prompt as memory_context for VisionVLM
        caption_instruction = self.caption_prompt.format(
            visual_query=visual_query,
            visual_focus_text=focus_text,
            question=question,
            options_text=opt_text,
        )

        # Call VLM
        try:
            options_list = [o.strip() for o in opt_text.split("\n") if o.strip()] if opt_text else []
            result = self.vision_vlm.infer(
                frames_np, caption_instruction, question, options_list,
            )
        except Exception as e:
            print(f"      [VisualJudge] VLM captioning error: {e}")
            result = {}
        finally:
            del frames_np
            gc.collect()

        if not isinstance(result, dict):
            result = {"frame_captions": [str(result)]}

        # Format caption_text for context injection
        caption_text = self._format_caption_text(result, visual_query)

        return {
            "frame_captions": result.get("frame_captions", []),
            "query_relevant_details": result.get("query_relevant_details", []),
            "key_observation": result.get("key_observation", ""),
            "scene_description": result.get("scene_description", ""),
            "caption_text": caption_text,
            "n_frames": len(frame_secs),
            "success": True,
        }

    # ==================================================================
    # Step 3: Re-Judge with Visual Captions
    # ==================================================================
    def rejudge(
        self,
        context: str,
        captions: dict,
        question: str,
        options: list[str],
        history_compact: str | None = None,
    ) -> dict:
        """Re-judge with visual caption-enriched context.

        Args:
            context: original text context
            captions: output from caption_frames()
            question: question text
            options: options list
            history_compact: search history text

        Returns:
            Same as judge_with_visual_need() + visual_helped, visual_contribution
        """
        default = {
            "answerable": False, "answer": None, "confidence": "low",
            "reasoning": "No LLM available",
            "needs_visual": False, "visual_query": None, "visual_focus": [],
            "missing_info": None, "search_direction": None,
            "visual_helped": False, "visual_contribution": "",
        }

        if not self.llm_fn:
            return default

        opt_text = "\n".join(
            f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        )

        caption_text = captions.get("caption_text", "")
        if not caption_text:
            caption_text = "(No visual captions available)"

        history_section = ""
        if history_compact:
            history_section = (
                "\n### Search History\n"
                + self.tb.truncate(history_compact, self.history_budget)
            )

        prompt = self.rejudge_prompt.format(
            question=question,
            options_text=opt_text,
            context=self.tb.truncate(context, self.context_budget),
            visual_captions=self.tb.truncate(caption_text, self.caption_budget),
            history_section=history_section,
        )

        try:
            result = self.llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"      [VisualJudge] Rejudge error: {e}")
            return default

        if not isinstance(result, dict):
            return {**default, "reasoning": str(result)[:500]}

        # Normalize
        result = self._normalize_judge_output(result)
        result.setdefault("visual_helped", False)
        result.setdefault("visual_contribution", "")

        return result

    # ==================================================================
    # Convenience: Full pipeline (judge → caption → rejudge)
    # ==================================================================
    def judge_full(
        self,
        context: str,
        question: str,
        options: list[str],
        video_path: str | None = None,
        time_ranges: list[tuple[float, float]] | None = None,
        history_compact: str | None = None,
        max_frames: int = 30,
    ) -> dict:
        """Full flow: judge → if needs_visual → caption → rejudge.

        Convenience method that runs the complete pipeline.
        Returns the final verdict with visual enrichment if applicable.
        """
        # Step 1: Initial judge
        verdict = self.judge_with_visual_need(
            context, question, options, history_compact,
        )

        print(f"      [VisualJudge] answerable={verdict.get('answerable')} | "
              f"conf={verdict.get('confidence')} | "
              f"needs_visual={verdict.get('needs_visual')}")

        # If high confidence and answerable → done (unless force_visual)
        if (not self.force_visual
                and verdict.get("answerable")
                and verdict.get("confidence") == "high"
                and verdict.get("answer")):
            verdict["used_visual"] = False
            return verdict

        # If needs visual → caption + rejudge (unless skip_captioning)
        if self.skip_captioning:
            verdict["used_visual"] = False
            verdict["skipped_captioning"] = True
            return verdict

        if (self.force_visual or verdict.get("needs_visual")) and video_path and time_ranges:
            visual_query = verdict.get("visual_query", "")
            visual_focus = verdict.get("visual_focus", [])

            # force_visual: generate fallback query if judge didn't produce one
            if not visual_query and self.force_visual:
                visual_query = f"Describe what you see related to: {question}"

            if visual_query:
                print(f"      [VisualJudge] Captioning: {visual_query[:80]}")

                captions = self.caption_frames(
                    visual_query=visual_query,
                    video_path=video_path,
                    time_ranges=time_ranges,
                    question=question,
                    options=options,
                    visual_focus=visual_focus,
                    max_frames=max_frames,
                )

                if captions["success"]:
                    rejudge_result = self.rejudge(
                        context, captions, question, options, history_compact,
                    )
                    rejudge_result["used_visual"] = True
                    rejudge_result["caption_text"] = captions["caption_text"]
                    rejudge_result["n_frames_used"] = captions["n_frames"]
                    return rejudge_result

        # Visual not needed or not available
        verdict["used_visual"] = False
        return verdict

    # ==================================================================
    # Two-Stage Visual: Scout → Focus
    # ==================================================================

    SCOUT_CAPTION_PROMPT = """Briefly describe what you see in these video frames.
Focus on: actions happening, objects visible, persons and their appearances, any text/subtitles on screen.
Keep it concise (2-3 sentences). Be specific and factual.

For reference, the question being investigated is: {question}"""

    FOCUS_SELECT_PROMPT = """You are selecting the most promising video regions to examine in detail for answering a question.

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

    FOCUS_CAPTION_PROMPT = """Observe these video frames carefully and provide a detailed and objective description of what is shown.
You should pay special attention to any visible texts, subtitles, text overlays, or voice-overs in the video.
In addition: {instruction}

For reference:
- Question: {question}
- Options: {options_text}

Describe what you see in detail. Do NOT try to answer the question directly — just provide thorough visual descriptions."""

    def scout_frames(
        self,
        video_path: str,
        time_ranges: list[tuple[float, float]],
        question: str,
        frames_per_region: int = 3,
    ) -> list[dict]:
        """Stage 1 — Scout: Brief captioning of each region separately.

        Args:
            video_path: path to video file
            time_ranges: [(start, end), ...] regions to scout
            question: question text for reference
            frames_per_region: frames to sample per region

        Returns:
            [
                {
                    "region_idx": int,
                    "interval": (start, end),
                    "caption": str,
                    "n_frames": int,
                },
                ...
            ]
        """
        if not self.vision_vlm or not self.frame_loader:
            print("      [Scout] No vision_vlm or frame_loader")
            return []

        if not video_path or not os.path.exists(video_path):
            print(f"      [Scout] Video not found: {video_path}")
            return []

        if not time_ranges:
            return []

        # Load frames per interval (separated)
        try:
            per_interval = self.frame_loader.load_per_interval(
                video_path, time_ranges, frames_per_interval=frames_per_region,
            )
        except Exception as e:
            print(f"      [Scout] Frame load error: {e}")
            return []

        scout_results = []
        for idx, interval_data in enumerate(per_interval):
            frames_np = interval_data["frames"]
            interval = interval_data["interval"]

            if frames_np is None or len(frames_np) == 0:
                scout_results.append({
                    "region_idx": idx,
                    "interval": interval,
                    "caption": "(no frames loaded)",
                    "n_frames": 0,
                })
                continue

            # Build scout prompt
            scout_prompt = self.SCOUT_CAPTION_PROMPT.format(
                question=question,
            )

            try:
                result = self.vision_vlm.infer(
                    frames_np, scout_prompt, question, [],
                )
                # Extract caption from VLM response
                caption = ""
                if isinstance(result, dict):
                    caption = (
                        result.get("observation", "")
                        or result.get("scene_description", "")
                        or result.get("answer", "")
                    )
                if not caption:
                    caption = str(result)[:300]
            except Exception as e:
                print(f"      [Scout] VLM error for region {idx}: {e}")
                caption = f"(error: {e})"
            finally:
                del frames_np

            scout_results.append({
                "region_idx": idx,
                "interval": interval,
                "caption": caption,
                "n_frames": len(interval_data["frame_seconds"]),
            })

            print(f"      [Scout] Region {idx} [{interval[0]:.1f}s-{interval[1]:.1f}s]: "
                  f"{caption[:80]}...")

        gc.collect()
        return scout_results

    def select_focus_regions(
        self,
        scout_results: list[dict],
        question: str,
        options: list[str],
    ) -> dict:
        """LLM selects promising regions + generates VLM instruction.

        Args:
            scout_results: output from scout_frames()
            question: question text
            options: answer options

        Returns:
            {
                "selected_indices": [int, ...],
                "instruction": str,
                "reasoning": str,
            }
        """
        default = {
            "selected_indices": [0] if scout_results else [],
            "instruction": "Describe all visible details carefully.",
            "reasoning": "fallback",
        }

        if not self.llm_fn or not scout_results:
            return default

        opt_text = "\n".join(
            f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        )

        # Format scout observations
        obs_lines = []
        for sr in scout_results:
            obs_lines.append(
                f"[Region {sr['region_idx']}] "
                f"{sr['interval'][0]:.1f}s - {sr['interval'][1]:.1f}s "
                f"({sr['n_frames']} frames)\n"
                f"  Caption: {sr['caption']}"
            )
        scout_text = "\n\n".join(obs_lines)

        prompt = self.FOCUS_SELECT_PROMPT.format(
            question=question,
            options_text=opt_text,
            scout_observations=scout_text,
        )

        try:
            result = self.llm_fn(prompt, max_tokens=300)
        except Exception as e:
            print(f"      [FocusSelect] Error: {e}")
            return default

        if not isinstance(result, dict):
            return default

        # Parse selected regions
        selected = result.get("selected_regions", [])
        if isinstance(selected, list):
            valid_indices = [
                int(i) for i in selected
                if isinstance(i, (int, float)) and 0 <= int(i) < len(scout_results)
            ]
        else:
            valid_indices = []

        if not valid_indices:
            valid_indices = [0]

        instruction = result.get("instruction", default["instruction"])
        reasoning = result.get("reasoning", "")

        print(f"      [FocusSelect] Selected regions: {valid_indices}")
        print(f"      [FocusSelect] Instruction: {instruction[:100]}...")

        return {
            "selected_indices": valid_indices[:3],
            "instruction": instruction,
            "reasoning": reasoning,
        }

    def focus_frames(
        self,
        video_path: str,
        focus_regions: list[dict],
        instruction: str,
        question: str,
        options: list[str],
        max_frames: int = 30,
    ) -> dict:
        """Stage 2 — Focus: Detailed captioning with LLM-generated instruction.

        Args:
            video_path: path to video file
            focus_regions: selected scout_results entries
            instruction: LLM-generated instruction for VLM
            question: question text
            options: answer options
            max_frames: max total frames across focus regions

        Returns:
            Same structure as caption_frames() output
        """
        empty = {
            "frame_captions": [], "query_relevant_details": [],
            "key_observation": "", "scene_description": "",
            "caption_text": "", "n_frames": 0, "success": False,
        }

        if not self.vision_vlm or not self.frame_loader:
            return empty

        if not focus_regions:
            return empty

        # Collect time ranges from selected regions
        focus_intervals = [r["interval"] for r in focus_regions]

        # Load frames (concentrated on fewer regions → more per region)
        try:
            frames_np, frame_secs = self.frame_loader.load(
                video_path, focus_intervals, max_frames=max_frames,
            )
        except Exception as e:
            print(f"      [Focus] Frame load error: {e}")
            return empty

        if frames_np is None or len(frame_secs) == 0:
            return empty

        print(f"      [Focus] Loaded {len(frame_secs)} frames for "
              f"{len(focus_regions)} regions")

        opt_text = "\n".join(
            f"{chr(65 + i)}. {o}" for i, o in enumerate(options)
        )

        # Build focus prompt with VideoLucy-style instruction
        focus_prompt = self.FOCUS_CAPTION_PROMPT.format(
            instruction=instruction,
            question=question,
            options_text=opt_text,
        )

        try:
            options_list = [o.strip() for o in opt_text.split("\n") if o.strip()]
            result = self.vision_vlm.infer(
                frames_np, focus_prompt, question, options_list,
            )
        except Exception as e:
            print(f"      [Focus] VLM error: {e}")
            result = {}
        finally:
            del frames_np
            gc.collect()

        if not isinstance(result, dict):
            result = {"frame_captions": [str(result)]}

        # Format with focus region info
        caption_text = self._format_focus_caption_text(
            result, focus_regions, instruction,
        )

        return {
            "frame_captions": result.get("frame_captions", []),
            "query_relevant_details": result.get("query_relevant_details", []),
            "key_observation": result.get("key_observation", ""),
            "scene_description": result.get("scene_description", ""),
            "caption_text": caption_text,
            "n_frames": len(frame_secs),
            "success": True,
        }

    def judge_full_two_stage(
        self,
        context: str,
        question: str,
        options: list[str],
        video_path: str | None = None,
        time_ranges: list[tuple[float, float]] | None = None,
        history_compact: str | None = None,
        max_frames: int = 30,
        scout_frames_per_region: int = 3,
    ) -> dict:
        """Two-stage visual: judge → scout → select → focus → rejudge.

        Args:
            context: assembled text context
            question: question text
            options: answer options
            video_path: path to video file
            time_ranges: [(start, end), ...] target regions
            history_compact: accumulated history text
            max_frames: max frames for focus stage
            scout_frames_per_region: frames per region in scout stage
        """
        # Step 1: Initial text-only judge
        verdict = self.judge_with_visual_need(
            context, question, options, history_compact,
        )

        print(f"      [TwoStage] answerable={verdict.get('answerable')} | "
              f"conf={verdict.get('confidence')} | "
              f"needs_visual={verdict.get('needs_visual')}")

        # If high confidence → done
        if (verdict.get("answerable")
                and verdict.get("confidence") == "high"
                and verdict.get("answer")):
            verdict["used_visual"] = False
            verdict["visual_stage"] = "none"
            return verdict

        # If no video or no regions → return text-only verdict
        if not video_path or not time_ranges:
            verdict["used_visual"] = False
            verdict["visual_stage"] = "none"
            return verdict

        # Step 2: Scout — brief caption per region
        print(f"      [TwoStage] Scout: {len(time_ranges)} regions × "
              f"{scout_frames_per_region} frames")

        scout_results = self.scout_frames(
            video_path, time_ranges, question,
            frames_per_region=scout_frames_per_region,
        )

        if not scout_results:
            verdict["used_visual"] = False
            verdict["visual_stage"] = "scout_failed"
            return verdict

        # Step 3: LLM selects focus regions + generates instruction
        focus_selection = self.select_focus_regions(
            scout_results, question, options,
        )

        selected_indices = focus_selection["selected_indices"]
        instruction = focus_selection["instruction"]
        selected_regions = [
            sr for sr in scout_results if sr["region_idx"] in selected_indices
        ]

        # Step 4: Focus — detailed captioning on selected regions
        print(f"      [TwoStage] Focus: {len(selected_regions)} regions × "
              f"{max_frames} frames total")

        focus_captions = self.focus_frames(
            video_path, selected_regions, instruction,
            question, options, max_frames=max_frames,
        )

        if not focus_captions["success"]:
            # Fall back to scout captions for rejudge
            scout_text = self._format_scout_as_caption(scout_results)
            focus_captions = {
                "caption_text": scout_text,
                "success": True,
                "n_frames": sum(sr["n_frames"] for sr in scout_results),
            }

        # Step 5: Rejudge with focus captions
        rejudge_result = self.rejudge(
            context, focus_captions, question, options, history_compact,
        )
        rejudge_result["used_visual"] = True
        rejudge_result["visual_stage"] = "two_stage"
        rejudge_result["caption_text"] = focus_captions["caption_text"]
        rejudge_result["n_frames_used"] = focus_captions["n_frames"]
        rejudge_result["scout_regions"] = len(scout_results)
        rejudge_result["focus_regions"] = len(selected_regions)
        rejudge_result["focus_instruction"] = instruction

        return rejudge_result

    def _format_focus_caption_text(
        self, caption_result: dict, focus_regions: list[dict],
        instruction: str,
    ) -> str:
        """Format focus caption with region info and instruction."""
        parts = []

        regions_str = ", ".join(
            f"{r['interval'][0]:.1f}s-{r['interval'][1]:.1f}s"
            for r in focus_regions
        )
        parts.append(f"=== Detailed Visual Observations ===")
        parts.append(f"Focus regions: {regions_str}")
        parts.append(f"Observation focus: {instruction[:200]}")

        scene = caption_result.get("scene_description", "")
        if scene:
            parts.append(f"Scene: {scene}")

        observation = caption_result.get("observation", "")
        if observation:
            parts.append(f"Observation: {observation}")

        captions = caption_result.get("frame_captions", [])
        if captions:
            parts.append("Frame descriptions:")
            for i, cap in enumerate(captions):
                parts.append(f"  {i + 1}. {cap}")

        details = caption_result.get("query_relevant_details", [])
        if details:
            parts.append("Relevant details:")
            for d in details:
                parts.append(f"  - {d}")

        key_obs = caption_result.get("key_observation", "")
        if key_obs:
            parts.append(f"Key observation: {key_obs}")

        return "\n".join(parts)

    def _format_scout_as_caption(self, scout_results: list[dict]) -> str:
        """Format scout results as fallback caption text."""
        parts = ["=== Scout Visual Observations ==="]
        for sr in scout_results:
            parts.append(
                f"[{sr['interval'][0]:.1f}s-{sr['interval'][1]:.1f}s]: "
                f"{sr['caption']}"
            )
        return "\n".join(parts)

    # ========================= Internal =========================

    def _normalize_judge_output(self, result: dict) -> dict:
        """Normalize and validate judge output fields."""
        # Answer: extract letter
        answer = result.get("answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            result["answer"] = m.group(0) if m else None

        # Confidence
        conf = str(result.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        result["confidence"] = conf

        # Evidence-based confidence downgrade:
        # "high" conf인데 evidence가 비어있으면 → "medium"으로 강제
        evidence = result.get("evidence", "")
        eliminates = result.get("eliminates", {})
        if conf == "high":
            if not evidence or len(str(evidence).strip()) < 10:
                result["confidence"] = "medium"
                result.setdefault("_downgrade_reason", "no evidence provided")
            elif not eliminates or len(eliminates) == 0:
                result["confidence"] = "medium"
                result.setdefault("_downgrade_reason", "no alternatives eliminated")

        # Booleans
        result["answerable"] = bool(result.get("answerable", False))
        result["needs_visual"] = bool(result.get("needs_visual", False))

        # Defaults
        result.setdefault("visual_query", None)
        result.setdefault("visual_focus", [])
        result.setdefault("reasoning", "")
        result.setdefault("missing_info", None)
        result.setdefault("search_direction", None)

        return result

    def _format_caption_text(self, caption_result: dict, visual_query: str) -> str:
        """Format caption result into readable text for context injection."""
        parts = []

        parts.append(f"=== Visual Observations (query: {visual_query}) ===")

        # Scene description
        scene = caption_result.get("scene_description", "")
        if scene:
            parts.append(f"Scene: {scene}")

        # Frame captions
        captions = caption_result.get("frame_captions", [])
        if captions:
            parts.append("Frame descriptions:")
            for i, cap in enumerate(captions):
                parts.append(f"  {i + 1}. {cap}")

        # Query-relevant details
        details = caption_result.get("query_relevant_details", [])
        if details:
            parts.append("Query-relevant details:")
            for d in details:
                parts.append(f"  - {d}")

        # Key observation
        key_obs = caption_result.get("key_observation", "")
        if key_obs:
            parts.append(f"Key observation: {key_obs}")

        return "\n".join(parts)
