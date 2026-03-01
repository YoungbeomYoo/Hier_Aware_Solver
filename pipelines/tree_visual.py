"""
Tree-Guided Visual Search (TGVS) Pipeline v2

Phase 1: Text coarse (hierarchical/flat summary → forced answer)
Phase 2: Tree-guided localization → VLM frame observation → re-judge (up to N hops)
Phase 3: Forced answer with all accumulated context

Modes:
- confidence_threshold: high/medium/low/none (none = skip Phase 1, always visual)
- localize_mode: key_elements / llm_select / combined / agentic
- vlm_mode: direct / caption
- max_intervals: 1~5 (1 = single segment per hop, VideoLucy style)
"""

import os
import re
import json
from pipelines.base import BasePipeline


class TreeVisualPipeline(BasePipeline):
    """Tree-Guided Visual Search pipeline.

    Uses memory tree for time localization, then VLM for visual observation.
    """

    PHASE0_PROMPT = (
        "The following provides descriptions of what's shown in the video "
        "during different time periods:\n\n"
        "{context}\n\n"
        "Now, a question has been raised regarding the content descriptions "
        "of this video.\n{question}\n\n{options_text}\n\n"
        "Please read and understand the given video content and question in "
        "depth. Strictly based on the video content, select the single best "
        "option. You must choose an option from these provided options. The "
        "answer you provide must include the English letters of the options "
        "[A, B, C, D].\n\n"
        "Please note that if an ordinal number appears in the provided "
        "question, in most cases, the meaning of this ordinal number is not "
        "related to the ordinal of the provided time period. You need to "
        "focus on analyzing the meaning of this ordinal number.\n\n"
        "Please output ONLY valid JSON in a strictly standardized format:\n"
        '{{\n'
        '    "answerable": true,\n'
        '    "answer": "A" or "B" or "C" or "D",\n'
        '    "confidence": "high" or "medium" or "low",\n'
        '    "reasoning": "Your reasoning about your judgment.",\n'
        '    "missing_info": "What specific information is missing or unclear",\n'
        '    "search_direction": "Where/what to look for next"\n'
        '}}\n'
        "- You MUST provide an answer. Do not refuse."
    )

    LOCALIZE_PROMPT = (
        "Based on the video overview below, identify which time periods are "
        "most likely to contain the answer to the question.\n\n"
        "=== Video Overview ===\n{context}\n\n"
        "Question: {question}\n{options_text}\n\n"
        "Select 1-3 time periods that are most relevant to answering this "
        "question. Output ONLY valid JSON:\n"
        '{{\n'
        '    "periods": [[start_sec, end_sec], ...],\n'
        '    "reason": "Brief explanation of why these periods are relevant"\n'
        '}}'
    )

    AGENTIC_PLAN_PROMPT = (
        "You are analyzing a long video to answer a question. "
        "You have access to these tools:\n"
        "1. scene_browse: Get overview of scenes in a time range\n"
        "2. caption_search: Search detailed segment descriptions by query\n"
        "3. visual_inspect: Look at actual video frames in a time range\n\n"
        "=== Video Overview ===\n{context}\n\n"
        "Question: {question}\n{options_text}\n\n"
        "{history}\n"
        "Based on the information so far, decide your next action. "
        "Output ONLY valid JSON:\n"
        '{{\n'
        '    "tool": "scene_browse" or "caption_search" or "visual_inspect" or "answer",\n'
        '    "query": "What to search for / inspect",\n'
        '    "time_range": [start_sec, end_sec] or null,\n'
        '    "reasoning": "Why this action",\n'
        '    "answer": "A/B/C/D (only if tool=answer)",\n'
        '    "confidence": "high/medium/low (only if tool=answer)"\n'
        '}}'
    )

    def __init__(self, components, adapter, config):
        super().__init__(components, adapter, config)
        # Note: config is already flattened pipeline_params (via solver.py pipe_config)
        self.coarse_mode = config.get("coarse_mode", "hierarchical")
        self.localize_mode = config.get("localize_mode", "key_elements")
        self.vlm_mode = config.get("vlm_mode", "direct")
        self.max_visual_iterations = config.get("max_visual_iterations", 1)
        self.max_frames = config.get("max_frames", 16)
        self.confidence_threshold = config.get("confidence_threshold", "medium")
        self.max_intervals = config.get("max_intervals", 5)
        # Budget-constrained selection
        self.leaf_budget = config.get("leaf_budget", 0)  # 0 = unlimited
        self.budget_strategy = config.get("budget_strategy", "uniform")  # uniform, sequential, hierarchy

    def solve(self, question_data, memory, video_id):
        question = question_data["question"]
        options = question_data.get("options", [])
        answer = question_data.get("answer")
        time_reference = question_data.get("time_reference", "")

        opt_text = "\n".join(
            f"{chr(65+i)}. {o}" for i, o in enumerate(options)
        )

        tree = memory.get("streaming_memory_tree", memory)

        # ====== Phase 1: Text Coarse Answer ======
        coarse_context = self._build_coarse_context(tree, question, options)
        verdict = self._judge_answer(coarse_context, question, opt_text)

        phase1_answer = verdict.get("answer")
        phase1_conf = verdict.get("confidence", "low")
        phase1_search_dir = verdict.get("search_direction")
        phase1_missing = verdict.get("missing_info")
        print(f"    [Phase1] answer={phase1_answer} conf={phase1_conf}")
        if phase1_search_dir:
            print(f"    [Phase1] search_direction={phase1_search_dir}")

        # Base result fields (shared across all phases)
        base_result = {}
        if time_reference:
            base_result["time_reference"] = time_reference

        if self._is_confident(phase1_conf):
            correct = self._check_correct(phase1_answer, answer)
            return {
                **base_result,
                "pred": phase1_answer,
                "answer": answer,
                "correct": correct,
                "phase": "phase1_coarse",
                "confidence": phase1_conf,
            }

        # ====== Agentic mode: separate loop ======
        if self.localize_mode == "agentic":
            return self._agentic_search(
                tree, question, options, opt_text, answer,
                coarse_context, video_id,
                phase1_answer, phase1_conf,
            )

        # ====== Verified mode: Vgent-style sub-question verification ======
        if self.localize_mode == "verified":
            return self._phase2_verified(
                tree, question, options, opt_text, answer,
                coarse_context, video_id, base_result,
                phase1_answer, phase1_conf,
                search_hint=phase1_search_dir or phase1_missing,
            )

        # ====== Phase 2: Tree-Guided Visual Search ======
        video_path = self.adapter.get_video_path(video_id)
        if not video_path or not os.path.exists(video_path):
            print(f"    [Phase2] No video file, skipping visual search")
            correct = self._check_correct(phase1_answer, answer)
            return {
                **base_result,
                "pred": phase1_answer,
                "answer": answer,
                "correct": correct,
                "phase": "phase1_no_video",
                "confidence": phase1_conf,
            }

        visual_observations = []
        leaf_caption_entries = []
        excluded_periods = set()
        all_selected_intervals = []
        best_answer = phase1_answer
        best_conf = phase1_conf
        search_hint = phase1_search_dir or phase1_missing

        for iteration in range(self.max_visual_iterations):
            print(f"    [Phase2] Iteration {iteration+1}/{self.max_visual_iterations}")

            # Step 1: Localize relevant time periods
            intervals = self._localize(
                tree, question, options, coarse_context,
                excluded_periods, search_hint=search_hint,
            )

            if not intervals:
                print(f"    [Phase2] No intervals found, stopping")
                break

            # Mark as explored
            for iv in intervals:
                excluded_periods.add((iv[0], iv[1]))
            all_selected_intervals.extend(intervals)

            print(f"    [Phase2] Intervals: {intervals}")

            # Step 1.5: Extract leaf raw captions for selected intervals
            leaf_captions = self._get_leaf_captions(tree, intervals)
            if leaf_captions:
                leaf_caption_entries.extend(leaf_captions)
                print(f"    [Phase2] Got {len(leaf_captions)} leaf captions "
                      f"(total: {len(leaf_caption_entries)})")

            # Step 2: VLM Observe
            obs_result = self._visual_observe(
                video_path, intervals, question, options,
                coarse_context, visual_observations,
            )

            if obs_result:
                visual_observations.append(obs_result)

            # Step 3: Re-judge with accumulated context
            if leaf_captions or obs_result:
                enriched_context = self._build_enriched_context(
                    coarse_context, visual_observations, leaf_caption_entries,
                )
                verdict = self._judge_answer(enriched_context, question, opt_text)

                new_answer = verdict.get("answer")
                new_conf = verdict.get("confidence", "low")
                search_hint = verdict.get("search_direction") or verdict.get("missing_info")
                print(f"    [Phase2] iter={iteration+1} answer={new_answer} conf={new_conf}")

                best_answer = new_answer
                best_conf = new_conf

                if self._is_confident(new_conf):
                    correct = self._check_correct(best_answer, answer)
                    return {
                        **base_result,
                        "pred": best_answer,
                        "answer": answer,
                        "correct": correct,
                        "phase": f"phase2_iter{iteration+1}",
                        "confidence": best_conf,
                        "visual_observations": len(visual_observations),
                        "selected_intervals": all_selected_intervals,
                    }

        # ====== Phase 3: Forced Answer ======
        if visual_observations or leaf_caption_entries:
            final_context = self._build_enriched_context(
                coarse_context, visual_observations, leaf_caption_entries,
            )
        else:
            final_context = coarse_context

        verdict = self._judge_answer(final_context, question, opt_text)
        final_answer = verdict.get("answer") or best_answer
        print(f"    [Phase3] forced answer={final_answer}")

        correct = self._check_correct(final_answer, answer)
        return {
            **base_result,
            "pred": final_answer,
            "answer": answer,
            "correct": correct,
            "phase": "phase3_forced",
            "confidence": verdict.get("confidence", "low"),
            "visual_observations": len(visual_observations),
            "selected_intervals": all_selected_intervals,
        }

    # ==================== Vgent-style Verified Search ====================

    def _phase2_verified(self, tree, question, options, opt_text, answer,
                         coarse_context, video_id, base_result,
                         phase1_answer, phase1_conf, search_hint=None):
        """Vgent-style: decompose → retrieve → VLM verify frames → aggregate → answer.

        Unlike iterative Phase 2, this does a single pass:
        1. Decompose question into binary sub-questions
        2. Retrieve candidate intervals (llm_index, broader)
        3. For each candidate: load frames → VLM verify+caption
        4. Filter: keep only verified observations
        5. Final answer with coarse + verified observations
        """
        llm = self.components.get("llm")
        frame_loader = self.components.get("frame_loader")
        vision_vlm = self.components.get("vision_vlm")

        video_path = self.adapter.get_video_path(video_id)
        if not video_path or not os.path.exists(video_path):
            print(f"    [Verified] No video file")
            correct = self._check_correct(phase1_answer, answer)
            return {
                **base_result,
                "pred": phase1_answer, "answer": answer, "correct": correct,
                "phase": "phase1_no_video", "confidence": phase1_conf,
            }

        # Step 1: Decompose question into sub-questions
        sub_questions = self._decompose_subquestions(llm, question, options)
        print(f"    [Verified] Sub-questions ({len(sub_questions)}): {sub_questions}")

        # Step 2: Retrieve candidate intervals (broader than usual)
        # Save current max_intervals, temporarily increase
        orig_max_intervals = self.max_intervals
        self.max_intervals = 10  # broader retrieval
        candidates = self._localize_llm_index(
            tree, question, options, set(), search_hint=search_hint,
        )
        self.max_intervals = orig_max_intervals

        if not candidates:
            print(f"    [Verified] No candidates, returning Phase 1 answer")
            correct = self._check_correct(phase1_answer, answer)
            return {
                **base_result,
                "pred": phase1_answer, "answer": answer, "correct": correct,
                "phase": "phase2_verified_no_candidates", "confidence": phase1_conf,
            }

        print(f"    [Verified] {len(candidates)} candidate intervals: {candidates}")

        # Step 3: For each candidate, load frames → VLM verify+caption
        verified_observations = []
        all_intervals = []
        sq_text = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_questions))

        for idx, interval in enumerate(candidates):
            st, et = interval
            try:
                frames_np, frame_secs = frame_loader.load(
                    video_path, [interval], max_frames=self.max_frames,
                )
            except Exception as e:
                print(f"    [Verified] Frame load error for {st:.0f}-{et:.0f}s: {e}")
                continue

            if frames_np is None or len(frames_np) == 0:
                continue

            # VLM: verify sub-questions + describe relevant content
            verify_context = (
                f"Look at these frames from {st:.0f}s-{et:.0f}s of the video.\n\n"
                f"Verification sub-questions:\n{sq_text}\n\n"
                f"Instructions:\n"
                f"1. For each sub-question, answer YES or NO based on what you see.\n"
                f"2. Describe any relevant visual details for: {question}\n"
                f"3. Be specific about what you observe."
            )

            try:
                vlm_result = vision_vlm.infer(
                    frames_np, verify_context, question, options,
                    max_tokens=500,
                )
            except Exception as e:
                print(f"    [Verified] VLM error for {st:.0f}-{et:.0f}s: {e}")
                continue

            obs_text = vlm_result.get("observation", vlm_result.get("raw_response", ""))
            vlm_answer = vlm_result.get("answer", "")

            # Check verification: does the response contain any YES?
            passes = "yes" in obs_text.lower()
            print(f"    [Verified] [{idx+1}/{len(candidates)}] "
                  f"{st:.0f}-{et:.0f}s: {'PASS' if passes else 'FAIL'} "
                  f"({len(frames_np)} frames)")

            if passes:
                verified_observations.append({
                    "interval": interval,
                    "observation": obs_text,
                    "vlm_answer": vlm_answer,
                    "n_frames": len(frames_np),
                })
                all_intervals.append(interval)

        print(f"    [Verified] {len(verified_observations)}/{len(candidates)} passed verification")

        # Step 4: Also get leaf captions for verified intervals
        leaf_captions = []
        for vobs in verified_observations:
            lc = self._get_leaf_captions(tree, [vobs["interval"]])
            if lc:
                leaf_captions.extend(lc)

        # Step 5: Build enriched context and final answer
        if verified_observations:
            # Build verified context
            obs_parts = []
            for vobs in verified_observations:
                st, et = vobs["interval"]
                obs_parts.append(
                    f"[{st:.0f}s-{et:.0f}s] ({vobs['n_frames']} frames observed)\n"
                    f"{vobs['observation']}"
                )

            obs_text_all = "\n\n".join(obs_parts)

            # Include leaf captions if available
            leaf_text = ""
            if leaf_captions:
                leaf_parts = [
                    f"[{lc['start']:.0f}s-{lc['end']:.0f}s] {lc['caption']}"
                    for lc in leaf_captions
                ]
                leaf_text = (
                    "\n\n=== Detailed Descriptions (verified segments) ===\n"
                    + "\n\n".join(leaf_parts)
                )

            enriched = (
                f"{coarse_context}\n\n"
                f"=== Visual Observations (verified) ===\n{obs_text_all}"
                f"{leaf_text}"
            )

            verdict = self._judge_answer(enriched, question, opt_text)
            final_answer = verdict.get("answer") or phase1_answer
            final_conf = verdict.get("confidence", "low")
        else:
            # No observations passed verification → fall back to Phase 1
            final_answer = phase1_answer
            final_conf = phase1_conf

        print(f"    [Verified] Final answer={final_answer} (verified={len(verified_observations)})")

        correct = self._check_correct(final_answer, answer)
        return {
            **base_result,
            "pred": final_answer,
            "answer": answer,
            "correct": correct,
            "phase": f"phase2_verified",
            "confidence": final_conf,
            "n_candidates": len(candidates),
            "n_verified": len(verified_observations),
            "sub_questions": sub_questions,
            "verified_intervals": all_intervals,
        }

    def _decompose_subquestions(self, llm, question, options):
        """Decompose question into binary verification sub-questions (Vgent-style)."""
        if not llm:
            return [question]

        opt_text = ""
        if options:
            opt_text = "Options: " + " / ".join(
                f"{chr(65+i)}. {o}" for i, o in enumerate(options)
            )

        prompt = (
            f"Question: {question}\n{opt_text}\n\n"
            "Decompose this question into 2-4 specific verification sub-questions. "
            "Each sub-question should be binary (answerable with YES/NO) and help "
            "verify if a video segment contains relevant information.\n"
            "Focus on: key entities, actions, events, objects, or temporal cues.\n\n"
            'Output ONLY valid JSON: {"sub_questions": ["Is X present?", "Does Y happen?", ...]}'
        )

        try:
            result = llm.reason(prompt, max_tokens=300)
            sqs = result.get("sub_questions", [])
            if sqs and isinstance(sqs, list):
                return sqs[:4]
        except Exception as e:
            print(f"    [Decompose] Error: {e}")

        return [f"Is there information relevant to: {question}"]

    # ==================== Agentic Search ====================

    def _agentic_search(self, tree, question, options, opt_text, answer,
                        coarse_context, video_id,
                        phase1_answer, phase1_conf):
        """Agentic think-act-observe loop with multi-granularity tools."""
        video_path = self.adapter.get_video_path(video_id)
        history_entries = []
        best_answer = phase1_answer
        best_conf = phase1_conf

        for step in range(self.max_visual_iterations):
            print(f"    [Agentic] Step {step+1}/{self.max_visual_iterations}")

            # Think: LLM decides which tool to use
            history_text = ""
            if history_entries:
                history_text = "=== Previous Actions ===\n"
                for i, h in enumerate(history_entries):
                    history_text += (
                        f"\nStep {i+1}: [{h['tool']}] query=\"{h['query']}\""
                        f" time={h.get('time_range', 'all')}\n"
                        f"Result: {h['result'][:500]}\n"
                    )

            plan = self._agentic_plan(
                coarse_context, question, opt_text, history_text,
            )
            if not plan:
                break

            tool = plan.get("tool", "")
            query = plan.get("query", question)
            time_range = plan.get("time_range")

            # Validate and normalize time_range
            if time_range is not None:
                if isinstance(time_range, str):
                    try:
                        time_range = json.loads(time_range)
                    except (json.JSONDecodeError, ValueError):
                        time_range = None
                if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
                    try:
                        time_range = [float(time_range[0]), float(time_range[1])]
                    except (TypeError, ValueError):
                        time_range = None
                else:
                    time_range = None

            print(f"    [Agentic] tool={tool} query={query[:60]} time={time_range}")

            # Answer: early exit
            if tool == "answer":
                ans = plan.get("answer")
                conf = plan.get("confidence", "medium")
                if ans:
                    best_answer = ans
                    best_conf = conf
                    correct = self._check_correct(best_answer, answer)
                    return {
                        "pred": best_answer,
                        "answer": answer,
                        "correct": correct,
                        "phase": f"agentic_step{step+1}",
                        "confidence": best_conf,
                        "agentic_steps": len(history_entries) + 1,
                    }

            # Act: execute the chosen tool
            try:
                result_text = self._agentic_execute(
                    tool, query, time_range, tree, video_path, question, options,
                )
            except Exception as e:
                print(f"    [Agentic] Execute error: {e}")
                result_text = f"Tool execution failed: {e}"

            # Observe: store result
            history_entries.append({
                "tool": tool,
                "query": query,
                "time_range": time_range,
                "result": result_text,
            })

        # Forced answer with all accumulated context
        all_results = "\n\n".join(
            f"[{h['tool']}] {h['result'][:800]}" for h in history_entries
        )
        final_context = (
            f"{coarse_context}\n\n"
            f"=== Investigation Results ===\n{all_results}"
        )
        verdict = self._judge_answer(final_context, question, opt_text)
        final_answer = verdict.get("answer") or best_answer
        print(f"    [Agentic] forced answer={final_answer}")

        correct = self._check_correct(final_answer, answer)
        return {
            "pred": final_answer,
            "answer": answer,
            "correct": correct,
            "phase": f"agentic_forced",
            "confidence": verdict.get("confidence", "low"),
            "agentic_steps": len(history_entries),
        }

    def _agentic_plan(self, context, question, opt_text, history_text):
        """LLM decides which tool to call next."""
        llm = self.components.get("llm")
        if not llm:
            return None

        prompt = self.AGENTIC_PLAN_PROMPT.format(
            context=context,
            question=question,
            options_text=opt_text,
            history=history_text,
        )

        try:
            result = llm.reason(prompt, max_tokens=400)
        except Exception as e:
            print(f"    [Agentic Plan] Error: {e}")
            return None

        # Normalize answer field
        if result.get("tool") == "answer":
            ans = result.get("answer", "")
            if ans:
                m = re.search(r"[ABCD]", str(ans).upper())
                result["answer"] = m.group(0) if m else None

        return result

    def _agentic_execute(self, tool, query, time_range, tree, video_path,
                         question, options):
        """Execute the selected tool and return result text."""
        if tool == "scene_browse":
            return self._tool_scene_browse(tree, query, time_range)
        elif tool == "caption_search":
            return self._tool_caption_search(tree, query, time_range)
        elif tool == "visual_inspect":
            return self._tool_visual_inspect(
                video_path, time_range, question, options,
            )
        return f"Unknown tool: {tool}"

    def _tool_scene_browse(self, tree, query, time_range):
        """Browse scene summaries, optionally filtered by time range."""
        parts = []
        tr = self._safe_time_range(time_range)
        for level_name in sorted(
            [k for k in tree.keys() if k.startswith("Level_")],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        ):
            for node in tree.get(level_name, []):
                start, end = self._get_time_range(node)
                if tr:
                    if end < tr[0] or start > tr[1]:
                        continue
                summary = node.get("summary", "")
                if summary:
                    ke = node.get("key_elements", {})
                    ke_str = ""
                    for f in ["actions", "objects", "persons"]:
                        vals = ke.get(f, [])
                        if vals:
                            ke_str += f" | {f}: {', '.join(str(v) for v in vals[:5])}"
                    parts.append(
                        f"[{level_name}] [{start:.0f}s-{end:.0f}s] {summary}{ke_str}"
                    )
        return "\n".join(parts) if parts else "No scenes found in this range."

    def _tool_caption_search(self, tree, query, time_range):
        """Search leaf raw captions matching query keywords."""
        tr = self._safe_time_range(time_range)
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        results = []
        for l1_node in tree.get("Level_1", []):
            for child in l1_node.get("children", []):
                st = float(child.get("start_time", 0))
                et = float(child.get("end_time", 0))
                if tr:
                    if et < tr[0] or st > tr[1]:
                        continue
                caption = child.get("caption", "")
                if not caption:
                    continue
                # Simple keyword relevance
                cap_lower = caption.lower()
                score = sum(1 for w in query_words if w in cap_lower)
                if score > 0:
                    results.append((score, st, et, caption))

        results.sort(key=lambda x: (-x[0], x[1]))
        if not results:
            return "No matching captions found."

        lines = []
        for score, st, et, caption in results[:3]:
            lines.append(f"[{st:.0f}s-{et:.0f}s] (relevance={score}) {caption}")
        return "\n\n".join(lines)

    def _tool_visual_inspect(self, video_path, time_range, question, options):
        """Load frames from time range and describe with VLM."""
        if not video_path or not os.path.exists(video_path):
            return "No video file available."
        tr = self._safe_time_range(time_range)
        if not tr:
            return "No valid time range specified for visual inspection."

        frame_loader = self.components.get("frame_loader")
        vision_vlm = self.components.get("vision_vlm")
        if not frame_loader or not vision_vlm:
            return "Visual tools not available."

        intervals = [(tr[0], tr[1])]
        try:
            frames_np, frame_secs = frame_loader.load(
                video_path, intervals, max_frames=self.max_frames,
            )
        except Exception as e:
            return f"Frame loading error: {e}"

        if frames_np is None or len(frames_np) == 0:
            return "No frames loaded."

        caption_context = (
            f"Describe what you see in these video frames in detail. "
            f"Focus on details relevant to: {question}"
        )
        try:
            vlm_result = vision_vlm.infer(
                frames_np, caption_context, question, options,
                max_tokens=500,
            )
        except Exception as e:
            return f"VLM error: {e}"

        obs = vlm_result.get("observation", vlm_result.get("raw_response", ""))
        return f"[{tr[0]:.0f}s-{tr[1]:.0f}s] ({len(frames_np)} frames) {obs}"

    # ==================== Coarse Context ====================

    def _build_coarse_context(self, tree, question=None, options=None):
        """Build coarse context from memory tree."""
        if self.leaf_budget > 0:
            return self._budget_context(tree, question, options)
        if self.coarse_mode == "flat":
            return self._flat_context(tree)
        return self._hierarchical_context(tree)

    def _hierarchical_context(self, tree):
        """Level_N ~ Level_1 summary + key_elements, time-sorted."""
        level_names = sorted(
            [k for k in tree.keys() if k.startswith("Level_")],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        )

        parts = []
        for level_name in level_names:
            for node in tree.get(level_name, []):
                summary = node.get("summary", "")
                if not summary:
                    continue
                start, end = self._get_time_range(node)

                segs = node.get("time_segments", [])
                seg_strs = []
                for s in segs:
                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                        seg_strs.append(f"{float(s[0]):.0f}s-{float(s[1]):.0f}s")
                time_str = ", ".join(seg_strs) if seg_strs else f"{start:.0f}s-{end:.0f}s"

                ke = node.get("key_elements", {})
                ke_brief = ""
                for field in ["actions", "objects", "persons"]:
                    vals = ke.get(field, [])
                    if vals:
                        ke_brief += f" | {field}: {', '.join(str(v) for v in vals[:5])}"

                parts.append({
                    "start": start,
                    "text": f"[{level_name}] [{time_str}] {summary}{ke_brief}",
                })

        parts.sort(key=lambda x: x["start"])
        return (
            "=== Video Overview (all segments) ===\n"
            + "\n".join(p["text"] for p in parts)
        )

    def _flat_context(self, tree):
        """Level_1 children(leaf) summary, time-sorted."""
        leaves = []
        for l1_node in tree.get("Level_1", []):
            for child in l1_node.get("children", []):
                st = float(child.get("start_time", 0))
                et = float(child.get("end_time", 0))
                summary = child.get("summary", "")
                if summary:
                    leaves.append((st, et, summary))
        leaves.sort(key=lambda x: x[0])
        lines = [f"[{st:.0f}s-{et:.0f}s] {s}" for st, et, s in leaves]
        return "=== Video Content (all segments) ===\n" + "\n".join(lines)

    def _get_all_leaves(self, tree):
        """Extract all leaf nodes sorted by time."""
        leaves = []
        for l1_node in tree.get("Level_1", []):
            for child in l1_node.get("children", []):
                st = float(child.get("start_time", 0))
                et = float(child.get("end_time", 0))
                summary = child.get("summary", "")
                if summary:
                    leaves.append((st, et, summary))
        leaves.sort(key=lambda x: x[0])
        return leaves

    def _budget_context(self, tree, question=None, options=None):
        """Budget-constrained leaf selection."""
        all_leaves = self._get_all_leaves(tree)
        K = min(self.leaf_budget, len(all_leaves))
        total = len(all_leaves)

        if K >= total:
            # Budget >= total leaves, just use all
            selected = all_leaves
        elif self.budget_strategy == "uniform":
            selected = self._select_uniform(all_leaves, K)
        elif self.budget_strategy == "sequential":
            selected = self._select_sequential(all_leaves, K)
        elif self.budget_strategy == "hierarchy":
            selected = self._select_hierarchy(tree, all_leaves, K, question, options)
        elif self.budget_strategy == "verified":
            selected = self._select_verified(all_leaves, K, question, options)
        else:
            selected = self._select_uniform(all_leaves, K)

        print(f"    [Budget] strategy={self.budget_strategy} K={K}/{total} selected={len(selected)}")

        lines = [f"[{st:.0f}s-{et:.0f}s] {s}" for st, et, s in selected]
        return (
            f"=== Video Content ({len(selected)}/{total} segments) ===\n"
            + "\n".join(lines)
        )

    def _select_uniform(self, leaves, K):
        """Uniformly spaced sampling."""
        total = len(leaves)
        if K >= total:
            return leaves
        # Evenly spaced indices
        indices = [int(i * (total - 1) / (K - 1)) for i in range(K)] if K > 1 else [0]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        return [leaves[i] for i in unique]

    def _select_sequential(self, leaves, K):
        """First K leaves chronologically."""
        return leaves[:K]

    def _select_verified(self, all_leaves, K, question=None, options=None):
        """Vgent-style verify-then-select: decompose question → verify each leaf → keep top K.

        Unlike hierarchy (top-down exclusion), this shows ALL leaves and lets LLM
        verify relevance bottom-up. No premature exclusion.
        """
        llm = self.components.get("llm")
        if not llm or not question:
            return self._select_uniform(all_leaves, K)

        opt_text = ""
        if options:
            opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))

        # Step 1: Decompose question into verification sub-criteria
        decomp_prompt = (
            f"Question: {question}\n{opt_text}\n\n"
            "Break this question into 2-4 specific verification criteria. "
            "What specific information would a video segment need to contain "
            "to help answer this question?\n"
            'Output ONLY valid JSON: {"criteria": ["criterion1", "criterion2", ...]}'
        )

        try:
            decomp_result = llm.reason(decomp_prompt, max_tokens=300)
            criteria = decomp_result.get("criteria", [])
        except Exception as e:
            print(f"    [Verified] Decompose error: {e}")
            criteria = []

        if not criteria:
            criteria = [question]

        criteria_text = "\n".join(f"  - {c}" for c in criteria)
        print(f"    [Verified] Sub-criteria ({len(criteria)}): {criteria}")

        # Step 2: Show all leaves, ask LLM to select K most relevant
        leaf_lines = []
        for i, (st, et, summary) in enumerate(all_leaves):
            leaf_lines.append(f"  [{i+1}] {st:.0f}s-{et:.0f}s: {summary}")

        # For very long leaf lists, chunk if needed (but typically 50-120 leaves fits)
        select_prompt = (
            f"Question: {question}\n{opt_text}\n\n"
            f"Verification criteria:\n{criteria_text}\n\n"
            f"Below are {len(all_leaves)} video segments:\n\n"
            + "\n".join(leaf_lines) + "\n\n"
            f"Select exactly {K} segments most relevant to answering the question. "
            f"For each segment, verify: does it match any verification criterion? "
            f"Only select segments that pass verification.\n"
            f'Output ONLY valid JSON: {{"selected": [segment_numbers]}}'
        )

        try:
            result = llm.reason(select_prompt, max_tokens=500)
        except Exception as e:
            print(f"    [Verified] Select error: {e}")
            return self._select_uniform(all_leaves, K)

        selected_indices = result.get("selected", [])
        if not selected_indices:
            print("    [Verified] No selection, fallback to uniform")
            return self._select_uniform(all_leaves, K)

        # Convert 1-indexed to 0-indexed, deduplicate, validate
        valid = []
        seen = set()
        for idx in selected_indices:
            idx_0 = idx - 1  # 1-indexed → 0-indexed
            if 0 <= idx_0 < len(all_leaves) and idx_0 not in seen:
                seen.add(idx_0)
                valid.append(idx_0)
        valid = valid[:K]

        # If too few verified, backfill with uniform from remaining
        if len(valid) < K:
            remaining = [i for i in range(len(all_leaves)) if i not in seen]
            import random
            random.shuffle(remaining)
            valid.extend(remaining[:K - len(valid)])

        valid.sort()
        selected = [all_leaves[i] for i in valid]
        print(f"    [Verified] Selected {len(selected)}/{K} verified segments")
        return selected

    def _select_hierarchy(self, tree, all_leaves, K, question=None, options=None):
        """Hierarchy-guided top-down selection: Level_3 → Level_2 → Level_1 → leaves.

        Recursively traverse the tree top-down:
        1. Level_3: LLM allocates budget K across L3 branches
        2. Level_2: For each L3 branch, LLM allocates its sub-budget across L2 children
        3. Level_1: For each L2 branch, LLM allocates across L1 children
        4. Leaves: Uniformly sample from selected L1 branches

        Total selected leaves = K.
        """
        llm = self.components.get("llm")
        if not llm or not question:
            return self._select_uniform(all_leaves, K)

        opt_text = ""
        if options:
            opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))

        # Build leaf lookup
        leaf_map = {}
        for i, (st, et, _) in enumerate(all_leaves):
            leaf_map[(st, et)] = i

        # Find highest meaningful level
        top_level_num = 1
        for lvl_num in [3, 2, 1]:
            lvl = f"Level_{lvl_num}"
            if lvl in tree and len(tree[lvl]) > 0:
                top_level_num = lvl_num
                break

        # Recursive top-down budget allocation
        top_nodes = tree.get(f"Level_{top_level_num}", [])
        if not top_nodes:
            return self._select_uniform(all_leaves, K)

        selected_indices = self._hierarchy_recurse(
            top_nodes, top_level_num, K, question, opt_text, leaf_map, llm,
        )

        selected_indices = sorted(set(selected_indices))[:K]
        print(f"    [Hierarchy] L{top_level_num} top-down → {len(selected_indices)} leaves")

        return [all_leaves[i] for i in selected_indices if i < len(all_leaves)]

    def _hierarchy_recurse(self, nodes, level_num, budget, question, opt_text, leaf_map, llm):
        """Recursively allocate budget through tree levels.

        At each level: show node summaries → LLM allocates budget → recurse into children.
        At Level_1: uniformly sample leaves from allocated budget.
        """
        if budget <= 0 or not nodes:
            return []

        # If only 1 node, give it the entire budget and go deeper
        if len(nodes) == 1:
            node = nodes[0]
            if level_num == 1:
                return self._sample_leaves_from_l1(node, budget, leaf_map)
            children = node.get("children", [])
            if children:
                return self._hierarchy_recurse(
                    children, level_num - 1, budget, question, opt_text, leaf_map, llm,
                )
            return []

        # Build section display for LLM
        node_infos = []
        for node in nodes:
            start, end = self._get_time_range(node)
            summary = node.get("summary", "")
            n_leaves = self._count_descendant_leaves(node, f"Level_{level_num}")
            ke = node.get("key_elements", {})
            node_infos.append({
                "node": node,
                "start": start, "end": end,
                "summary": summary,
                "n_leaves": n_leaves,
                "key_elements": ke,
            })
        node_infos.sort(key=lambda x: x["start"])

        node_lines = []
        for i, ni in enumerate(node_infos):
            line = (
                f"  [{i+1}] {ni['start']:.0f}s-{ni['end']:.0f}s "
                f"({ni['n_leaves']} segments): {ni['summary']}"
            )
            # Add key_elements if available
            ke = ni.get("key_elements", {})
            if ke:
                ke_parts = []
                for cat in ("persons", "objects", "actions", "locations", "attributes", "text_ocr"):
                    vals = ke.get(cat, [])
                    if vals:
                        if isinstance(vals, list):
                            ke_parts.append(f"{cat}: {', '.join(str(v) for v in vals)}")
                        else:
                            ke_parts.append(f"{cat}: {vals}")
                if ke_parts:
                    line += f"\n      Key elements: {' | '.join(ke_parts)}"
            node_lines.append(line)

        level_label = f"Level_{level_num}"
        prompt = (
            f"[{level_label}] The video section has {len(node_infos)} sub-sections:\n\n"
            + "\n".join(node_lines) + "\n\n"
            f"Question: {question}\n{opt_text}\n\n"
            f"You have a budget of {budget} segments. "
            f"Allocate to each sub-section (more to relevant, 0 to irrelevant). "
            f"Total must equal {budget}.\n"
            f'Output ONLY valid JSON: {{"allocation": [{{"section": 1, "budget": N}}, ...]}}'
        )

        try:
            result = llm.reason(prompt, max_tokens=300)
        except Exception as e:
            print(f"    [Hierarchy L{level_num}] LLM error: {e}")
            # Fallback: distribute evenly
            per_node = max(1, budget // len(node_infos))
            result = {"allocation": [
                {"section": i+1, "budget": per_node} for i in range(len(node_infos))
            ]}

        allocations = result.get("allocation", [])
        if not allocations:
            per_node = max(1, budget // len(node_infos))
            allocations = [{"section": i+1, "budget": per_node} for i in range(len(node_infos))]

        # Parse allocations
        section_budgets = {}
        for alloc in allocations:
            sec_id = alloc.get("section", 0)
            b = alloc.get("budget", 0)
            if 1 <= sec_id <= len(node_infos) and b > 0:
                section_budgets[sec_id - 1] = b

        # Normalize to budget
        total_alloc = sum(section_budgets.values())
        if total_alloc > 0 and total_alloc != budget:
            factor = budget / total_alloc
            section_budgets = {
                k: max(1, round(v * factor)) for k, v in section_budgets.items()
            }

        print(f"    [Hierarchy L{level_num}] {len(node_infos)} nodes, "
              f"budget={budget} → {dict(section_budgets)}")

        # Recurse into each allocated section
        selected = []
        for sec_idx, sec_budget in section_budgets.items():
            if sec_idx >= len(node_infos):
                continue
            node = node_infos[sec_idx]["node"]

            if level_num == 1:
                # Bottom: sample leaves directly
                selected.extend(self._sample_leaves_from_l1(node, sec_budget, leaf_map))
            else:
                # Recurse into children
                children = node.get("children", [])
                if children:
                    selected.extend(self._hierarchy_recurse(
                        children, level_num - 1, sec_budget,
                        question, opt_text, leaf_map, llm,
                    ))
                else:
                    # No children structure, get descendant leaves directly
                    desc = self._get_descendant_leaf_indices(
                        node, f"Level_{level_num}", leaf_map
                    )
                    if desc:
                        step = max(1, len(desc) / sec_budget)
                        for j in range(min(sec_budget, len(desc))):
                            selected.append(desc[int(j * step)])

        return selected

    def _sample_leaves_from_l1(self, l1_node, budget, leaf_map):
        """Uniformly sample leaf indices from a Level_1 node."""
        indices = []
        for child in l1_node.get("children", []):
            st = float(child.get("start_time", 0))
            et = float(child.get("end_time", 0))
            if (st, et) in leaf_map:
                indices.append(leaf_map[(st, et)])
        indices.sort()
        if not indices:
            return []
        if budget >= len(indices):
            return indices
        step = len(indices) / budget
        return [indices[int(j * step)] for j in range(budget)]

    def _count_descendant_leaves(self, node, level_name):
        """Count leaf nodes under a given node."""
        level_num = int(level_name.split("_")[1])
        if level_num == 1:
            return len([c for c in node.get("children", []) if c.get("summary")])
        count = 0
        for child in node.get("children", []):
            count += self._count_descendant_leaves(child, f"Level_{level_num - 1}")
        return count if count > 0 else 1

    def _get_descendant_leaf_indices(self, node, level_name, leaf_map):
        """Get sorted list of leaf indices under a node."""
        level_num = int(level_name.split("_")[1])
        if level_num == 1:
            indices = []
            for child in node.get("children", []):
                st = float(child.get("start_time", 0))
                et = float(child.get("end_time", 0))
                if (st, et) in leaf_map:
                    indices.append(leaf_map[(st, et)])
            return sorted(indices)
        indices = []
        for child in node.get("children", []):
            indices.extend(
                self._get_descendant_leaf_indices(child, f"Level_{level_num - 1}", leaf_map)
            )
        return sorted(indices)

    # ==================== Localization ====================

    def _localize(self, tree, question, options, coarse_context,
                  excluded_periods, search_hint=None):
        """Find relevant time intervals using tree structure."""
        if self.localize_mode == "key_elements":
            return self._localize_key_elements(tree, question, options, excluded_periods)
        elif self.localize_mode == "llm_select":
            return self._localize_llm_select(
                question, options, coarse_context, excluded_periods,
                search_hint=search_hint,
            )
        elif self.localize_mode == "llm_index":
            return self._localize_llm_index(
                tree, question, options, excluded_periods,
                search_hint=search_hint,
            )
        elif self.localize_mode == "combined":
            # Try key_elements first, fall back to llm_select
            intervals = self._localize_key_elements(tree, question, options, excluded_periods)
            if not intervals:
                intervals = self._localize_llm_select(
                    question, options, coarse_context, excluded_periods,
                    search_hint=search_hint,
                )
            return intervals
        return []

    def _localize_key_elements(self, tree, question, options, excluded_periods):
        """Use TreeFilter key_elements matching to find relevant intervals."""
        tree_filter = self.components.get("tree_filter")
        query_analyzer = self.components.get("query_analyzer")

        if not tree_filter:
            print("    [Localize] No tree_filter component")
            return []

        # Extract cues from question + options
        if query_analyzer:
            analysis = query_analyzer.analyze(question, options)
            cues = analysis.get("cues", [])
        else:
            # Simple fallback: split question into words
            cues = [w for w in question.lower().split()
                    if len(w) > 3 and w not in {
                        "what", "when", "where", "which", "does", "this",
                        "that", "they", "them", "with", "from", "have",
                        "been", "into", "about", "video",
                    }]

        if not cues:
            return []

        filtered = tree_filter.build(tree, cues)
        priority_leaves = filtered.get("priority_leaves", [])

        if not priority_leaves:
            # Fallback: use all_leaves sorted by score
            all_leaves = filtered.get("all_leaves", [])
            priority_leaves = sorted(all_leaves, key=lambda x: -x["score"])

        # Extract raw intervals, skip excluded
        raw_intervals = []
        for entry in priority_leaves:
            st = entry.get("start_time")
            et = entry.get("end_time")
            if st is not None and et is not None:
                key = (st, et)
                if key not in excluded_periods:
                    raw_intervals.append((float(st), float(et)))

        if not raw_intervals:
            return []

        # Merge adjacent/overlapping intervals (e.g. [0-30, 30-60] → [0-60])
        # but keep separated intervals separate (e.g. [0-60, 120-180])
        raw_intervals.sort(key=lambda x: x[0])
        merged = [raw_intervals[0]]
        for st, et in raw_intervals[1:]:
            prev_st, prev_et = merged[-1]
            if st <= prev_et:  # overlapping or adjacent
                merged[-1] = (prev_st, max(prev_et, et))
            else:
                merged.append((st, et))

        return merged[:self.max_intervals]

    def _localize_llm_select(self, question, options, coarse_context,
                             excluded_periods, search_hint=None):
        """Use LLM to select relevant time periods from coarse context."""
        llm = self.components.get("llm")
        if not llm:
            return []

        opt_text = "\n".join(
            f"{chr(65+i)}. {o}" for i, o in enumerate(options)
        )

        excluded_str = ""
        if excluded_periods:
            excluded_str = (
                "\n\nAlready explored periods (do NOT select these): "
                + str(sorted(excluded_periods))
            )

        hint_str = ""
        if search_hint:
            hint_str = f"\n\nHint — focus on finding: {search_hint}"

        prompt = self.LOCALIZE_PROMPT.format(
            context=coarse_context,
            question=question,
            options_text=opt_text,
        ) + excluded_str + hint_str

        try:
            result = llm.reason(prompt, max_tokens=300)
        except Exception as e:
            print(f"    [Localize LLM] Error: {e}")
            return []

        periods = result.get("periods", [])
        intervals = []
        for p in periods:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                st, et = float(p[0]), float(p[1])
                if (st, et) not in excluded_periods:
                    intervals.append((st, et))

        return intervals

    def _localize_llm_index(self, tree, question, options, excluded_periods,
                            search_hint=None):
        """LLM selects segment indices from numbered list (more robust than free-form time).

        Shows hierarchical context: Level_2+ summaries as section headers,
        Level_1 children (leaves) as numbered selectable segments.
        """
        llm = self.components.get("llm")
        if not llm:
            return []

        # Build numbered segment list with hierarchical context
        segments = []  # flat list: (start, end, summary)
        seg_lines = []
        idx = 0

        # Get Level_1 nodes sorted by time
        l1_nodes = []
        for l1_node in tree.get("Level_1", []):
            l1_start, l1_end = self._get_time_range(l1_node)
            l1_nodes.append((l1_start, l1_end, l1_node))
        l1_nodes.sort(key=lambda x: x[0])

        # Show Level_2+ summaries as section context
        level_names = sorted(
            [k for k in tree.keys() if k.startswith("Level_") and k != "Level_1"],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        )
        if level_names:
            section_headers = []
            for level_name in level_names[:2]:  # Top 2 levels only
                for node in tree.get(level_name, []):
                    start, end = self._get_time_range(node)
                    summary = node.get("summary", "")
                    if summary:
                        section_headers.append(
                            f"  [{level_name}] {start:.0f}s-{end:.0f}s: {summary}"
                        )
            if section_headers:
                seg_lines.append("=== Video Structure ===")
                seg_lines.extend(sorted(section_headers, key=lambda x: x))
                seg_lines.append("")

        seg_lines.append("=== Selectable Segments ===")
        for l1_start, l1_end, l1_node in l1_nodes:
            l1_summary = l1_node.get("summary", "")
            if l1_summary:
                seg_lines.append(f"--- Section {l1_start:.0f}s-{l1_end:.0f}s: {l1_summary} ---")

            for child in l1_node.get("children", []):
                st = float(child.get("start_time", 0))
                et = float(child.get("end_time", 0))
                summary = child.get("summary", "")
                if summary and (st, et) not in excluded_periods:
                    idx += 1
                    segments.append((st, et, summary))
                    seg_lines.append(f"  [{idx}] {st:.0f}s-{et:.0f}s: {summary}")

        if not segments:
            return []

        seg_text = "\n".join(seg_lines)

        opt_text = "\n".join(
            f"{chr(65+i)}. {o}" for i, o in enumerate(options)
        )

        hint_str = ""
        if search_hint:
            hint_str = f"\nHint — focus on finding: {search_hint}\n"

        prompt = (
            f"{seg_text}\n\n"
            f"Question: {question}\n{opt_text}\n{hint_str}\n"
            f"Which segments are most relevant to answering this question? "
            f"Select up to {self.max_intervals} segment numbers.\n"
            f'Output ONLY valid JSON: {{"indices": [1, 5, 12]}}'
        )

        try:
            result = llm.reason(prompt, max_tokens=200)
        except Exception as e:
            print(f"    [Localize Index] Error: {e}")
            return []

        indices = result.get("indices", [])
        raw_intervals = []
        for idx in indices:
            try:
                i = int(idx) - 1  # 1-indexed to 0-indexed
                if 0 <= i < len(segments):
                    raw_intervals.append((segments[i][0], segments[i][1]))
            except (TypeError, ValueError):
                continue

        if not raw_intervals:
            return []

        # Merge adjacent/overlapping intervals
        raw_intervals.sort(key=lambda x: x[0])
        merged = [raw_intervals[0]]
        for st, et in raw_intervals[1:]:
            prev_st, prev_et = merged[-1]
            if st <= prev_et:
                merged[-1] = (prev_st, max(prev_et, et))
            else:
                merged.append((st, et))

        return merged[:self.max_intervals]

    # ==================== Visual Observation ====================

    def _visual_observe(self, video_path, intervals, question, options,
                        coarse_context, prev_observations):
        """Load frames and observe with VLM."""
        frame_loader = self.components.get("frame_loader")
        if not frame_loader:
            print("    [Visual] No frame_loader component")
            return None

        try:
            frames_np, frame_secs = frame_loader.load(
                video_path, intervals, max_frames=self.max_frames,
            )
        except Exception as e:
            print(f"    [Visual] Frame loading error: {e}")
            return None

        if frames_np is None or len(frames_np) == 0:
            print("    [Visual] No frames loaded")
            return None

        print(f"    [Visual] Loaded {len(frames_np)} frames from {len(intervals)} intervals")

        if self.vlm_mode == "direct":
            return self._vlm_direct(
                frames_np, frame_secs, intervals, question, options,
                coarse_context, prev_observations,
            )
        elif self.vlm_mode == "caption":
            return self._vlm_caption(
                frames_np, frame_secs, intervals, question, options,
            )
        return None

    def _vlm_direct(self, frames_np, frame_secs, intervals, question,
                    options, context, prev_observations):
        """VLM directly answers the question using frames."""
        vision_vlm = self.components.get("vision_vlm")
        if not vision_vlm:
            return None

        # Build hop history from previous observations
        hop_history = []
        for i, obs in enumerate(prev_observations):
            if obs.get("observation"):
                hop_history.append({
                    "hop": i + 1,
                    "type": "frame_inference",
                    "observation": obs["observation"],
                    "answer": obs.get("answer", "N/A"),
                    "confidence": obs.get("confidence", "N/A"),
                })

        try:
            vlm_result = vision_vlm.infer(
                frames_np, context, question, options,
                hop_history=hop_history if hop_history else None,
                max_tokens=300,
            )
        except Exception as e:
            print(f"    [VLM Direct] Error: {e}")
            return None

        obs = vlm_result.get("observation", "")
        answer = vlm_result.get("answer", "")
        confidence = vlm_result.get("confidence", "low")

        print(f"    [VLM Direct] answer={answer} conf={confidence}")

        return {
            "type": "direct",
            "intervals": intervals,
            "n_frames": len(frames_np),
            "frame_seconds": frame_secs,
            "observation": obs,
            "answer": answer,
            "confidence": confidence,
        }

    def _vlm_caption(self, frames_np, frame_secs, intervals, question,
                     options):
        """VLM captions the frames (question-aware)."""
        vision_vlm = self.components.get("vision_vlm")
        if not vision_vlm:
            return None

        # Use a captioning prompt
        caption_context = (
            f"Describe the video frames in detail. "
            f"Pay attention to details that might help answer: {question}"
        )

        try:
            vlm_result = vision_vlm.infer(
                frames_np, caption_context, question, options,
                max_tokens=500,
            )
        except Exception as e:
            print(f"    [VLM Caption] Error: {e}")
            return None

        obs = vlm_result.get("observation", "")
        raw = vlm_result.get("raw_response", obs)
        print(f"    [VLM Caption] {len(obs)} chars observation")

        return {
            "type": "caption",
            "intervals": intervals,
            "n_frames": len(frames_np),
            "frame_seconds": frame_secs,
            "observation": obs or raw,
            "answer": vlm_result.get("answer", ""),
            "confidence": vlm_result.get("confidence", "low"),
        }

    # ==================== Leaf Caption Extraction ====================

    def _get_leaf_captions(self, tree, intervals):
        """Extract raw captions from leaf nodes overlapping with intervals.

        Args:
            tree: memory tree dict
            intervals: [(start_sec, end_sec), ...]

        Returns:
            list of {"time": (start, end), "caption": str}
        """
        if not intervals or "Level_1" not in tree:
            return []

        results = []
        seen = set()

        for l1_node in tree.get("Level_1", []):
            for child in l1_node.get("children", []):
                if "start_time" not in child:
                    continue
                st = float(child["start_time"])
                et = float(child["end_time"])

                # Check overlap with any requested interval
                for iv_start, iv_end in intervals:
                    if st < iv_end and et > iv_start:
                        key = (st, et)
                        if key not in seen:
                            seen.add(key)
                            caption = child.get("caption", "")
                            if caption:
                                results.append({
                                    "time": (st, et),
                                    "caption": caption,
                                })
                        break

        results.sort(key=lambda x: x["time"][0])
        return results

    # ==================== Context Building ====================

    def _build_enriched_context(self, coarse_context, visual_observations,
                                leaf_captions=None):
        """Combine coarse context + leaf raw captions + visual observations."""
        parts = [coarse_context]

        # Add leaf raw captions (detailed text from memory tree)
        if leaf_captions:
            caption_lines = []
            for entry in leaf_captions:
                st, et = entry["time"]
                caption_lines.append(
                    f"[{st:.0f}s-{et:.0f}s] {entry['caption']}"
                )
            parts.append(
                "\n=== Detailed Descriptions (selected segments) ===\n"
                + "\n\n".join(caption_lines)
            )

        # Add VLM visual observations
        for i, obs in enumerate(visual_observations):
            intervals_str = ", ".join(
                f"[{iv[0]:.0f}s-{iv[1]:.0f}s]" for iv in obs.get("intervals", [])
            )
            obs_text = obs.get("observation", "")
            if obs_text:
                parts.append(
                    f"\n=== Visual Observation {i+1} ({intervals_str}, "
                    f"{obs['n_frames']} frames) ===\n{obs_text}"
                )
        return "\n".join(parts)

    # ==================== Judge ====================

    def _judge_answer(self, context, question, opt_text):
        """Use LLM to answer the question given context."""
        llm = self.components.get("llm")
        if not llm:
            return {"answer": None, "confidence": "low"}

        prompt = self.PHASE0_PROMPT.format(
            context=context,
            question=question,
            options_text=opt_text,
        )

        try:
            verdict = llm.reason(prompt, max_tokens=400)
        except Exception as e:
            print(f"    [Judge] Error: {e}")
            return {"answer": None, "confidence": "low"}

        if not isinstance(verdict, dict):
            return {"answer": None, "confidence": "low"}

        # Normalize answer
        answer = verdict.get("answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            verdict["answer"] = m.group(0) if m else None

        conf = str(verdict.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        verdict["confidence"] = conf

        return verdict

    # ==================== Helpers ====================

    def _is_confident(self, confidence):
        """Check if confidence meets threshold. 'none' = never confident (always Phase 2)."""
        if self.confidence_threshold == "none":
            return False
        levels = {"high": 3, "medium": 2, "low": 1}
        threshold = levels.get(self.confidence_threshold, 2)
        current = levels.get(confidence, 1)
        return current >= threshold

    def _check_correct(self, pred, answer):
        if pred is None or answer is None:
            return False
        return pred.upper().strip() == str(answer).upper().strip()

    def _safe_time_range(self, time_range):
        """Validate and normalize time_range to [float, float] or None."""
        if time_range is None:
            return None
        if isinstance(time_range, str):
            try:
                time_range = json.loads(time_range)
            except (json.JSONDecodeError, ValueError):
                return None
        if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
            try:
                return [float(time_range[0]), float(time_range[1])]
            except (TypeError, ValueError):
                return None
        return None

    def _get_time_range(self, node):
        """Extract (start, end) from node's time_segments."""
        segs = node.get("time_segments", [])
        flat = []
        for s in segs:
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                flat.extend([float(s[0]), float(s[1])])
            elif isinstance(s, (int, float)):
                flat.append(float(s))
        if not flat:
            return 0.0, 0.0
        return min(flat), max(flat)
