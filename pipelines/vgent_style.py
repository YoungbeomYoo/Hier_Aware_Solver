from __future__ import annotations

"""
Vgent-Style Structured Reasoning Pipeline

Replaces the hop loop with Vgent-style structured reasoning:
  Phase 0: Coarse Answer (all-level summary → force answer → early return if confident)
  Step 1: Query Analysis (extract cues + question type)
  Step 2: Node Retrieval (tree_filter → top-N leaves)
  Step 3: Sub-question Generation (decompose question into yes/no sub-Qs)
  Step 4: Sub-question Verification (per-leaf verification, text or visual)
  Step 5: Filtering + Aggregation (hard negative removal + info summarization)
  Step 6: Final Answer (filtered context + aggregation summary → forced answer)
"""

import os
import re
from pipelines.base import BasePipeline


class VgentStylePipeline(BasePipeline):
    """Vgent-style structured reasoning pipeline.

    Required components:
        query_analyzer, tree_filter, judge,
        sub_question_generator, sub_question_verifier, info_aggregator

    Optional components:
        semantic_matcher, vision_vlm, frame_loader
    """

    # Reuse the same PHASE0_PROMPT as TreeSearchPipeline
    PHASE0_PROMPT = """The following provides descriptions of what's shown in the video during different time periods:

{context}

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
    "reasoning": "Your reasoning about your judgment. You need to ensure and check that your reasoning must be able to absolutely support your answer.",
    "missing_info": null,
    "search_direction": null
}}
- You MUST provide an answer. Do not refuse."""

    # Enhanced final answer prompt with aggregation context
    FINAL_PROMPT = """The following provides descriptions of what's shown in the video during different time periods:

{context}

Additional analysis from structured reasoning:
{aggregation_summary}

Now, a question has been raised regarding the content descriptions of this video.
{question}

{options_text}

Please read and understand the given video content, the additional analysis, and the question in depth. Strictly based on the video content, select the single best option. You must choose an option from these provided options. The answer you provide must include the English letters of the options [A, B, C, D].

Please note that if an ordinal number appears in the provided question, in most cases, the meaning of this ordinal number is not related to the ordinal of the provided time period. You need to focus on analyzing the meaning of this ordinal number.

Please output ONLY valid JSON in a strictly standardized format:
{{
    "answerable": true,
    "answer": "A" or "B" or "C" or "D",
    "confidence": "high" or "medium" or "low",
    "reasoning": "Your reasoning about your judgment. You need to ensure and check that your reasoning must be able to absolutely support your answer.",
    "missing_info": null,
    "search_direction": null
}}
- You MUST provide an answer. Do not refuse."""

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")
        time_ref = question_data.get("time_reference", "")
        tree = memory.get("streaming_memory_tree", {})

        # Config
        n_retrieval = self.config.get("n_retrieval", 20)
        n_refine = self.config.get("n_refine", 5)
        verify_mode = self.config.get("verify_mode", "text")
        max_frames_per_leaf = self.config.get("max_frames_per_leaf", 8)

        # Components
        analyzer = self.components["query_analyzer"]
        tree_filter = self.components["tree_filter"]
        judge = self.components["judge"]
        sub_q_gen = self.components["sub_question_generator"]
        sub_q_ver = self.components["sub_question_verifier"]
        aggregator = self.components["info_aggregator"]
        semantic_matcher = self.components.get("semantic_matcher")

        # ============================================================
        # PHASE 0: Coarse overview answer attempt
        # ============================================================
        phase0_verdict = None
        coarse_ctx = ""
        if self.config.get("coarse_first", False) and tree:
            phase0_verdict, coarse_ctx = self._phase0_coarse_answer(
                tree, question, options, judge,
            )
            print(f"    [Phase0] answerable={phase0_verdict.get('answerable')} | "
                  f"conf={phase0_verdict.get('confidence')} | "
                  f"ans={phase0_verdict.get('answer')}")

            if (phase0_verdict.get("answerable")
                    and phase0_verdict.get("confidence") == "high"
                    and phase0_verdict.get("answer")):
                pred = phase0_verdict["answer"]
                correct = self.adapter.check_correct(pred, answer)
                print(f"    [Phase0] Early return: pred={pred} correct={correct}")
                return {
                    "pred": pred,
                    "answer": answer,
                    "correct": correct,
                    "method": "phase0_coarse",
                    "confidence": "high",
                    "phase": "phase0",
                    "used_visual": False,
                    "traversal_log": [],
                    "phase0_verdict": phase0_verdict,
                }

        # ============================================================
        # STEP 1: Query Analysis
        # ============================================================
        analysis = analyzer.analyze(question, options, time_ref)
        cues = analysis["cues"]
        target_fields = analysis.get("target_fields", ["summary"])
        print(f"    [Step1] cues={cues[:4]}")

        # ============================================================
        # STEP 1.5: Semantic matching (optional)
        # ============================================================
        semantic_scores = None
        if semantic_matcher:
            q_elements = semantic_matcher.extract_question_elements(
                question, options, cues,
            )
            semantic_scores = semantic_matcher.select_top_nodes(
                q_elements, tree, level="Level_1",
            )
            print(f"    [Semantic] top {len(semantic_scores['selected_indices'])}/"
                  f"{semantic_scores['total_nodes']} L1 nodes")

        # ============================================================
        # STEP 2: Node Retrieval (top-N leaves)
        # ============================================================
        filtered = tree_filter.build(
            tree, cues, target_fields,
            semantic_scores=semantic_scores,
        )
        all_leaves = self._collect_all_leaves(tree)
        priority_leaves = filtered.get("priority_leaves", [])

        # Select top-N by priority score
        retrieved_leaves = priority_leaves[:n_retrieval]
        if not retrieved_leaves:
            retrieved_leaves = all_leaves[:n_retrieval]

        print(f"    [Step2] Retrieved {len(retrieved_leaves)} leaves "
              f"(priority={len(priority_leaves)}, total={len(all_leaves)})")

        # ============================================================
        # STEP 3: Sub-question Generation
        # ============================================================
        sub_questions = sub_q_gen.generate(question, options)
        if not sub_questions:
            print("    [Step3] Sub-question generation failed, fallback to flat")
            return self._fallback_flat(tree, question, options, answer, judge,
                                       phase0_verdict)

        print(f"    [Step3] Sub-questions: {sub_questions}")

        # ============================================================
        # STEP 4: Sub-question Verification
        # ============================================================
        if verify_mode == "visual":
            video_root = self.config.get("video_root", "")
            video_path = os.path.join(video_root, f"{video_id}.mp4")
            check_results = sub_q_ver.verify_visual(
                retrieved_leaves, sub_questions, video_path,
                max_frames=max_frames_per_leaf,
            )
        else:
            check_results = sub_q_ver.verify_text(
                retrieved_leaves, sub_questions,
            )

        print(f"    [Step4] Verified {len(check_results)} leaves")

        # ============================================================
        # STEP 5: Filtering + Aggregation
        # ============================================================
        selected_indices = sub_q_ver.filter_and_rank(
            check_results, sub_questions, n_refine=n_refine,
        )

        if not selected_indices:
            print("    [Step5] No positive leaves after filtering, fallback to flat")
            return self._fallback_flat(tree, question, options, answer, judge,
                                       phase0_verdict)

        # Aggregation
        aggregation_summary = aggregator.aggregate(
            sub_questions, check_results,
            retrieved_leaves, selected_indices,
            question, options,
        )

        # ============================================================
        # STEP 6: Final Answer
        # ============================================================
        # Build context from selected leaves
        context_lines = []
        for leaf_idx in selected_indices:
            leaf = retrieved_leaves[leaf_idx]
            st = float(leaf.get("start_time", 0))
            et = float(leaf.get("end_time", 0))
            summary = leaf.get("summary", "") or leaf.get("caption", "")
            context_lines.append(f"[{st:.0f}s-{et:.0f}s] {summary}")

        refined_context = "=== Relevant Video Segments ===\n" + "\n".join(context_lines)

        # Also include coarse context for broader perspective
        if coarse_ctx:
            full_context = coarse_ctx + "\n\n" + refined_context
        else:
            full_context = refined_context

        verdict = self._final_answer(
            full_context, aggregation_summary, question, options, judge,
        )

        pred = verdict.get("answer", "A") if verdict.get("answer") else "A"
        correct = self.adapter.check_correct(pred, answer)

        print(f"    [Step6] pred={pred} correct={correct} "
              f"conf={verdict.get('confidence')}")

        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": "vgent_structured",
            "confidence": verdict.get("confidence", "low"),
            "phase": "structured_reasoning",
            "used_visual": verify_mode == "visual",
            "traversal_log": [],
            "n_retrieved": len(retrieved_leaves),
            "n_verified": len(check_results),
            "n_selected": len(selected_indices),
            "sub_questions": sub_questions,
            "aggregation_summary": aggregation_summary,
            "phase0_verdict": phase0_verdict,
            "cues": cues,
        }

    # ==================== Phase 0: Coarse Answer ====================

    def _phase0_coarse_answer(self, tree, question, options, judge):
        """Phase 0: all-level summary → forced answer attempt."""
        level_names = sorted(
            [k for k in tree.keys() if k.startswith("Level_")],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        )

        coarse_parts = []
        for level_name in level_names:
            nodes = tree.get(level_name, [])
            for node in nodes:
                summary = node.get("summary", "")
                if not summary:
                    continue
                segs = node.get("time_segments", [])
                start, end = self._get_time_range(segs)

                seg_strs = []
                for s in segs:
                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                        seg_strs.append("%.0fs-%.0fs" % (float(s[0]), float(s[1])))
                time_range_str = (
                    ", ".join(seg_strs)
                    if seg_strs
                    else "%.0fs-%.0fs" % (start, end)
                )

                ke = node.get("key_elements", {})
                ke_brief = ""
                for field in ["actions", "objects", "persons"]:
                    vals = ke.get(field, [])
                    if vals:
                        ke_brief += " | %s: %s" % (
                            field,
                            ", ".join(str(v) for v in vals[:5]),
                        )

                coarse_parts.append({
                    "level": level_name,
                    "start": start,
                    "end": end,
                    "text": "[%s] [%s] %s%s" % (
                        level_name, time_range_str, summary, ke_brief,
                    ),
                })

        coarse_parts.sort(key=lambda x: x["start"])
        coarse_context = (
            "=== Video Overview (all segments) ===\n"
            + "\n".join(p["text"] for p in coarse_parts)
        )

        opt_text = "\n".join(
            "%s. %s" % (chr(65 + i), o) for i, o in enumerate(options)
        )
        prompt = self.PHASE0_PROMPT.format(
            context=coarse_context,
            question=question,
            options_text=opt_text,
        )

        default = {
            "answerable": False, "answer": None, "confidence": "low",
            "reasoning": "Phase 0 failed", "missing_info": None,
            "search_direction": None,
        }

        try:
            verdict = judge.llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"      [Phase0] Error: {e}")
            return default, coarse_context

        if not isinstance(verdict, dict):
            return default, coarse_context

        answer = verdict.get("answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            verdict["answer"] = m.group(0) if m else None

        conf = str(verdict.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        verdict["confidence"] = conf
        verdict["answerable"] = bool(verdict.get("answerable", False))

        return verdict, coarse_context

    # ==================== Final Answer ====================

    def _final_answer(self, context, aggregation_summary, question, options, judge):
        """Generate final answer with context + aggregation summary."""
        opt_text = "\n".join(
            "%s. %s" % (chr(65 + i), o) for i, o in enumerate(options)
        )
        prompt = self.FINAL_PROMPT.format(
            context=context,
            aggregation_summary=aggregation_summary,
            question=question,
            options_text=opt_text,
        )
        default = {
            "answerable": False, "answer": None, "confidence": "low",
            "reasoning": "Final answer failed",
        }
        try:
            verdict = judge.llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"      [FinalAnswer] Error: {e}")
            return default

        if not isinstance(verdict, dict):
            return default

        answer = verdict.get("answer")
        if answer and isinstance(answer, str):
            m = re.search(r"[ABCD]", answer.upper())
            verdict["answer"] = m.group(0) if m else None

        conf = str(verdict.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        verdict["confidence"] = conf
        return verdict

    # ==================== Fallback ====================

    def _fallback_flat(self, tree, question, options, answer, judge,
                       phase0_verdict):
        """Fallback: flat baseline when structured reasoning fails."""
        flat_ctx = self._flat_baseline_context(tree)
        verdict = self._forced_answer(flat_ctx, question, options, judge)
        pred = verdict.get("answer", "A") if verdict.get("answer") else "A"
        correct = self.adapter.check_correct(pred, answer)
        print(f"    [Fallback] pred={pred} correct={correct}")
        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": "vgent_fallback_flat",
            "confidence": verdict.get("confidence", "low"),
            "phase": "fallback",
            "used_visual": False,
            "traversal_log": [],
            "phase0_verdict": phase0_verdict,
        }

    def _flat_baseline_context(self, tree):
        """Flat: Level_1 children (leaf) summaries in time order."""
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

    def _forced_answer(self, context, question, options, judge):
        """Force prompt to produce answer from any context."""
        opt_text = "\n".join(
            "%s. %s" % (chr(65 + i), o) for i, o in enumerate(options)
        )
        prompt = self.PHASE0_PROMPT.format(
            context=context,
            question=question,
            options_text=opt_text,
        )
        default = {
            "answerable": False, "answer": None, "confidence": "low",
            "reasoning": "Forced answer failed",
        }
        try:
            verdict = judge.llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"      [ForcedAnswer] Error: {e}")
            return default

        if not isinstance(verdict, dict):
            return default

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

    @staticmethod
    def _collect_all_leaves(tree) -> list[dict]:
        """Collect all leaf nodes from tree, sorted by time."""
        leaves = []
        for l1_node in tree.get("Level_1", []):
            for child in l1_node.get("children", []):
                leaves.append(child)
        leaves.sort(key=lambda x: float(x.get("start_time", 0)))
        return leaves

    @staticmethod
    def _get_time_range(time_segments):
        """Extract (start, end) from time_segments list."""
        if not time_segments:
            return 0.0, 0.0
        flat = []
        for s in time_segments:
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                flat.extend([float(s[0]), float(s[1])])
            else:
                flat.append(float(s))
        if not flat:
            return 0.0, 0.0
        return min(flat), max(flat)
