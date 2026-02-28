"""
Cognitive Pipeline — 인지과학 기반 4단계 파이프라인

인지과학적 근거:
  Stage 1: SAM (Search of Associative Memory) — 질의 분해 + 범위 결정
  Stage 2: Schema-based Retrieval (Bartlett) + Hippocampal Indexing — 구조적 검색
  Stage 3: Prediction Error → Backtracking — 불확실성 기반 시각적 역추적
  Stage 4: Elimination-by-Aspects (Tversky) — 소거법 기반 최종 답변

Flow:
  QueryAnalyzer → Track Split:
    Track A (explicit time): MetadataFilter.filter_by_time → SpreadingActivation
    Track B (semantic):      MetadataFilter.filter_by_fields → LLM refinement
  → Context Assembly → UncertaintyChecker →
    certain/likely → EliminationReasoner → done
    uncertain/insufficient → FrameLoader → VLM → EliminationReasoner → done
"""

import os
import gc
import re
from pipelines.base import BasePipeline
from components.answer_parser import extract_choice_letter


class CognitivePipeline(BasePipeline):
    """4-stage Cognitive Science-inspired solver pipeline.

    Required components:
        - query_analyzer: QueryAnalyzer
        - flattener: LeafFlattener
        - metadata_filter: MetadataTargetedFilter
        - spreading_activation: SpreadingActivation
        - uncertainty_checker: UncertaintyChecker
        - elimination_reasoner: EliminationReasoner
        - formatter: MemoryContextFormatter
        - fallback: ForcedAnswerFallback

    Optional components:
        - llm: TextOnlyLLM (for LLM-powered query analysis)
        - leaf_selector: LLMLeafSelector (for Track B refinement)
        - frame_loader: TargetedFrameLoader
        - vision_vlm: VisionVLM
        - coverage: CoverageAnalyzer
    """

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        # --- Unpack components ---
        analyzer = self.components["query_analyzer"]
        flattener = self.components["flattener"]
        meta_filter = self.components["metadata_filter"]
        activation = self.components["spreading_activation"]
        uncertainty = self.components["uncertainty_checker"]
        eliminator = self.components["elimination_reasoner"]
        formatter = self.components["formatter"]
        fallback = self.components["fallback"]

        leaf_selector = self.components.get("leaf_selector")
        frame_loader = self.components.get("frame_loader")
        vision_vlm = self.components.get("vision_vlm")
        coverage_analyzer = self.components.get("coverage")

        max_frames = self.config.get("max_frames", 32)
        max_visual_retries = self.config.get("max_visual_retries", 1)
        include_siblings = self.config.get("include_siblings", False)
        leaf_budget = self.config.get("leaf_budget", 15)

        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")
        time_ref = question_data.get("time_reference", "")

        tree = memory.get("streaming_memory_tree", {})
        hop_history = []
        used_visual = False

        # ==============================================================
        # STAGE 1: Query Decoupling & Scope Parsing (SAM)
        # ==============================================================
        analysis = analyzer.analyze(question, options, time_ref)
        q_type = analysis["question_type"]
        target_fields = analysis["target_fields"]
        secondary_fields = analysis["secondary_fields"]
        cues = analysis["cues"]
        has_explicit_time = analysis["has_explicit_time"]
        all_time_ranges = analysis["all_time_ranges"]
        requires_visual = analysis["requires_visual"]

        print(f"    [Stage 1] type={q_type} | fields={target_fields} | "
              f"cues={cues[:3]} | time={has_explicit_time}")

        hop_history.append({
            "stage": 1, "type": "query_analysis",
            "question_type": q_type,
            "target_fields": target_fields,
            "cues": cues,
            "has_explicit_time": has_explicit_time,
            "requires_visual": requires_visual,
        })

        # ==============================================================
        # STAGE 2: Metadata Cross-Filtering + Context Assembly
        # ==============================================================
        all_leaves = flattener.flatten(tree)
        print(f"    [Stage 2] Total leaves: {len(all_leaves)}")

        # --- Track Split ---
        if has_explicit_time and all_time_ranges:
            # Track A: Direct time indexing + Spreading Activation
            track = "A"
            time_matched = meta_filter.filter_by_time(all_leaves, all_time_ranges)

            if time_matched:
                # Also apply field filter to time-matched for precision
                if cues:
                    field_refined, _ = meta_filter.filter_by_fields(
                        time_matched, cues, target_fields, secondary_fields
                    )
                    target_leaves = field_refined if field_refined else time_matched
                else:
                    target_leaves = time_matched
            else:
                # Time didn't match anything → fall through to Track B
                track = "B_fallback"
                target_leaves, _ = meta_filter.filter_by_fields(
                    all_leaves, cues, target_fields, secondary_fields
                )
        else:
            # Track B: Semantic field-based retrieval
            track = "B"
            target_leaves, _ = meta_filter.filter_by_fields(
                all_leaves, cues, target_fields, secondary_fields
            )

        # --- LLM refinement for Track B when too many candidates ---
        if track.startswith("B") and leaf_selector and len(target_leaves) > leaf_budget:
            print(f"    [Stage 2] LLM refining {len(target_leaves)} → budget {leaf_budget}")
            selected_ids = leaf_selector.select(
                target_leaves, cues, question, options, hop_history,
                budget=leaf_budget
            )
            target_leaves = [target_leaves[i] for i in selected_ids]

        # Cap to avoid context overflow
        if len(target_leaves) > leaf_budget * 2:
            target_leaves = target_leaves[:leaf_budget * 2]

        print(f"    [Stage 2] Track {track} → {len(target_leaves)} target leaves")

        hop_history.append({
            "stage": 2, "type": "metadata_filter",
            "track": track,
            "target_count": len(target_leaves),
            "total_leaves": len(all_leaves),
        })

        # --- Bottom-Up Spreading Activation ---
        act_result = activation.activate(tree, target_leaves, include_siblings)
        activated_context = act_result["activated_context"]

        # Build context: activated hierarchy + leaf details
        leaf_context = formatter.format_leaf_batch(target_leaves)
        full_context = ""
        if activated_context:
            full_context += activated_context + "\n\n"
        full_context += "=== Matched Segment Details ===\n" + leaf_context

        # Temporal reasoning shortcut: if question asks for Nth event
        nth_match = re.search(
            r"(?:the\s+)?(first|second|third|fourth|fifth|last|next)\s+",
            question.lower()
        )
        if nth_match and q_type == "temporal":
            ordinal_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
            ordinal = nth_match.group(1)
            n = ordinal_map.get(ordinal)
            if n:
                nth_entry = meta_filter.get_nth_event(target_leaves, n)
                if nth_entry:
                    nth_context = formatter.format_leaf_batch([nth_entry])
                    full_context += f"\n\n=== Nth Event (ordinal={ordinal}) ===\n{nth_context}"

        # ==============================================================
        # STAGE 3: Uncertainty-Driven Visual Backtracking
        # ==============================================================
        unc = uncertainty.assess(full_context, question, options, hop_history)
        confidence = unc["confidence"]
        needs_visual = unc["needs_visual"]
        pre_elimination = unc.get("elimination", {})

        print(f"    [Stage 3] confidence={confidence} | needs_visual={needs_visual}")

        hop_history.append({
            "stage": 3, "type": "uncertainty_check",
            "confidence": confidence,
            "reasoning": unc.get("reasoning", ""),
            "needs_visual": needs_visual,
            "visual_reason": unc.get("visual_reason", ""),
        })

        # Early answer from uncertainty check (if certain/likely with answer)
        unc_answer = unc.get("answer")

        # --- Visual Backtracking ---
        visual_observation = ""
        if needs_visual and frame_loader and vision_vlm:
            video_path = self.adapter.get_video_path(video_id)
            if video_path and os.path.exists(video_path):
                visual_result = self._visual_backtrack(
                    target_leaves, unc, video_path,
                    frame_loader, vision_vlm, formatter,
                    question, options, full_context, hop_history,
                    max_frames, max_visual_retries,
                )
                used_visual = visual_result["used"]
                if visual_result.get("observation"):
                    visual_observation = visual_result["observation"]
                    full_context += f"\n\n=== Visual Observation ===\n{visual_observation}"
                if visual_result.get("answer"):
                    unc_answer = visual_result["answer"]
                    confidence = visual_result.get("confidence", confidence)

        # If certain/likely + answer and no visual override needed
        if confidence in ("certain", "likely") and unc_answer:
            # Validate with quick elimination check
            if not needs_visual or used_visual:
                result_answer = unc_answer
                method = f"cognitive_stage3_{confidence}"
                print(f"    [Stage 3] Direct answer: {result_answer} ({confidence})")

                correct = self.adapter.check_correct(result_answer, answer)
                return self._build_result(
                    result_answer, answer, correct, method, confidence,
                    hop_history, used_visual, time_ref, coverage_analyzer,
                    target_leaves, all_leaves, q_type, question_data,
                )

        # ==============================================================
        # STAGE 4: Elimination-Based Answer Selection
        # ==============================================================
        elim_result = eliminator.eliminate(
            full_context, question, options,
            hop_history=hop_history,
            pre_elimination=pre_elimination if pre_elimination.get("eliminated") else None,
        )
        pred = elim_result["answer"]
        elim_confidence = elim_result["confidence"]
        method = f"cognitive_stage4_{elim_result['method']}"

        print(f"    [Stage 4] answer={pred} | confidence={elim_confidence} | "
              f"eliminated={elim_result['eliminated']} | method={elim_result['method']}")

        hop_history.append({
            "stage": 4, "type": "elimination",
            "answer": pred,
            "confidence": elim_confidence,
            "eliminated": elim_result["eliminated"],
            "remaining": elim_result["remaining"],
            "reasoning": elim_result["reasoning"],
        })

        # --- Fallback if no answer ---
        if not pred:
            print(f"    [Fallback] Forcing answer...")
            pred = fallback.force_answer(full_context, question, options)
            method = "cognitive_fallback"

        correct = self.adapter.check_correct(pred, answer)

        return self._build_result(
            pred, answer, correct, method, elim_confidence,
            hop_history, used_visual, time_ref, coverage_analyzer,
            target_leaves, all_leaves, q_type, question_data,
        )

    def _visual_backtrack(
        self,
        target_leaves: list[dict],
        unc_result: dict,
        video_path: str,
        frame_loader,
        vision_vlm,
        formatter,
        question: str,
        options: list[str],
        memory_context: str,
        hop_history: list,
        max_frames: int,
        max_retries: int,
    ) -> dict:
        """VideoLucy-style targeted visual backtracking.

        UncertaintyChecker가 지정한 시간대의 raw video clip만 로딩.
        """
        # Determine which time ranges to load
        visual_ranges = unc_result.get("visual_time_ranges", [])

        if not visual_ranges:
            # Use target leaves' time ranges as fallback
            visual_ranges = []
            for entry in target_leaves[:10]:
                leaf = entry["leaf"]
                if "start_time" in leaf:
                    visual_ranges.append(
                        (float(leaf["start_time"]), float(leaf["end_time"]))
                    )

        if not visual_ranges:
            return {"used": False, "answer": None, "observation": None}

        # Convert nested lists to tuples
        visual_ranges = [(float(r[0]), float(r[1])) for r in visual_ranges]
        visual_ranges.sort()

        result = {"used": False, "answer": None, "observation": None, "confidence": "low"}

        for attempt in range(max_retries):
            try:
                frames_np, frame_secs = frame_loader.load(
                    video_path, visual_ranges, max_frames=max_frames
                )
            except Exception as e:
                print(f"      [Visual] Frame load error: {e}")
                break

            if frames_np is None or len(frame_secs) == 0:
                break

            print(f"      [Visual] attempt={attempt+1} | {len(frame_secs)} frames "
                  f"from {len(visual_ranges)} ranges")

            vlm_result = vision_vlm.infer(
                frames_np, memory_context, question, options, hop_history
            )
            result["used"] = True

            ans = vlm_result.get("answer", "")
            if ans:
                m = re.search(r"[ABCD]", ans.upper())
                result["answer"] = m.group(0) if m else None

            result["observation"] = vlm_result.get("observation", "")
            result["confidence"] = vlm_result.get("confidence", "medium")

            hop_history.append({
                "stage": 3, "type": "visual_backtrack",
                "attempt": attempt + 1,
                "n_frames": len(frame_secs),
                "n_ranges": len(visual_ranges),
                "answer": result["answer"],
                "confidence": result["confidence"],
                "observation": result["observation"][:200] if result["observation"] else "",
                "visual_reason": unc_result.get("visual_reason", ""),
            })

            del frames_np
            gc.collect()

            # If high confidence, stop
            if result["confidence"] == "high" and result["answer"]:
                break

        return result

    def _build_result(
        self, pred, answer, correct, method, confidence,
        hop_history, used_visual, time_ref, coverage_analyzer,
        target_leaves, all_leaves, q_type, question_data,
    ) -> dict:
        """Build standardized result dict."""
        time_coverage = {}
        if time_ref and coverage_analyzer:
            intervals = [
                (float(e["leaf"]["start_time"]), float(e["leaf"]["end_time"]))
                for e in target_leaves if "start_time" in e.get("leaf", {})
            ]
            time_coverage = coverage_analyzer.compute(intervals, time_ref)

        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": method,
            "confidence": confidence,
            "total_stages": max((h.get("stage", 0) for h in hop_history), default=0),
            "used_visual": used_visual,
            "hop_history": hop_history,
            "time_reference": time_ref,
            "time_coverage": time_coverage,
            "total_leaves": len(all_leaves),
            "target_leaves": len(target_leaves),
            "question_type": q_type,
            "dataset_question_type": question_data.get("question_type", []),
        }
