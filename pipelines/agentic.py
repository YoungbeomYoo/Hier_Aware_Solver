"""
Agentic Pipeline — Multi-hop solver with solvability check (LVBench style)

Flow:
  QueryDecomposer → LeafFlattener → RuleBasedFilter →
  Hop Loop:
    LLMLeafSelector → SolvabilityChecker →
      solvable? → done
      needs_depth? → FrameLoader → VisionVLM → confidence high? → done
    → next hop
  → ForcedAnswerFallback → AnswerParser
"""

import os
import gc
import math
import re
from pipelines.base import BasePipeline
from components.answer_parser import extract_choice_letter


class AgenticPipeline(BasePipeline):
    """Multi-hop agentic solver pipeline.

    Required components:
        - decomposer: QueryDecomposer
        - flattener: LeafFlattener
        - rule_filter: RuleBasedFilter
        - leaf_selector: LLMLeafSelector
        - solvability: SolvabilityChecker
        - formatter: MemoryContextFormatter
        - fallback: ForcedAnswerFallback

    Optional components:
        - frame_loader: TargetedFrameLoader (for visual inference)
        - vision_vlm: VisionVLM (for visual inference)
        - coverage: CoverageAnalyzer
        - time_router: TimeRouter (for time_reference pre-filter)
    """

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        decomposer = self.components["decomposer"]
        flattener = self.components["flattener"]
        rule_filter = self.components["rule_filter"]
        leaf_selector = self.components["leaf_selector"]
        solvability = self.components["solvability"]
        formatter = self.components["formatter"]
        fallback = self.components["fallback"]

        frame_loader = self.components.get("frame_loader")
        vision_vlm = self.components.get("vision_vlm")
        coverage_analyzer = self.components.get("coverage")
        time_router = self.components.get("time_router")

        max_hops = self.config.get("max_hops", 5)
        leaf_budget = self.config.get("leaf_budget", 10)
        depth_budget = self.config.get("depth_budget", 5)
        max_frames = self.config.get("max_frames", 32)

        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")
        time_ref = question_data.get("time_reference", "")

        tree = memory.get("streaming_memory_tree", {})

        hop_history = []
        best_answer = None
        best_confidence = "low"
        used_visual = False

        # ============================================================
        # Step 1: Question Decomposition
        # ============================================================
        decomp = decomposer.decompose(question, options)
        cues = decomp["cues"]
        print(f"    [Step 1] Cues: {cues} | Target: {decomp['target_action']}")

        hop_history.append({
            "hop": 1, "type": "decomposition",
            "cues": cues, "target_action": decomp["target_action"],
        })

        # ============================================================
        # Step 2: Flatten all leaves
        # ============================================================
        all_leaves = flattener.flatten(tree)
        print(f"    [Step 2] Total leaves: {len(all_leaves)}")

        # ============================================================
        # Step 2.5: Time-reference pre-filter (optional)
        # ============================================================
        if time_ref and time_router:
            time_range = time_router.parse_time_reference(time_ref)
            if time_range:
                ref_start, ref_end = time_range
                time_filtered = [
                    e for e in all_leaves
                    if float(e["leaf"]["end_time"]) > ref_start
                    and float(e["leaf"]["start_time"]) < ref_end
                ]
                if time_filtered:
                    print(f"    [Time filter] {time_ref} → {len(time_filtered)}/{len(all_leaves)} leaves")
                    all_leaves = time_filtered

        # ============================================================
        # Step 3: Rule-based keyword filter
        # ============================================================
        marked_leaves, unmarked_leaves = rule_filter.filter(all_leaves, cues)
        print(f"    [Step 3] Marked: {len(marked_leaves)}, Unmarked: {len(unmarked_leaves)}")

        hop_history.append({
            "hop": 2, "type": "rule_filter",
            "total_leaves": len(all_leaves),
            "marked_count": len(marked_leaves),
            "unmarked_count": len(unmarked_leaves),
        })

        # ============================================================
        # Step 4: Hop Loop
        # ============================================================
        examined_ids = set()
        all_examined_entries = []
        expansion_used = False
        video_path = self.adapter.get_video_path(video_id)
        est_batches = math.ceil(len(marked_leaves) / leaf_budget) if marked_leaves else 1

        for hop_num in range(1, max_hops + 1):
            print(f"    [Hop {hop_num}]")

            # Determine candidate pool
            candidates = [e for e in marked_leaves if e["leaf_id"] not in examined_ids]
            source = "marked"

            if not candidates:
                if not expansion_used and unmarked_leaves:
                    candidates = [e for e in unmarked_leaves if e["leaf_id"] not in examined_ids]
                    source = "unmarked_expansion"
                    expansion_used = True
                    print(f"      Expanding to {len(candidates)} unmarked leaves")
                if not candidates:
                    print(f"      No more candidates. Stopping.")
                    break

            # LLM selects best N
            if len(candidates) <= leaf_budget:
                selected_ids = list(range(len(candidates)))
            else:
                selected_ids = leaf_selector.select(
                    candidates, cues, question, options, hop_history, budget=leaf_budget
                )

            current_batch = [candidates[i] for i in selected_ids]
            batch_ids = {e["leaf_id"] for e in current_batch}
            examined_ids.update(batch_ids)
            all_examined_entries.extend(current_batch)

            # ---- Phase A: Solvability check ----
            batch_context = formatter.format_leaf_batch(current_batch)
            solv = solvability.check(
                batch_context, question, options, hop_history,
                batch_info=(hop_num, est_batches)
            )
            print(f"      [Phase A] Solvable={solv['solvable']} | Answer={solv['answer']} | Depth={solv['needs_depth']}")

            hop_history.append({
                "hop": len(hop_history) + 1,
                "type": "leaf_solvability",
                "source": source,
                "batch_size": len(current_batch),
                "solvable": solv["solvable"],
                "answer": solv["answer"],
                "reasoning": solv["reasoning"],
                "needs_depth": solv["needs_depth"],
            })

            if solv["solvable"] and solv["answer"]:
                best_answer = solv["answer"]
                best_confidence = "high"
                print(f"      >> Solved from memory! Answer: {best_answer}")
                break

            # ---- Phase B: Frame Loading + VLM ----
            if frame_loader and vision_vlm and max_frames > 0 and os.path.exists(video_path):
                use_depth = solv.get("needs_depth", False)
                frame_entries = current_batch

                if use_depth and len(current_batch) > depth_budget:
                    depth_ids = leaf_selector.select(
                        current_batch, cues, question, options, hop_history, budget=depth_budget
                    )
                    frame_entries = [current_batch[i] for i in depth_ids]
                    mode = "depth"
                else:
                    mode = "width"

                intervals = []
                for entry in frame_entries:
                    leaf = entry["leaf"]
                    if "start_time" in leaf:
                        intervals.append((float(leaf["start_time"]), float(leaf["end_time"])))
                intervals.sort()

                if intervals:
                    frames_np, frame_secs = frame_loader.load(video_path, intervals, max_frames=max_frames)

                    if frames_np is not None:
                        print(f"      [Phase B] {mode} mode | {len(frame_secs)} frames")
                        frame_context = formatter.format_leaf_batch(frame_entries)
                        vlm_result = vision_vlm.infer(
                            frames_np, frame_context, question, options, hop_history
                        )
                        used_visual = True

                        ans = vlm_result.get("answer", "")
                        if ans:
                            m = re.search(r"[ABCD]", ans.upper())
                            ans = m.group(0) if m else None

                        hop_history.append({
                            "hop": len(hop_history) + 1,
                            "type": "frame_inference",
                            "mode": mode,
                            "n_frames": len(frame_secs),
                            "answer": ans,
                            "confidence": vlm_result["confidence"],
                            "observation": vlm_result["observation"],
                        })

                        if vlm_result["confidence"] == "high" and ans:
                            best_answer = ans
                            best_confidence = "high"
                            print(f"      >> High confidence VLM! Answer: {best_answer}")
                            del frames_np
                            gc.collect()
                            break
                        elif ans:
                            best_answer = ans
                            best_confidence = vlm_result["confidence"]

                        del frames_np
                        gc.collect()

        # ============================================================
        # Step 5: Forced answer fallback
        # ============================================================
        pred = best_answer
        if not pred:
            # Try from hop history
            for h in reversed(hop_history):
                if h.get("answer"):
                    m = re.search(r"[ABCD]", str(h["answer"]).upper())
                    if m:
                        pred = m.group(0)
                        break

        if not pred:
            print(f"      [Fallback] Forcing answer...")
            all_context = formatter.format_leaf_batch(all_examined_entries)
            pred = fallback.force_answer(all_context, question, options)

        correct = self.adapter.check_correct(pred, answer)

        # Coverage analysis
        time_coverage = {}
        if time_ref and coverage_analyzer:
            intervals = [(float(e["leaf"]["start_time"]), float(e["leaf"]["end_time"]))
                         for e in all_examined_entries if "start_time" in e["leaf"]]
            time_coverage = coverage_analyzer.compute(intervals, time_ref)

        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": "agentic",
            "confidence": best_confidence,
            "total_hops": len(hop_history),
            "used_visual": used_visual,
            "hop_history": hop_history,
            "time_reference": time_ref,
            "time_coverage": time_coverage,
            "total_leaves": len(all_leaves),
            "marked_leaves": len(marked_leaves),
            "examined_leaves": len(examined_ids),
            "question_type": question_data.get("question_type", []),
        }
