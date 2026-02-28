from __future__ import annotations

"""
Tree Search Pipeline — Filtered tree 기반 budget-aware 탐색 + judge + history compaction

Flow:
1. Decompose question → cues, time info
2. Build filtered tree (top-down activation via key_elements)
3. Select initial targets:
   - Time-direct: 시간 정보 있으면 해당 구간으로 바로 jump
   - Priority: 없으면 score 높은 path부터
4. Hop loop:
   a. Assemble context within max_text_budget (caption + parent summaries)
   b. Judge: "답할 수 있다/없다 + 이유"
   c. If answerable + high → done
   d. If answerable + medium → visual confirmation 시도
   e. If not answerable → compact history + navigate to next target
5. Return with traversal log (tree 경로 추적 가능)
"""

import os
import re
from pipelines.base import BasePipeline


class TreeSearchPipeline(BasePipeline):
    """Tree-traversal pipeline with judge-driven multi-hop search.

    Required components:
        query_analyzer, tree_filter, context_assembler, judge,
        history_compactor, formatter, fallback

    Optional components:
        visual_observer (for frame-based confirmation)
    """

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        max_hops = self.config.get("max_hops", 5)
        max_frames = self.config.get("max_frames", 30)
        max_text_budget = self.config.get("max_text_budget", 15000)

        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")
        time_ref = question_data.get("time_reference", "")
        tree = memory.get("streaming_memory_tree", {})

        # Components
        analyzer = self.components["query_analyzer"]
        tree_filter = self.components["tree_filter"]
        assembler = self.components["context_assembler"]
        judge = self.components["judge"]
        judge_visual = self.components.get("judge_visual")  # VisualJudge (optional)
        compactor = self.components.get("history_compactor")
        accumulator = self.components.get("history_accumulator")
        observer = self.components.get("visual_observer")
        fallback = self.components.get("fallback")
        semantic_matcher = self.components.get("semantic_matcher")

        use_visual_judge = judge_visual is not None
        use_accumulator = accumulator is not None
        use_two_stage = self.config.get("two_stage_visual", False)
        scout_frames_per_region = self.config.get("scout_frames_per_region", 3)
        planner = self.components.get("tree_planner")

        # D5: Visual Context Enrichment (caption → context → C3 rejudge)
        vision_vlm = self.components.get("vision_vlm")
        frame_loader = self.components.get("frame_loader")
        use_visual_enrich = (
            not use_visual_judge
            and self.config.get("visual_enrich", False)
            and vision_vlm is not None
            and frame_loader is not None
        )

        # ============================================================
        # STAGE 0: Planning (optional — prune tree via LLM overview)
        # ============================================================
        plan_info = None
        if planner and tree:
            tree, plan_info = planner.plan(tree, question, options)
            if plan_info and not plan_info.get("skipped_planning"):
                print(f"    [Planner] Selected {plan_info['selected_count']}/"
                      f"{plan_info['total_l1_regions']} L1 regions: "
                      f"{plan_info['selected_indices'][:8]}")

        # ============================================================
        # STAGE 1: Decompose
        # ============================================================
        analysis = analyzer.analyze(question, options, time_ref)
        cues = analysis["cues"]
        target_fields = analysis.get("target_fields", ["summary"])
        has_time = analysis.get("has_explicit_time", False)
        all_time_ranges = analysis.get("all_time_ranges", [])

        print(f"    [TreeSearch] cues={cues[:4]} | has_time={has_time}")

        # ============================================================
        # STAGE 1.5: Semantic matching (optional)
        # ============================================================
        semantic_scores = None
        if semantic_matcher:
            q_elements = semantic_matcher.extract_question_elements(
                question, options, cues,
            )
            semantic_scores = semantic_matcher.select_top_nodes(
                q_elements, tree, level="Level_1",
            )
            print(f"    [Semantic] {len(q_elements)} q_elements → "
                  f"top {len(semantic_scores['selected_indices'])}/"
                  f"{semantic_scores['total_nodes']} L1 nodes selected")
            if semantic_scores["scores"]:
                top3 = semantic_scores["scores"][:3]
                for s in top3:
                    print(f"      L1[{s['node_idx']}] score={s['score']:.2f} "
                          f"({s['n_elements']} elems)")

        # ============================================================
        # STAGE 2: Build filtered tree
        # ============================================================
        filtered = tree_filter.build(
            tree, cues, target_fields,
            semantic_scores=semantic_scores,
        )

        n_active = len(filtered["priority_leaves"])
        n_total = len(filtered["all_leaves"])
        print(f"    [TreeSearch] Filtered tree: {n_active} active / {n_total} total")

        # ============================================================
        # STAGE 2.5: Phase 0 — Coarse overview answer attempt
        # ============================================================
        phase0_verdict = None
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
                # Phase 0 high confidence → early return
                pred = phase0_verdict["answer"]
                correct = self.adapter.check_correct(pred, answer)
                print(f"    [Phase0] Early return: pred={pred} correct={correct}")
                return {
                    "pred": pred,
                    "answer": answer,
                    "correct": correct,
                    "method": "phase0_coarse",
                    "confidence": "high",
                    "used_visual": False,
                    "traversal_log": [],
                    "traversal_summary": "phase0_coarse",
                    "total_hops": 0,
                    "total_leaves": n_total,
                    "active_leaves": n_active,
                    "question_type": analysis.get("question_type", "global"),
                    "dataset_question_type": question_data.get("question_type", []),
                    "cues": cues,
                    "hop_contexts": [],
                    "plan_info": plan_info,
                    "memory_overview": self._build_memory_overview(tree),
                    "phase0_verdict": phase0_verdict,
                    "semantic_match": {
                        "selected_l1": semantic_scores["selected_indices"],
                        "top_scores": [
                            {"idx": s["node_idx"], "score": s["score"],
                             "top_matches": s["top_matches"][:3]}
                            for s in semantic_scores["scores"][:10]
                        ],
                    } if semantic_scores else None,
                }

        # ============================================================
        # R10a: Flat Baseline — skip tree navigation, dump all leaf summaries
        # ============================================================
        if self.config.get("flat_baseline", False) and tree:
            flat_ctx = self._flat_baseline_context(tree)
            verdict = self._forced_answer(flat_ctx, question, options, judge)
            pred = verdict.get("answer", "A") if verdict.get("answer") else "A"
            correct = self.adapter.check_correct(pred, answer)
            print(f"    [Flat] pred={pred} correct={correct} "
                  f"conf={verdict.get('confidence')}")
            return {
                "pred": pred,
                "answer": answer,
                "correct": correct,
                "method": "flat_baseline",
                "confidence": verdict.get("confidence", "low"),
                "used_visual": False,
                "traversal_log": [],
                "traversal_summary": "flat_baseline",
                "total_hops": 0,
                "total_leaves": n_total,
                "active_leaves": n_active,
                "question_type": analysis.get("question_type", "global"),
                "dataset_question_type": question_data.get("question_type", []),
                "cues": cues,
                "hop_contexts": [],
                "plan_info": None,
                "memory_overview": None,
                "phase0_verdict": verdict,
                "semantic_match": None,
            }

        # ============================================================
        # STAGE 2.7: A2 Recovery Cue — LLM-guided temporal selection
        # ============================================================
        recovery_info = None
        recovery_targets = []
        if (self.config.get("recovery_cue", False)
                and phase0_verdict is not None
                and phase0_verdict.get("confidence") != "high"
                and tree):
            recovery_info = self._recovery_cue(
                coarse_ctx, question, options, phase0_verdict, judge,
            )
            if recovery_info and recovery_info.get("time_periods"):
                # Re-select targets based on recovery cue time periods
                raw_targets = []
                for tp in recovery_info["time_periods"]:
                    found = tree_filter.find_by_time_range(
                        filtered, tp[0], tp[1], expand_window=15.0,
                    )
                    for f in found[:5]:
                        raw_targets.append(f["entry"])
                # Deduplicate
                seen_rt = set()
                for t in raw_targets:
                    key = self._leaf_id(t)
                    if key not in seen_rt:
                        seen_rt.add(key)
                        recovery_targets.append(t)
                recovery_targets = recovery_targets[:10]
                if recovery_targets:
                    print(f"    [A2] Recovery cue selected {len(recovery_targets)} "
                          f"targets from {len(recovery_info['time_periods'])} "
                          f"time periods")

        # ============================================================
        # STAGE 3: Select initial targets
        # ============================================================
        if recovery_targets:
            targets = recovery_targets
            target_source = "recovery_cue"
        else:
            targets, target_source = self._select_initial_targets(
                filtered, tree_filter, has_time, all_time_ranges,
            )

        print(f"    [TreeSearch] Initial: {len(targets)} targets ({target_source})")

        if not targets:
            return self._empty_result(question_data, "no_targets")

        # ============================================================
        # STAGE 4: Hop loop
        # ============================================================
        traversal_log = []
        hop_contexts = []  # 각 hop에서 모델이 본 context 저장
        history_compact = None  # used by compactor mode
        # accumulator is stateful — already initialized in components
        seen_ids = set()
        # Carry Phase 0 answer as baseline if available
        if phase0_verdict and phase0_verdict.get("answer"):
            best_pred = phase0_verdict["answer"]
            best_confidence = phase0_verdict.get("confidence", "low")
        else:
            best_pred = None
            best_confidence = "low"
        used_visual = False
        last_full_context = ""

        for hop in range(max_hops):
            # Track seen leaves
            for t in targets:
                seen_ids.add(self._leaf_id(t))

            # Build ancestors map for current targets
            ancestors_map = {}
            for t in targets:
                key = self._leaf_id(t)
                ancestors_map[key] = tree_filter.get_ancestors(tree, t)

            # Assemble context within budget
            ctx_result = assembler.assemble_with_neighbors(
                targets, ancestors_map, tree, max_budget=max_text_budget,
            )

            # Merge with history for LLM consumption
            full_context = assembler.format_for_hop(
                ctx_result["context"], history_compact, max_text_budget,
            )
            last_full_context = full_context

            # Traversal log entry
            log_entry = {
                "hop": hop + 1,
                "path": ctx_result["hierarchy_path"],
                "target_count": len(targets),
                "target_source": target_source if hop == 0 else "navigation",
                "budget_used": ctx_result["budget_used"],
            }
            traversal_log.append(log_entry)

            # Save context for this hop (모델이 실제로 본 정보)
            hop_target_details = []
            for t in targets:
                node = t.get("node", t)
                detail = {
                    "start_time": node.get("start_time"),
                    "end_time": node.get("end_time"),
                    "caption": node.get("caption", "")[:300],
                    "summary": node.get("summary", "")[:200],
                }
                # key_elements
                ke = node.get("key_elements", {})
                ke_brief = {}
                for field in ["actions", "objects", "persons"]:
                    vals = ke.get(field, [])
                    if vals:
                        ke_brief[field] = [str(v) for v in vals[:5]]
                if ke_brief:
                    detail["key_elements"] = ke_brief
                # Parent context
                if t.get("parent_l1_summary"):
                    detail["parent_summary"] = t["parent_l1_summary"][:150]
                hop_target_details.append(detail)

            # Ancestors (이 hop에서 본 상위 노드들)
            hop_ancestors = []
            seen_anc = set()
            for t in targets:
                key = self._leaf_id(t)
                for anc in ancestors_map.get(key, []):
                    anc_key = (anc.get("level"), anc.get("summary", "")[:50])
                    if anc_key not in seen_anc:
                        seen_anc.add(anc_key)
                        hop_ancestors.append({
                            "level": anc.get("level"),
                            "summary": anc.get("summary", "")[:200],
                        })

            hop_ctx_entry = {
                "hop": hop + 1,
                "path": ctx_result["hierarchy_path"],
                "budget_used": ctx_result["budget_used"],
                "context_text": full_context,
                "target_leaves": hop_target_details,
                "ancestors": hop_ancestors,
            }
            hop_contexts.append(hop_ctx_entry)

            print(f"    [Hop {hop + 1}/{max_hops}] "
                  f"path={ctx_result['hierarchy_path']} | "
                  f"targets={len(targets)} | "
                  f"budget={ctx_result['budget_used']}")

            # ========================================
            # Judge: can we answer?
            # ========================================
            if use_accumulator:
                history_text = accumulator.format_for_judge(
                    max_tokens=self.config.get("history_budget", 20000),
                )
                history_text = history_text if history_text else None
            else:
                history_text = (
                    history_compact["compact_text"]
                    if history_compact else None
                )

            # Resolve video path + time ranges for visual judge / D5
            video_path = None
            target_time_ranges = []
            if use_visual_judge or use_visual_enrich:
                video_path = (
                    self.adapter.get_video_path(video_id) if video_id else None
                )
                for t in targets:
                    node = t.get("node", t)
                    st = node.get("start_time")
                    et = node.get("end_time")
                    if st is not None and et is not None:
                        target_time_ranges.append((float(st), float(et)))

            if use_visual_judge:
                if use_two_stage:
                    # Two-stage: scout → select → focus → rejudge
                    verdict = judge_visual.judge_full_two_stage(
                        full_context, question, options,
                        video_path=video_path,
                        time_ranges=target_time_ranges,
                        history_compact=history_text,
                        max_frames=max_frames,
                        scout_frames_per_region=scout_frames_per_region,
                    )
                else:
                    # Original: judge → caption → rejudge
                    verdict = judge_visual.judge_full(
                        full_context, question, options,
                        video_path=video_path,
                        time_ranges=target_time_ranges,
                        history_compact=history_text,
                        max_frames=max_frames,
                    )
                if verdict.get("used_visual"):
                    used_visual = True
                    # Inject caption text into context for future hops
                    caption_text = verdict.get("caption_text", "")
                    if caption_text:
                        full_context += "\n\n" + caption_text
                        last_full_context = full_context
            else:
                # Original judge (text-only)
                verdict = judge.judge(
                    full_context, question, options, history_text,
                )

                # --- D5: Visual Context Enrichment ---
                if (use_visual_enrich
                        and verdict.get("confidence") in ("low", "medium")
                        and video_path
                        and target_time_ranges):
                    try:
                        frames_np, frame_secs = frame_loader.load(
                            video_path, target_time_ranges,
                            max_frames=max_frames,
                        )
                        if frames_np is not None and len(frames_np) > 0:
                            vlm_result = vision_vlm.infer(
                                frames_np, full_context, question, options,
                            )
                            obs = vlm_result.get("observation", "")
                            if obs:
                                caption_text = (
                                    "\n\n=== Visual Observations ===\n" + obs
                                )
                                enriched_context = full_context + caption_text
                                verdict = judge.judge(
                                    enriched_context, question, options,
                                    history_text,
                                )
                                verdict["used_visual"] = True
                                verdict["caption_text"] = caption_text
                                full_context = enriched_context
                                last_full_context = full_context
                                used_visual = True
                                print(f"    [D5] Visual enrich: "
                                      f"conf={verdict.get('confidence')} | "
                                      f"ans={verdict.get('answer')}")
                    except Exception as e:
                        print(f"    [D5] Visual enrich failed: {e}")

            log_entry["judge"] = {
                "answerable": verdict.get("answerable"),
                "confidence": verdict.get("confidence"),
                "answer": verdict.get("answer"),
                "reasoning": verdict.get("reasoning", "")[:200],
                "missing_info": verdict.get("missing_info", ""),
                "search_direction": verdict.get("search_direction", ""),
                "needs_visual": verdict.get("needs_visual"),
                "visual_helped": verdict.get("visual_helped"),
                "used_visual": verdict.get("used_visual", False),
            }

            print(f"    [Judge] answerable={verdict.get('answerable')} | "
                  f"conf={verdict.get('confidence')} | "
                  f"ans={verdict.get('answer')}"
                  + (f" | visual={verdict.get('used_visual')}"
                     if use_visual_judge else ""))

            # ========================================
            # Decision based on verdict
            # ========================================
            if verdict.get("answerable") and verdict.get("answer"):
                conf = verdict["confidence"]

                if conf == "high":
                    best_pred = verdict["answer"]
                    best_confidence = "high"
                    break

                if conf == "medium":
                    best_pred = verdict["answer"]
                    best_confidence = "medium"

                    if not use_visual_judge:
                        # Legacy path: observer-based visual confirmation
                        vis_result = self._try_visual(
                            observer, targets, full_context, question,
                            options, verdict.get("reasoning", ""),
                            video_id, max_frames,
                        )
                        if vis_result:
                            used_visual = True
                            log_entry["visual"] = {
                                "confidence": vis_result.get("confidence"),
                                "answer": vis_result.get("answer"),
                            }

                            if vis_result.get("confidence") == "high":
                                best_pred = vis_result["answer"]
                                best_confidence = "high"
                                break

                            if vis_result.get("observations_text"):
                                full_context += (
                                    "\n\n=== Visual Observations ===\n"
                                    + vis_result["observations_text"]
                                )
                                last_full_context = full_context

                    # Even medium, don't break yet — try to confirm

                # Low confidence: save tentative answer, keep searching
                if conf == "low" and verdict.get("answer"):
                    if not best_pred:
                        best_pred = verdict["answer"]
                        best_confidence = "low"

            # ========================================
            # History: accumulate or compact
            # ========================================
            if use_accumulator:
                accumulator.accumulate(
                    hop_number=hop + 1,
                    context=ctx_result["context"],
                    verdict=verdict,
                    target_time_ranges=target_time_ranges if use_visual_judge else None,
                    visual_captions=verdict.get("caption_text"),
                )
            elif compactor:
                history_compact = compactor.compact(
                    question, ctx_result["context"], verdict,
                    hop + 1, traversal_log, history_compact,
                )

            # ========================================
            # Navigate to next target
            # ========================================
            next_targets = self._navigate_next(
                tree, filtered, seen_ids, verdict, targets, tree_filter,
                question=question,
            )

            if not next_targets:
                print(f"    [Hop {hop + 1}] No more targets. Done.")
                break

            targets = next_targets

        # ============================================================
        # Final answer
        # ============================================================
        method_name = f"tree_search_hop{len(traversal_log)}"

        if not best_pred or best_confidence not in ("high",):
            # Try visual on best targets if not done
            if not used_visual and observer:
                vis_result = self._try_visual(
                    observer, targets, last_full_context, question,
                    options, "Final attempt",
                    video_id, max_frames,
                )
                if vis_result and vis_result.get("answer"):
                    used_visual = True
                    best_pred = vis_result["answer"]
                    best_confidence = vis_result.get("confidence", "medium")

            # Forced fallback
            if not best_pred:
                if fallback and fallback.llm_fn:
                    best_pred = fallback.force_answer(
                        last_full_context, question, options,
                    )
                    method_name = "tree_search_fallback"

                # Final safety net
                if not best_pred:
                    best_pred = "A"
                    method_name = "tree_search_random"

        correct = self.adapter.check_correct(best_pred, answer)

        # Build traversal summary
        traversal_summary = " → ".join(
            f"Hop{e['hop']}({e['path']})"
            for e in traversal_log
        )

        print(f"    [TreeSearch] pred={best_pred} | correct={correct} | "
              f"hops={len(traversal_log)} | visual={used_visual}")
        print(f"    [Traversal] {traversal_summary}")

        # Full memory overview (간소화)
        memory_overview = self._build_memory_overview(tree)

        return {
            "pred": best_pred,
            "answer": answer,
            "correct": correct,
            "method": method_name,
            "confidence": best_confidence,
            "used_visual": used_visual,
            "traversal_log": traversal_log,
            "traversal_summary": traversal_summary,
            "total_hops": len(traversal_log),
            "total_leaves": n_total,
            "active_leaves": n_active,
            "question_type": analysis.get("question_type", "global"),
            "dataset_question_type": question_data.get("question_type", []),
            "cues": cues,
            "hop_contexts": hop_contexts,
            "plan_info": plan_info,
            "memory_overview": memory_overview,
            "semantic_match": {
                "selected_l1": semantic_scores["selected_indices"],
                "top_scores": [
                    {"idx": s["node_idx"], "score": s["score"],
                     "top_matches": s["top_matches"][:3]}
                    for s in semantic_scores["scores"][:10]
                ],
            } if semantic_scores else None,
            "recovery_info": recovery_info,
        }

    # ==================== Target Selection ====================

    def _select_initial_targets(
        self, filtered, tree_filter, has_time, all_time_ranges,
    ):
        """Select initial targets based on time or priority."""
        if has_time and all_time_ranges:
            # Time-direct jump + neighbors
            targets = []
            for tr in all_time_ranges:
                if isinstance(tr, (list, tuple)) and len(tr) >= 2:
                    found = tree_filter.find_by_time_range(
                        filtered, float(tr[0]), float(tr[1]),
                        expand_window=15.0,
                    )
                    for f in found[:5]:
                        targets.append(f["entry"])
                else:
                    mid = float(tr)
                    found = tree_filter.find_by_time(
                        filtered, mid, window=30.0,
                    )
                    for f in found[:5]:
                        targets.append(f["entry"])

            # Deduplicate
            seen = set()
            unique = []
            for t in targets:
                key = self._leaf_id(t)
                if key not in seen:
                    seen.add(key)
                    unique.append(t)
            targets = unique

            if targets:
                return targets[:10], "time_direct"

        # Priority-based: top scoring paths
        priority = filtered.get("priority_leaves", [])
        if priority:
            return priority[:10], "priority"

        # Fallback: any leaves
        all_leaves = filtered.get("all_leaves", [])
        if all_leaves:
            return all_leaves[:10], "all_fallback"

        return [], "none"

    # ==================== Visual ====================

    def _try_visual(
        self, observer, targets, context, question, options,
        reasoning, video_id, max_frames,
    ):
        """Try visual observation for confirmation."""
        if not observer:
            return None

        video_path = (
            self.adapter.get_video_path(video_id) if video_id else None
        )
        if not video_path or not os.path.exists(video_path):
            return None

        time_ranges = []
        for t in targets:
            node = t.get("node", t)
            st = node.get("start_time")
            et = node.get("end_time")
            if st is not None and et is not None:
                time_ranges.append((float(st), float(et)))

        if not time_ranges:
            return None

        result = observer.observe_and_answer(
            context=context,
            question=question,
            options=options,
            previous_reasoning=reasoning,
            time_ranges=time_ranges,
            video_path=video_path,
            max_frames=max_frames,
        )

        # Build observations text
        obs_text = ""
        if result.get("observations"):
            obs_text = "\n".join(f"- {o}" for o in result["observations"])
            if result.get("key_finding"):
                obs_text += f"\nKey finding: {result['key_finding']}"
        result["observations_text"] = obs_text

        return result

    # ==================== Navigation ====================

    def _navigate_next(
        self, tree, filtered, seen_ids, verdict, current_targets,
        tree_filter, question="",
    ):
        """Determine next targets based on judge's search_direction."""
        direction = verdict.get("search_direction", "")

        # Strategy 1: If judge says "same_region_detail" → siblings
        if direction == "same_region_detail":
            siblings = []
            for t in current_targets:
                sibs = tree_filter.get_siblings(tree, t, seen_ids)
                for s in sibs:
                    key = self._leaf_id(s)
                    if key not in seen_ids:
                        siblings.append(s)
                        seen_ids.add(key)
            if siblings:
                return siblings[:8]

        # Strategy 2: If judge says "earlier_time" or "later_time"
        if direction in ("earlier_time", "later_time"):
            current_times = [self._leaf_id(t) for t in current_targets]
            if current_times:
                if direction == "earlier_time":
                    ref_time = min(t[0] for t in current_times)
                    found = tree_filter.find_by_time(
                        filtered, ref_time - 30, window=60.0,
                    )
                else:
                    ref_time = max(t[1] for t in current_times)
                    found = tree_filter.find_by_time(
                        filtered, ref_time + 30, window=60.0,
                    )
                next_targets = []
                for f in found:
                    key = self._leaf_id(f["entry"])
                    if key not in seen_ids:
                        next_targets.append(f["entry"])
                        seen_ids.add(key)
                if next_targets:
                    return next_targets[:8]

        # Strategy 3: Next priority leaves (unseen)
        unseen = tree_filter.get_unseen_leaves(
            filtered, seen_ids, budget=8,
        )
        if unseen:
            for u in unseen:
                seen_ids.add(self._leaf_id(u))
            return unseen

        # Strategy 4: Siblings of current targets
        siblings = []
        for t in current_targets:
            sibs = tree_filter.get_siblings(tree, t, seen_ids)
            for s in sibs:
                key = self._leaf_id(s)
                if key not in seen_ids:
                    siblings.append(s)
                    seen_ids.add(key)
        if siblings:
            return siblings[:8]

        # Strategy 5: LLM-guided global navigation (zoom out)
        # Go up one level at a time: Level_1 → Level_2 → Level_3 ...
        global_targets = self._navigate_global(
            tree, tree_filter, seen_ids, verdict, question,
        )
        if global_targets:
            for t in global_targets:
                seen_ids.add(self._leaf_id(t))
            return global_targets

        return []

    # ==================== Global Navigation ====================

    GLOBAL_NAV_PROMPT = """You are navigating a video to answer a question. The activated regions have been exhausted — now you need to pick the most promising unexplored region.

### Question
{question}

### What we know so far
{reasoning}

### What's missing
{missing}

### Unexplored Regions
{regions_text}

Pick the 1-2 most promising region indices that are most likely to contain the missing information. Output ONLY a JSON list of indices, e.g. [0] or [2, 5]."""

    def _navigate_global(
        self, tree, tree_filter, seen_ids, verdict, question="",
    ):
        """Zoom out to higher levels, show summaries to LLM, pick next region."""
        llm = self.components.get("llm")
        if not llm:
            return []

        reasoning = (verdict.get("reasoning") or "")[:300]
        missing = (verdict.get("missing_info") or "continue searching")[:200]

        # Try each level: Level_1 → Level_2 → Level_3 → ...
        levels = sorted(
            [k for k in tree.keys()],
            key=lambda x: int(x.split("_")[1]),
        )

        for level_name in levels:
            regions = tree_filter.get_unexplored_regions(
                tree, seen_ids, level=level_name,
            )
            if not regions:
                continue

            # Format regions for LLM
            lines = []
            for i, r in enumerate(regions):
                ke_str = ""
                for field, vals in r.get("key_elements_brief", {}).items():
                    ke_str += f" | {field}: {', '.join(vals)}"
                lines.append(
                    f"[{i}] [{r['start_time']:.0f}s-{r['end_time']:.0f}s] "
                    f"{r['summary'][:150]}"
                    f" ({r['unseen_count']} unseen/{r['total_count']} total)"
                    f"{ke_str}"
                )

            regions_text = "\n".join(lines)
            prompt = self.GLOBAL_NAV_PROMPT.format(
                question=question,
                reasoning=reasoning,
                missing=missing,
                regions_text=regions_text,
            )

            try:
                result = llm.reason(prompt, max_tokens=50)
            except Exception:
                result = {}

            # Parse picked indices
            picked_indices = []
            if isinstance(result, dict):
                raw = result.get("answer", result.get("reasoning", ""))
            else:
                raw = str(result)

            import re
            nums = re.findall(r'\d+', str(raw))
            for n in nums:
                idx = int(n)
                if 0 <= idx < len(regions):
                    picked_indices.append(idx)

            if not picked_indices:
                picked_indices = [0]  # fallback: pick the region with most unseen

            # Collect leaves from picked regions
            targets = []
            for idx in picked_indices[:2]:
                leaves = tree_filter.get_leaves_under_region(
                    tree, regions[idx], seen_ids, budget=8,
                )
                targets.extend(leaves)

            if targets:
                print(f"    [GlobalNav] Zoomed out to {level_name}: "
                      f"picked {len(targets)} leaves from "
                      f"{len(picked_indices)} region(s)")
                return targets[:10]

        return []

    # ==================== Phase 0: Coarse Overview ====================

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

    # Phase 0 uses force prompt — always produce an answer from overview
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

    def _phase0_coarse_answer(self, tree, question, options, judge):
        """Phase 0: Level_N ~ Level_1 전체 summary로 forced answer 시도.

        Leaf caption은 제외, summary만 사용.
        Force prompt로 항상 답을 내놓게 함.

        Returns:
            (verdict, coarse_context)
        """
        # Collect all levels with summaries (highest first)
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

                ke = node.get("key_elements", {})
                ke_brief = ""
                for field in ["actions", "objects", "persons"]:
                    vals = ke.get(field, [])
                    if vals:
                        ke_brief += " | %s: %s" % (
                            field, ", ".join(str(v) for v in vals[:5]),
                        )

                coarse_parts.append({
                    "level": level_name,
                    "start": start,
                    "end": end,
                    "text": "[%s] [%.0fs-%.0fs] %s%s" % (
                        level_name, start, end, summary, ke_brief,
                    ),
                })

        # Sort by time
        coarse_parts.sort(key=lambda x: x["start"])
        coarse_context = (
            "=== Video Overview (all segments) ===\n"
            + "\n".join(p["text"] for p in coarse_parts)
        )

        # Use force prompt directly via judge's LLM
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

        # Normalize answer
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

    # ==================== A2: Recovery Cue ====================

    RECOVERY_CUE_PROMPT = """The following provides rough descriptions of what's shown in the video during different time periods:

{context}

A question has been raised regarding this video:
{question}

{options_text}

A previous attempt to answer this question from the overview above was not confident enough.
Previous reasoning: {prev_reasoning}

Since the overview descriptions are rough and some detailed information is lost, your task is to identify the most relevant time periods that need closer examination to answer this question more accurately.

Please analyze the question carefully and determine:
1. Which specific time periods from the video are most likely to contain the key information needed?
2. What specific details should be looked for in those time periods?

Output ONLY valid JSON:
{{
    "time_periods": [[start1, end1], [start2, end2], [start3, end3]],
    "focus_points": "What to look for in these time periods",
    "reasoning": "Why these time periods are relevant"
}}

Rules:
- Select up to 3 most relevant time periods from the descriptions above
- Use the exact time ranges (in seconds) from the descriptions
- If the question requires tracking something across the whole video (e.g. counting occurrences), select time periods that are spread across the video
- If the question is about a specific event, select the time periods most likely to contain that event"""

    def _recovery_cue(self, coarse_context, question, options, phase0_verdict, judge):
        """A2: Phase 0 이후 recovery cue 생성.

        Phase 0의 coarse_context + 질문을 보고, LLM이 어떤 시간대를
        더 자세히 봐야 하는지 추론. 반환된 시간대로 tree에서 leaf를 재선정.

        Returns:
            dict with time_periods, focus_points, reasoning or None
        """
        opt_text = "\n".join(
            "%s. %s" % (chr(65 + i), o) for i, o in enumerate(options)
        )
        prev_reasoning = (phase0_verdict.get("reasoning") or "No reasoning")[:300]

        prompt = self.RECOVERY_CUE_PROMPT.format(
            context=coarse_context,
            question=question,
            options_text=opt_text,
            prev_reasoning=prev_reasoning,
        )

        try:
            result = judge.llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"      [A2] Recovery cue error: {e}")
            return None

        if not isinstance(result, dict):
            print(f"      [A2] Recovery cue: non-dict result")
            return None

        # Parse time_periods
        time_periods = result.get("time_periods", [])
        parsed_periods = []
        for tp in time_periods:
            if isinstance(tp, (list, tuple)) and len(tp) >= 2:
                try:
                    parsed_periods.append((float(tp[0]), float(tp[1])))
                except (ValueError, TypeError):
                    continue

        if not parsed_periods:
            print(f"      [A2] Recovery cue: no valid time periods")
            return None

        info = {
            "time_periods": parsed_periods,
            "focus_points": result.get("focus_points", ""),
            "reasoning": result.get("reasoning", ""),
        }
        print(f"      [A2] Recovery cue: {len(parsed_periods)} time periods, "
              f"focus={info['focus_points'][:80]}")
        return info

    # ==================== R10a: Flat Baseline ====================

    def _flat_baseline_context(self, tree):
        """Flat: Level_1 children(leaf)의 summary만 시간순 concat."""
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
        print(f"    [Flat] {len(leaves)} leaf summaries assembled")
        return "=== Video Content (all segments) ===\n" + "\n".join(lines)

    def _forced_answer(self, context, question, options, judge):
        """Force prompt로 답을 강제하는 공용 메서드 (Phase 0, Flat 공용)."""
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
            "reasoning": "Forced answer failed", "missing_info": None,
            "search_direction": None,
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
        verdict["answerable"] = bool(verdict.get("answerable", False))

        return verdict

    # ==================== Helpers ====================

    @staticmethod
    def _leaf_id(entry):
        """Extract (start_time, end_time) from a leaf entry."""
        node = entry.get("node", entry)
        return (
            float(node.get("start_time", entry.get("start_time", 0))),
            float(node.get("end_time", entry.get("end_time", 0))),
        )

    @staticmethod
    def _build_memory_overview(tree: dict) -> dict:
        """Full memory tree를 간소화한 overview dict.

        각 레벨의 노드 summary + key_elements + time_range만 포함.
        Leaf caption은 첫 100자로 잘라서 포함.
        """
        overview = {}
        for level_name, nodes in tree.items():
            level_nodes = []
            for idx, node in enumerate(nodes):
                entry = {
                    "idx": idx,
                    "summary": node.get("summary", "")[:300],
                }
                # Time range
                segs = node.get("time_segments", [])
                if segs:
                    flat = []
                    for s in segs:
                        if isinstance(s, (list, tuple)) and len(s) >= 2:
                            flat.extend([float(s[0]), float(s[1])])
                        else:
                            flat.append(float(s))
                    if flat:
                        entry["time_range"] = [min(flat), max(flat)]

                # Key elements (간략)
                ke = node.get("key_elements", {})
                ke_brief = {}
                for field in ["actions", "objects", "persons",
                              "locations", "text_ocr"]:
                    vals = ke.get(field, [])
                    if vals:
                        ke_brief[field] = [str(v) for v in vals[:5]]
                if ke_brief:
                    entry["key_elements"] = ke_brief

                # Children count + leaf info (Level_1만)
                children = node.get("children", [])
                if children and "start_time" in children[0]:
                    entry["n_leaves"] = len(children)
                    entry["leaves"] = []
                    for c in children:
                        leaf_info = {
                            "time": f"{float(c.get('start_time', 0)):.1f}-{float(c.get('end_time', 0)):.1f}",
                            "summary": c.get("summary", "")[:100],
                            "caption_preview": c.get("caption", "")[:100],
                        }
                        entry["leaves"].append(leaf_info)

                level_nodes.append(entry)
            overview[level_name] = level_nodes
        return overview

    def _empty_result(self, question_data, reason):
        """Return an empty result when no targets found."""
        answer = question_data.get("answer")
        return {
            "pred": "A",
            "answer": answer,
            "correct": self.adapter.check_correct("A", answer),
            "method": f"tree_search_{reason}",
            "confidence": "low",
            "used_visual": False,
            "traversal_log": [],
            "traversal_summary": "none",
            "total_hops": 0,
            "total_leaves": 0,
            "active_leaves": 0,
            "question_type": "unknown",
            "dataset_question_type": question_data.get("question_type", []),
        }
