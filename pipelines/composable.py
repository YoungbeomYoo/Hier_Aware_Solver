"""
Composable Pipeline — Config-driven 단계별 조합 파이프라인

기존 4개 파이프라인(memory_only, routed, agentic, cognitive)은 flow가 고정.
ComposablePipeline은 config에서 각 단계의 전략을 자유롭게 지정:

stages:
  decompose:   query_analyzer | query_decomposer
  filter:      hierarchical_scorer | metadata_filter | rule_filter
  context:     spreading_activation | flat | scored_hierarchy
  search:      escalation | single_pass | multi_hop
  check:       uncertainty | solvability | none
  visual:      on_uncertainty | always | never
  answer:      elimination | solvability | direct

Example config:
  pipeline: composable
  stages:
    decompose: query_analyzer
    filter: hierarchical_scorer
    context: scored_hierarchy
    search: multi_hop
    check: uncertainty
    visual: on_uncertainty
    answer: elimination
"""

import os
import gc
import re
from pipelines.base import BasePipeline
from components.answer_parser import extract_choice_letter


class ComposablePipeline(BasePipeline):
    """Config-driven composable solver pipeline.

    stages config에 따라 각 단계를 자유롭게 조합.
    모든 컴포넌트는 components dict에서 가져옴.
    """

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        stages = self.config.get("stages", {})
        max_hops = self.config.get("max_hops", 5)
        max_frames = self.config.get("max_frames", 30)
        leaf_budget = self.config.get("leaf_budget", 15)

        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")
        time_ref = question_data.get("time_reference", "")
        tree = memory.get("streaming_memory_tree", {})

        hop_history = []
        used_visual = False

        # ============================================================
        # STAGE 1: Decompose
        # ============================================================
        decompose_method = stages.get("decompose", "query_analyzer")
        analysis = self._decompose(decompose_method, question, options, time_ref)

        cues = analysis["cues"]
        target_fields = analysis.get("target_fields", ["summary"])
        q_type = analysis.get("question_type", "global")
        has_time = analysis.get("has_explicit_time", False)
        all_time_ranges = analysis.get("all_time_ranges", [])

        print(f"    [Decompose:{decompose_method}] type={q_type} | cues={cues[:4]}")
        hop_history.append({
            "stage": "decompose", "method": decompose_method,
            "question_type": q_type, "cues": cues,
            "target_fields": target_fields,
        })

        # ============================================================
        # STAGE 2: Filter / Score
        # ============================================================
        filter_method = stages.get("filter", "hierarchical_scorer")
        flattener = self.components["flattener"]
        all_leaves = flattener.flatten(tree)

        filter_result = self._filter(
            filter_method, tree, all_leaves, cues, target_fields,
            has_time, all_time_ranges, leaf_budget,
        )
        target_leaves = filter_result["target_leaves"]
        score_result = filter_result.get("score_result")

        print(f"    [Filter:{filter_method}] {len(target_leaves)} targets / {len(all_leaves)} total")
        hop_history.append({
            "stage": "filter", "method": filter_method,
            "target_count": len(target_leaves),
            "total_leaves": len(all_leaves),
        })

        # ============================================================
        # STAGE 3: Context Assembly
        # ============================================================
        context_method = stages.get("context", "scored_hierarchy")
        formatter = self.components["formatter"]

        full_context = self._build_context(
            context_method, tree, target_leaves, score_result, formatter,
        )

        print(f"    [Context:{context_method}] {len(full_context)} chars")

        # ============================================================
        # STAGE 4: Search (single_pass or multi_hop)
        # ============================================================
        search_method = stages.get("search", "single_pass")
        check_method = stages.get("check", "uncertainty")
        visual_method = stages.get("visual", "on_uncertainty")
        answer_method = stages.get("answer", "elimination")

        if search_method == "escalation":
            result = self._escalation_search(
                tree, all_leaves, target_leaves, score_result, cues,
                full_context, question, options, hop_history,
                answer_method, video_id, max_hops, max_frames,
                leaf_budget, formatter, has_time,
            )
            used_visual = result.get("used_visual", False)
            pred = result["pred"]
            confidence = result.get("confidence", "medium")
            method_name = f"composable_escalation_{result.get('escalation_level', 0)}"

        elif search_method == "multi_hop" and score_result:
            result = self._multi_hop_search(
                tree, all_leaves, target_leaves, score_result, cues,
                full_context, question, options, hop_history,
                check_method, visual_method, answer_method,
                video_id, max_hops, max_frames, leaf_budget, formatter,
            )
            used_visual = result.get("used_visual", False)
            pred = result["pred"]
            confidence = result.get("confidence", "medium")
            method_name = f"composable_multihop_{answer_method}"

        else:
            # Single pass: check → visual → answer
            result = self._single_pass(
                full_context, target_leaves, question, options, hop_history,
                check_method, visual_method, answer_method,
                video_id, max_frames, formatter,
            )
            used_visual = result.get("used_visual", False)
            pred = result["pred"]
            confidence = result.get("confidence", "medium")
            method_name = f"composable_single_{answer_method}"

        # ============================================================
        # Fallback
        # ============================================================
        if not pred:
            fallback = self.components.get("fallback")
            if fallback and fallback.llm_fn:
                pred = fallback.force_answer(full_context, question, options)
                method_name = "composable_fallback"
            else:
                pred = "A"
                method_name = "composable_random"

        correct = self.adapter.check_correct(pred, answer)
        print(f"    [Answer:{answer_method}] pred={pred} | correct={correct}")

        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": method_name,
            "confidence": confidence,
            "used_visual": used_visual,
            "hop_history": hop_history,
            "stages": stages,
            "total_leaves": len(all_leaves),
            "target_leaves": len(target_leaves),
            "question_type": q_type,
            "dataset_question_type": question_data.get("question_type", []),
        }

    # ==================== Decompose ====================

    def _decompose(self, method: str, question: str, options: list[str],
                   time_ref: str) -> dict:
        if method == "query_analyzer":
            analyzer = self.components["query_analyzer"]
            return analyzer.analyze(question, options, time_ref)

        elif method == "query_decomposer":
            decomposer = self.components["decomposer"]
            result = decomposer.decompose(question, options)
            # Normalize output
            return {
                "cues": result.get("cues", []),
                "target_fields": ["summary", "actions"],
                "question_type": "global",
                "has_explicit_time": False,
                "all_time_ranges": [],
                "target_action": result.get("target_action", ""),
            }

        raise ValueError(f"Unknown decompose method: {method}")

    # ==================== Filter ====================

    def _filter(self, method: str, tree: dict, all_leaves: list,
                cues: list, target_fields: list,
                has_time: bool, time_ranges: list,
                budget: int) -> dict:

        if method == "hierarchical_scorer":
            scorer = self.components.get("hierarchical_scorer")
            if not scorer:
                from components.hierarchical_scorer import HierarchicalScorer
                scorer = HierarchicalScorer()

            score_result = scorer.score_tree(tree, cues, target_fields)
            target_leaves = scorer.get_priority_leaves(score_result, budget=budget)

            # Time-based boost: if explicit time, also include time matches
            if has_time and time_ranges:
                meta_filter = self.components.get("metadata_filter")
                if meta_filter:
                    time_matched = meta_filter.filter_by_time(all_leaves, time_ranges)
                    existing_ids = {e["leaf_id"] for e in target_leaves}
                    for tm in time_matched:
                        if tm["leaf_id"] not in existing_ids:
                            target_leaves.append(tm)
                            existing_ids.add(tm["leaf_id"])

            return {"target_leaves": target_leaves, "score_result": score_result}

        elif method == "metadata_filter":
            meta_filter = self.components["metadata_filter"]
            secondary = self.components["query_analyzer"].analyze(
                "", [], ""
            ).get("secondary_fields", []) if "query_analyzer" in self.components else []

            if has_time and time_ranges:
                target_leaves = meta_filter.filter_by_time(all_leaves, time_ranges)
                if cues:
                    refined, _ = meta_filter.filter_by_fields(
                        target_leaves or all_leaves, cues, target_fields, secondary
                    )
                    target_leaves = refined or target_leaves
            else:
                target_leaves, _ = meta_filter.filter_by_fields(
                    all_leaves, cues, target_fields, secondary
                )

            return {"target_leaves": target_leaves[:budget * 2], "score_result": None}

        elif method == "rule_filter":
            rule_filter = self.components["rule_filter"]
            marked, _ = rule_filter.filter(all_leaves, cues)
            return {"target_leaves": marked[:budget * 2], "score_result": None}

        raise ValueError(f"Unknown filter method: {method}")

    # ==================== Context ====================

    def _build_context(self, method: str, tree: dict,
                       target_leaves: list, score_result: dict | None,
                       formatter) -> str:

        if method == "scored_hierarchy" and score_result:
            scorer = self.components.get("hierarchical_scorer")
            if not scorer:
                from components.hierarchical_scorer import HierarchicalScorer
                scorer = HierarchicalScorer()
            hierarchy_ctx = scorer.get_activated_context(tree, score_result)
            leaf_ctx = formatter.format_leaf_batch(target_leaves)
            return hierarchy_ctx + "\n\n=== Detailed Segments ===\n" + leaf_ctx

        elif method == "spreading_activation":
            activation = self.components.get("spreading_activation")
            if activation:
                act_result = activation.activate(tree, target_leaves)
                leaf_ctx = formatter.format_leaf_batch(target_leaves)
                return act_result["activated_context"] + "\n\n" + leaf_ctx
            # Fallback to flat
            return formatter.format_leaf_batch(target_leaves)

        elif method == "flat":
            return formatter.format_leaf_batch(target_leaves)

        # Default
        return formatter.format_leaf_batch(target_leaves)

    # ==================== Single Pass ====================

    def _single_pass(self, context: str, target_leaves: list,
                     question: str, options: list, hop_history: list,
                     check_method: str, visual_method: str,
                     answer_method: str, video_id: str,
                     max_frames: int, formatter) -> dict:
        result = {"pred": None, "confidence": "low", "used_visual": False}

        # Check
        check_result = self._run_check(check_method, context, question, options, hop_history)
        pre_elimination = check_result.get("elimination", {})

        # Visual
        if self._should_load_visual(visual_method, check_result):
            vis = self._load_visual(
                target_leaves, check_result, video_id, context,
                question, options, hop_history, max_frames, formatter,
            )
            if vis["used"]:
                result["used_visual"] = True
                if vis.get("observation"):
                    context += f"\n\n=== Visual Observation ===\n{vis['observation']}"

        # Early exit if confident from check
        if check_result.get("confidence") in ("certain", "likely") and check_result.get("answer"):
            result["pred"] = check_result["answer"]
            result["confidence"] = check_result["confidence"]
            return result

        # Answer
        result.update(self._run_answer(
            answer_method, context, question, options,
            hop_history, pre_elimination,
        ))
        return result

    # ==================== Multi-Hop ====================

    def _multi_hop_search(
        self, tree: dict, all_leaves: list, initial_targets: list,
        score_result: dict, cues: list, initial_context: str,
        question: str, options: list, hop_history: list,
        check_method: str, visual_method: str, answer_method: str,
        video_id: str, max_hops: int, max_frames: int,
        leaf_budget: int, formatter,
    ) -> dict:
        """Multi-hop: 가장 좋은 매칭 먼저 보고, 놓친 cue를 커버하는 다음 hop."""
        scorer = self.components.get("hierarchical_scorer")
        if not scorer:
            from components.hierarchical_scorer import HierarchicalScorer
            scorer = HierarchicalScorer()

        seen_ids = {e["leaf_id"] for e in initial_targets}
        covered_cues = set()
        for entry in initial_targets:
            covered_cues.update(c.lower() for c in entry.get("matched_cues", []))

        current_context = initial_context
        current_targets = initial_targets
        best_pred = None
        best_confidence = "low"
        used_visual = False

        for hop in range(max_hops):
            print(f"    [Hop {hop+1}/{max_hops}] targets={len(current_targets)} | "
                  f"covered_cues={len(covered_cues)}/{len(cues)}")

            # Check
            check_result = self._run_check(
                check_method, current_context, question, options, hop_history,
            )
            pre_elimination = check_result.get("elimination", {})

            hop_history.append({
                "stage": "check", "hop": hop + 1,
                "method": check_method,
                "confidence": check_result.get("confidence", "unknown"),
                "answer": check_result.get("answer"),
            })

            # If certain → early answer
            if check_result.get("confidence") == "certain" and check_result.get("answer"):
                best_pred = check_result["answer"]
                best_confidence = "high"
                print(f"    [Hop {hop+1}] Certain! → {best_pred}")
                break

            # Visual if needed
            if self._should_load_visual(visual_method, check_result):
                vis = self._load_visual(
                    current_targets, check_result, video_id,
                    current_context, question, options, hop_history,
                    max_frames, formatter,
                )
                if vis["used"]:
                    used_visual = True
                    if vis.get("observation"):
                        current_context += f"\n\n=== Visual (hop {hop+1}) ===\n{vis['observation']}"
                    if vis.get("answer") and vis.get("confidence") == "high":
                        best_pred = vis["answer"]
                        best_confidence = "high"
                        break

            # If likely with answer → save but continue to see if we can be more sure
            if check_result.get("confidence") == "likely" and check_result.get("answer"):
                best_pred = check_result["answer"]
                best_confidence = "medium"

            # Find gap leaves for next hop
            gap_leaves = scorer.find_gap_leaves(
                score_result, seen_ids, cues, covered_cues,
                budget=leaf_budget,
            )

            if not gap_leaves:
                print(f"    [Hop {hop+1}] No more gap leaves. Done.")
                break

            # Update state
            for entry in gap_leaves:
                seen_ids.add(entry["leaf_id"])
                covered_cues.update(c.lower() for c in entry.get("matched_cues", []))

            current_targets = gap_leaves
            gap_context = formatter.format_leaf_batch(gap_leaves)
            current_context += f"\n\n=== Additional Context (hop {hop+1}) ===\n{gap_context}"

        # Final answer
        if not best_pred or best_confidence != "high":
            ans_result = self._run_answer(
                answer_method, current_context, question, options,
                hop_history, pre_elimination if 'pre_elimination' in dir() else {},
            )
            if ans_result.get("pred"):
                best_pred = ans_result["pred"]
                best_confidence = ans_result.get("confidence", best_confidence)

        return {
            "pred": best_pred,
            "confidence": best_confidence,
            "used_visual": used_visual,
        }

    # ==================== Escalation Search ====================

    def _escalation_search(
        self, tree: dict, all_leaves: list, target_leaves: list,
        score_result: dict, cues: list, initial_context: str,
        question: str, options: list, hop_history: list,
        answer_method: str, video_id: str, max_hops: int,
        max_frames: int, leaf_budget: int, formatter, has_time: bool,
    ) -> dict:
        """Three-stage escalation search:

        1. Context-first: caption + parent hierarchy → try answer
        2. History-aware visual: failure reason → targeted frame observation
        3. Sibling/broader expansion: same group or wider search
        """
        scorer = self.components.get("hierarchical_scorer")
        observer = self.components.get("visual_observer")

        current_context = initial_context
        seen_ids = {e["leaf_id"] for e in target_leaves}
        best_pred = None
        best_confidence = "low"
        used_visual = False
        escalation_level = 0

        # =============================================
        # Stage 1: Context-first answer attempt
        # =============================================
        print(f"    [Escalation 1/3] Context-first answer attempt")
        check_result = self._run_check(
            "uncertainty", current_context, question, options, hop_history,
        )

        hop_history.append({
            "stage": "escalation_check", "level": 1,
            "confidence": check_result.get("confidence"),
            "answer": check_result.get("answer"),
        })

        # If certain → return immediately
        if check_result.get("confidence") == "certain" and check_result.get("answer"):
            return {
                "pred": check_result["answer"],
                "confidence": "high",
                "used_visual": False,
                "escalation_level": 1,
            }

        # Try answer method
        pre_elimination = check_result.get("elimination", {})
        ans_result = self._run_answer(
            answer_method, current_context, question, options,
            hop_history, pre_elimination,
        )

        if ans_result.get("pred") and ans_result.get("confidence") == "high":
            return {
                "pred": ans_result["pred"],
                "confidence": "high",
                "used_visual": False,
                "escalation_level": 1,
            }

        if ans_result.get("pred"):
            best_pred = ans_result["pred"]
            best_confidence = ans_result.get("confidence", "medium")

        escalation_level = 1
        previous_reasoning = check_result.get("reasoning", "")
        if not previous_reasoning:
            previous_reasoning = (
                f"Confidence: {check_result.get('confidence', 'unknown')}. "
                f"Answer attempted: {best_pred or 'none'}"
            )

        # =============================================
        # Stage 2: History-aware visual observation
        # =============================================
        video_path = self.adapter.get_video_path(video_id) if video_id else None

        if observer and video_path and os.path.exists(video_path):
            print(f"    [Escalation 2/3] History-aware visual observation")

            # Determine time ranges from target leaves
            visual_ranges = [
                (float(e["leaf"]["start_time"]), float(e["leaf"]["end_time"]))
                for e in target_leaves[:15] if "start_time" in e.get("leaf", {})
            ]

            if visual_ranges:
                obs_result = observer.observe_and_answer(
                    context=current_context,
                    question=question,
                    options=options,
                    previous_reasoning=previous_reasoning,
                    time_ranges=visual_ranges,
                    video_path=video_path,
                    max_frames=max_frames,
                    hop_history=hop_history,
                )

                used_visual = obs_result.get("used_visual", False)

                hop_history.append({
                    "stage": "visual_observation", "level": 2,
                    "confidence": obs_result.get("confidence"),
                    "answer": obs_result.get("answer"),
                    "key_finding": obs_result.get("key_finding", ""),
                    "observations": obs_result.get("observations", [])[:3],
                })

                if obs_result.get("answer") and obs_result.get("confidence") == "high":
                    return {
                        "pred": obs_result["answer"],
                        "confidence": "high",
                        "used_visual": True,
                        "escalation_level": 2,
                    }

                if obs_result.get("answer"):
                    best_pred = obs_result["answer"]
                    best_confidence = obs_result.get("confidence", best_confidence)

                # Enrich context with observations
                if obs_result.get("observations"):
                    obs_text = "\n".join(f"- {o}" for o in obs_result["observations"])
                    current_context += f"\n\n=== Visual Observations ===\n{obs_text}"
                    if obs_result.get("key_finding"):
                        current_context += f"\nKey finding: {obs_result['key_finding']}"

                previous_reasoning = obs_result.get("reasoning", previous_reasoning)
                if obs_result.get("ambiguity_reason"):
                    previous_reasoning = (
                        f"Visual observation done but still uncertain. "
                        f"Reason: {obs_result['ambiguity_reason']}"
                    )

            escalation_level = 2

        # Medium confidence → try elimination to confirm
        if best_pred and best_confidence == "medium":
            ans_result = self._run_answer(
                answer_method, current_context, question, options,
                hop_history, {},
            )
            if ans_result.get("pred") and ans_result.get("confidence") == "high":
                return {
                    "pred": ans_result["pred"],
                    "confidence": "high",
                    "used_visual": used_visual,
                    "escalation_level": escalation_level,
                }

        # =============================================
        # Stage 3: Sibling + Broader expansion
        # =============================================
        if scorer and (not best_pred or best_confidence == "low"):
            for expansion_round in range(min(max_hops, 3)):
                # First round: siblings (same Level_1 parent)
                # Later rounds: broader (Level_2 parent → other Level_1 groups)
                if expansion_round == 0:
                    new_leaves = scorer.get_sibling_leaves(
                        tree, seen_ids, budget=leaf_budget,
                    )
                    source = "sibling"
                else:
                    new_leaves = scorer.get_broader_leaves(
                        tree, seen_ids, seen_ids, budget=leaf_budget,
                    )
                    source = "broader"

                if not new_leaves and score_result:
                    # Fallback: gap leaves from score_result
                    covered_cues = set()
                    for e in target_leaves:
                        covered_cues.update(
                            c.lower() for c in e.get("matched_cues", [])
                        )
                    new_leaves = scorer.find_gap_leaves(
                        score_result, seen_ids, cues, covered_cues,
                        budget=leaf_budget,
                    )
                    source = "gap"

                if not new_leaves:
                    print(f"    [Escalation 3] No more leaves. Done.")
                    break

                print(f"    [Escalation 3/3] {source} expansion round {expansion_round + 1} "
                      f"({len(new_leaves)} leaves)")

                for entry in new_leaves:
                    seen_ids.add(entry["leaf_id"])

                new_ctx = formatter.format_leaf_batch(new_leaves)
                current_context += (
                    f"\n\n=== {source.title()} Expansion "
                    f"(round {expansion_round + 1}) ===\n{new_ctx}"
                )

                hop_history.append({
                    "stage": f"expansion_{source}",
                    "round": expansion_round + 1,
                    "new_leaves": len(new_leaves),
                })

                # Re-check with expanded context
                check_result = self._run_check(
                    "uncertainty", current_context, question, options,
                    hop_history,
                )

                if (check_result.get("confidence") in ("certain", "likely")
                        and check_result.get("answer")):
                    best_pred = check_result["answer"]
                    best_confidence = (
                        "high" if check_result["confidence"] == "certain"
                        else "medium"
                    )
                    escalation_level = 3
                    if best_confidence == "high":
                        break

                # Visual observation on new expansion leaves
                if observer and video_path and os.path.exists(video_path):
                    new_ranges = [
                        (float(e["leaf"]["start_time"]),
                         float(e["leaf"]["end_time"]))
                        for e in new_leaves if "start_time" in e.get("leaf", {})
                    ]
                    if new_ranges:
                        obs_result = observer.observe_and_answer(
                            context=current_context,
                            question=question,
                            options=options,
                            previous_reasoning=previous_reasoning,
                            time_ranges=new_ranges,
                            video_path=video_path,
                            max_frames=max_frames,
                            hop_history=hop_history,
                        )
                        used_visual = True

                        if (obs_result.get("answer")
                                and obs_result.get("confidence") == "high"):
                            return {
                                "pred": obs_result["answer"],
                                "confidence": "high",
                                "used_visual": True,
                                "escalation_level": 3,
                            }

                        if obs_result.get("answer"):
                            best_pred = obs_result["answer"]
                            best_confidence = obs_result.get(
                                "confidence", best_confidence
                            )

                        if obs_result.get("observations"):
                            obs_text = "\n".join(
                                f"- {o}" for o in obs_result["observations"]
                            )
                            current_context += (
                                f"\n\n=== Visual ({source}) ===\n{obs_text}"
                            )

                escalation_level = 3

        # Final answer attempt
        if not best_pred or best_confidence not in ("high",):
            ans_result = self._run_answer(
                answer_method, current_context, question, options,
                hop_history, {},
            )
            if ans_result.get("pred"):
                best_pred = ans_result["pred"]
                best_confidence = ans_result.get("confidence", best_confidence)

        return {
            "pred": best_pred,
            "confidence": best_confidence,
            "used_visual": used_visual,
            "escalation_level": escalation_level,
        }

    # ==================== Check ====================

    def _run_check(self, method: str, context: str, question: str,
                   options: list, hop_history: list) -> dict:
        if method == "uncertainty":
            checker = self.components.get("uncertainty_checker")
            if checker and checker.llm_fn:
                return checker.assess(context, question, options, hop_history)
            return {"confidence": "uncertain", "answer": None, "elimination": {}}

        elif method == "solvability":
            checker = self.components.get("solvability")
            if checker and checker.llm_fn:
                result = checker.check(context, question, options, hop_history)
                return {
                    "confidence": "certain" if result["solvable"] else "uncertain",
                    "answer": result.get("answer"),
                    "needs_visual": result.get("needs_depth", False),
                    "elimination": {},
                }
            return {"confidence": "uncertain", "answer": None, "elimination": {}}

        elif method == "none":
            return {"confidence": "uncertain", "answer": None, "elimination": {}}

        return {"confidence": "uncertain", "answer": None, "elimination": {}}

    # ==================== Visual ====================

    def _should_load_visual(self, method: str, check_result: dict) -> bool:
        if method == "always":
            return True
        if method == "never":
            return False
        if method == "on_uncertainty":
            return check_result.get("needs_visual", False) or \
                   check_result.get("confidence") in ("uncertain", "insufficient")
        return False

    def _load_visual(self, targets: list, check_result: dict,
                     video_id: str, context: str, question: str,
                     options: list, hop_history: list,
                     max_frames: int, formatter) -> dict:
        frame_loader = self.components.get("frame_loader")
        vision_vlm = self.components.get("vision_vlm")

        if not frame_loader or not vision_vlm:
            return {"used": False}

        video_path = self.adapter.get_video_path(video_id)
        if not video_path or not os.path.exists(video_path):
            return {"used": False}

        # Determine time ranges
        visual_ranges = check_result.get("visual_time_ranges", [])
        if not visual_ranges:
            visual_ranges = [
                (float(e["leaf"]["start_time"]), float(e["leaf"]["end_time"]))
                for e in targets[:10] if "start_time" in e.get("leaf", {})
            ]

        if not visual_ranges:
            return {"used": False}

        visual_ranges = [(float(r[0]), float(r[1])) for r in visual_ranges]
        visual_ranges.sort()

        try:
            frames_np, frame_secs = frame_loader.load(
                video_path, visual_ranges, max_frames=max_frames
            )
        except Exception as e:
            print(f"      [Visual] Error: {e}")
            return {"used": False}

        if frames_np is None:
            return {"used": False}

        vlm_result = vision_vlm.infer(
            frames_np, context, question, options, hop_history
        )

        ans = vlm_result.get("answer", "")
        if ans:
            m = re.search(r"[ABCD]", ans.upper())
            ans = m.group(0) if m else None

        del frames_np
        gc.collect()

        return {
            "used": True,
            "answer": ans,
            "confidence": vlm_result.get("confidence", "medium"),
            "observation": vlm_result.get("observation", ""),
        }

    # ==================== Answer ====================

    def _run_answer(self, method: str, context: str, question: str,
                    options: list, hop_history: list,
                    pre_elimination: dict) -> dict:
        if method == "elimination":
            reasoner = self.components.get("elimination_reasoner")
            if reasoner and reasoner.llm_fn:
                result = reasoner.eliminate(
                    context, question, options,
                    hop_history=hop_history,
                    pre_elimination=pre_elimination if pre_elimination.get("eliminated") else None,
                )
                return {"pred": result["answer"], "confidence": result["confidence"]}
            return {"pred": None, "confidence": "low"}

        elif method == "solvability":
            checker = self.components.get("solvability")
            if checker and checker.llm_fn:
                result = checker.check(context, question, options, hop_history)
                return {"pred": result.get("answer"), "confidence": "high" if result["solvable"] else "low"}
            return {"pred": None, "confidence": "low"}

        elif method == "direct":
            simple_vlm = self.components.get("simple_vlm")
            if simple_vlm:
                from components.answer_parser import extract_choice_letter
                opt_str = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
                prompt = f"Question: {question}\n{opt_str}\nContext: {context[:8000]}\nAnswer:"
                raw = simple_vlm.infer(prompt, max_new_tokens=2)
                letter = extract_choice_letter(raw)
                return {"pred": letter, "confidence": "medium"}
            return {"pred": None, "confidence": "low"}

        return {"pred": None, "confidence": "low"}
