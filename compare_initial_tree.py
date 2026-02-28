#!/usr/bin/env python3
"""
compare_initial_tree.py — Compare initial tree initialization quality between two experiment runs.

Usage:
    python compare_initial_tree.py DIR_A DIR_B [--detail]

Example:
    python compare_initial_tree.py \
        output/video_mme_full_semantic/v2 \
        output/video_mme_full_visual/v1 \
        --detail
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_results(directory: str) -> dict:
    """Load all by_qid/*.json results into a dict keyed by question ID."""
    by_qid_dir = os.path.join(directory, "by_qid")
    if not os.path.isdir(by_qid_dir):
        print(f"ERROR: {by_qid_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = {}
    for fn in sorted(os.listdir(by_qid_dir)):
        if not fn.endswith(".json"):
            continue
        qid = fn.replace(".json", "")
        with open(os.path.join(by_qid_dir, fn)) as f:
            results[qid] = json.load(f)
    return results


def fmt_pct(n: float, total: float) -> str:
    """Format a count/total as 'N/T (XX.X%)'."""
    if total == 0:
        return "0/0 (--)"
    return f"{int(n)}/{int(total)} ({100 * n / total:.1f}%)"


def fmt_float(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}" if v is not None else "--"


def bar(value: float, width: int = 20) -> str:
    """Tiny inline bar for visual comparison."""
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def separator(char: str = "─", width: int = 90) -> str:
    return char * width


def header(title: str, width: int = 90) -> str:
    pad = width - len(title) - 4
    left = pad // 2
    right = pad - left
    return f"{'─' * left}[ {title} ]{'─' * right}"


# ──────────────────────────────────────────────────────────────────────────────
# Per-question feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(data: dict) -> dict:
    """Pull the comparison-relevant features from a single question result."""
    total_leaves = data.get("total_leaves", 0)
    active_leaves = data.get("active_leaves", 0)
    activation_rate = active_leaves / total_leaves if total_leaves > 0 else 0.0

    total_hops = data.get("total_hops", 0)
    correct = data.get("correct", False)
    pred = data.get("pred")
    answer = data.get("answer")
    confidence = data.get("confidence", "")

    # Hop 1 judge info
    hop1_conf = None
    hop1_answer = None
    hop1_answerable = None
    traversal_log = data.get("traversal_log", [])
    if traversal_log:
        judge = traversal_log[0].get("judge", {})
        hop1_conf = judge.get("confidence")
        hop1_answer = judge.get("answer")
        hop1_answerable = judge.get("answerable")

    # Hop 1 target leaves (initial targets)
    hop_contexts = data.get("hop_contexts", [])
    hop1_targets = []
    hop1_target_times = set()
    if hop_contexts:
        hop1_targets = hop_contexts[0].get("target_leaves", [])
        for tl in hop1_targets:
            hop1_target_times.add((tl.get("start_time"), tl.get("end_time")))

    # Semantic match info
    semantic_match = data.get("semantic_match")
    has_semantic = semantic_match is not None and bool(semantic_match)
    top_scores = []
    if has_semantic and isinstance(semantic_match, dict):
        for entry in semantic_match.get("top_scores", []):
            top_scores.append(entry.get("score", 0.0))

    return {
        "total_leaves": total_leaves,
        "active_leaves": active_leaves,
        "activation_rate": activation_rate,
        "total_hops": total_hops,
        "correct": correct,
        "pred": pred,
        "answer": answer,
        "confidence": confidence,
        "hop1_conf": hop1_conf,
        "hop1_answer": hop1_answer,
        "hop1_answerable": hop1_answerable,
        "hop1_target_count": len(hop1_targets),
        "hop1_target_times": hop1_target_times,
        "has_semantic": has_semantic,
        "semantic_top_scores": top_scores,
        "question_type": data.get("question_type", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metric aggregation
# ──────────────────────────────────────────────────────────────────────────────

def compute_aggregate(features_by_qid: dict) -> dict:
    """Compute aggregate statistics over all questions."""
    n = len(features_by_qid)
    if n == 0:
        return {}

    correct_count = sum(1 for f in features_by_qid.values() if f["correct"])
    activation_rates = [f["activation_rate"] for f in features_by_qid.values()]
    total_hops_list = [f["total_hops"] for f in features_by_qid.values()]

    # Hop 1 solve rate: hop1 gave high confidence AND correct answer
    hop1_high_conf = sum(
        1 for f in features_by_qid.values()
        if f["hop1_conf"] == "high"
    )
    hop1_solved = sum(
        1 for f in features_by_qid.values()
        if f["hop1_conf"] == "high" and f["hop1_answer"] == f["answer"]
    )

    # Hop distribution
    hop_dist = defaultdict(int)
    for f in features_by_qid.values():
        hop_dist[f["total_hops"]] += 1

    # Hop-bucketed accuracy
    hop_correct = defaultdict(int)
    hop_total = defaultdict(int)
    for f in features_by_qid.values():
        h = f["total_hops"]
        hop_total[h] += 1
        if f["correct"]:
            hop_correct[h] += 1

    # Semantic scores (only for those that have them)
    all_top1_scores = []
    all_top_scores_flat = []
    for f in features_by_qid.values():
        if f["semantic_top_scores"]:
            all_top1_scores.append(f["semantic_top_scores"][0])
            all_top_scores_flat.extend(f["semantic_top_scores"])

    sem_count = sum(1 for f in features_by_qid.values() if f["has_semantic"])

    return {
        "n": n,
        "accuracy": correct_count / n,
        "correct_count": correct_count,
        "avg_activation_rate": sum(activation_rates) / n,
        "min_activation_rate": min(activation_rates),
        "max_activation_rate": max(activation_rates),
        "avg_hops": sum(total_hops_list) / n,
        "hop1_high_conf_count": hop1_high_conf,
        "hop1_high_conf_rate": hop1_high_conf / n,
        "hop1_solved_count": hop1_solved,
        "hop1_solve_rate": hop1_solved / n,
        "hop_dist": dict(sorted(hop_dist.items())),
        "hop_correct": dict(sorted(hop_correct.items())),
        "hop_total": dict(sorted(hop_total.items())),
        "semantic_count": sem_count,
        "semantic_top1_scores": all_top1_scores,
        "semantic_all_scores": all_top_scores_flat,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Printing
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison(name_a: str, agg_a: dict, name_b: str, agg_b: dict,
                     feats_a: dict, feats_b: dict, common_qids: list,
                     show_detail: bool):
    """Print side-by-side comparison tables."""

    W = 92
    print()
    print(header("INITIAL TREE COMPARISON", W))
    print(f"  A: {name_a}  ({agg_a['n']} questions)")
    print(f"  B: {name_b}  ({agg_b['n']} questions)")
    print(f"  Common questions: {len(common_qids)}")
    print(separator(width=W))

    # ── 1. Overall accuracy ──
    print()
    print(header("1. Overall Accuracy", W))
    print(f"  {'Metric':<35} {'A':>20} {'B':>20}")
    print(f"  {'─' * 35} {'─' * 20} {'─' * 20}")
    print(f"  {'Accuracy':<35} {fmt_pct(agg_a['correct_count'], agg_a['n']):>20} {fmt_pct(agg_b['correct_count'], agg_b['n']):>20}")

    # Also compute accuracy only on common questions
    common_correct_a = sum(1 for q in common_qids if feats_a[q]["correct"])
    common_correct_b = sum(1 for q in common_qids if feats_b[q]["correct"])
    nc = len(common_qids)
    print(f"  {'Accuracy (common Qs only)':<35} {fmt_pct(common_correct_a, nc):>20} {fmt_pct(common_correct_b, nc):>20}")

    # ── 2. Active leaves ──
    print()
    print(header("2. Tree Activation (active/total leaves)", W))
    print(f"  {'Metric':<35} {'A':>20} {'B':>20}")
    print(f"  {'─' * 35} {'─' * 20} {'─' * 20}")
    print(f"  {'Avg activation rate':<35} {fmt_float(agg_a['avg_activation_rate']):>20} {fmt_float(agg_b['avg_activation_rate']):>20}")
    print(f"  {'Min activation rate':<35} {fmt_float(agg_a['min_activation_rate']):>20} {fmt_float(agg_b['min_activation_rate']):>20}")
    print(f"  {'Max activation rate':<35} {fmt_float(agg_a['max_activation_rate']):>20} {fmt_float(agg_b['max_activation_rate']):>20}")

    # ── 3. Hop 1 solve rate ──
    print()
    print(header("3. Hop 1 Quality (first-hop effectiveness)", W))
    print(f"  {'Metric':<35} {'A':>20} {'B':>20}")
    print(f"  {'─' * 35} {'─' * 20} {'─' * 20}")
    print(f"  {'Hop 1 high-conf rate':<35} {fmt_pct(agg_a['hop1_high_conf_count'], agg_a['n']):>20} {fmt_pct(agg_b['hop1_high_conf_count'], agg_b['n']):>20}")
    print(f"  {'Hop 1 solve rate (high & correct)':<35} {fmt_pct(agg_a['hop1_solved_count'], agg_a['n']):>20} {fmt_pct(agg_b['hop1_solved_count'], agg_b['n']):>20}")
    print(f"  {'Avg total hops used':<35} {fmt_float(agg_a['avg_hops'], 2):>20} {fmt_float(agg_b['avg_hops'], 2):>20}")

    # ── 4. Hop distribution ──
    print()
    print(header("4. Hop Distribution & Accuracy by Hop Count", W))
    all_hops = sorted(set(list(agg_a["hop_dist"].keys()) + list(agg_b["hop_dist"].keys())))
    print(f"  {'Hops':<6} {'A count':>10} {'A acc':>12} {'B count':>10} {'B acc':>12}")
    print(f"  {'─' * 6} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 12}")
    for h in all_hops:
        ca = agg_a["hop_correct"].get(h, 0)
        ta = agg_a["hop_total"].get(h, 0)
        cb = agg_b["hop_correct"].get(h, 0)
        tb = agg_b["hop_total"].get(h, 0)
        acc_a = f"{100 * ca / ta:.1f}%" if ta else "--"
        acc_b = f"{100 * cb / tb:.1f}%" if tb else "--"
        print(f"  {h:<6} {ta:>10} {acc_a:>12} {tb:>10} {acc_b:>12}")

    # ── 5. Semantic match scores ──
    print()
    print(header("5. Semantic Match Mode", W))
    print(f"  {'Metric':<35} {'A':>20} {'B':>20}")
    print(f"  {'─' * 35} {'─' * 20} {'─' * 20}")
    print(f"  {'Has semantic_match':<35} {agg_a['semantic_count']:>20} {agg_b['semantic_count']:>20}")
    if agg_a["semantic_top1_scores"]:
        scores = agg_a["semantic_top1_scores"]
        print(f"  A top-1 semantic scores:  min={min(scores):.3f}  mean={sum(scores)/len(scores):.3f}  max={max(scores):.3f}  (n={len(scores)})")
    else:
        print(f"  A: no semantic match scores")
    if agg_b["semantic_top1_scores"]:
        scores = agg_b["semantic_top1_scores"]
        print(f"  B top-1 semantic scores:  min={min(scores):.3f}  mean={sum(scores)/len(scores):.3f}  max={max(scores):.3f}  (n={len(scores)})")
    else:
        print(f"  B: no semantic match scores")

    # Score distribution buckets for whichever has scores
    for label, agg in [("A", agg_a), ("B", agg_b)]:
        if agg["semantic_top1_scores"]:
            scores = sorted(agg["semantic_top1_scores"])
            n_s = len(scores)
            print(f"  {label} top-1 score percentiles: p10={scores[int(n_s*0.1)]:.3f}  p25={scores[int(n_s*0.25)]:.3f}  p50={scores[int(n_s*0.5)]:.3f}  p75={scores[int(n_s*0.75)]:.3f}  p90={scores[int(n_s*0.9)]:.3f}")

    # ── 6. Initial target selection comparison ──
    print()
    print(header("6. Initial Target Selection Quality (common Qs)", W))

    # How often do the two experiments pick different initial targets?
    same_targets = 0
    diff_targets = 0
    diff_both_correct = 0
    diff_a_only_correct = 0
    diff_b_only_correct = 0
    diff_both_wrong = 0
    overlap_ratios = []

    for qid in common_qids:
        fa = feats_a[qid]
        fb = feats_b[qid]
        times_a = fa["hop1_target_times"]
        times_b = fb["hop1_target_times"]

        if times_a == times_b:
            same_targets += 1
        else:
            diff_targets += 1
            # Overlap ratio (Jaccard)
            if times_a or times_b:
                inter = len(times_a & times_b)
                union = len(times_a | times_b)
                overlap_ratios.append(inter / union if union > 0 else 0.0)

            if fa["correct"] and fb["correct"]:
                diff_both_correct += 1
            elif fa["correct"] and not fb["correct"]:
                diff_a_only_correct += 1
            elif not fa["correct"] and fb["correct"]:
                diff_b_only_correct += 1
            else:
                diff_both_wrong += 1

    print(f"  Same initial targets:        {same_targets:>6}  ({100*same_targets/nc:.1f}%)")
    print(f"  Different initial targets:    {diff_targets:>6}  ({100*diff_targets/nc:.1f}%)")
    if overlap_ratios:
        avg_overlap = sum(overlap_ratios) / len(overlap_ratios)
        print(f"    Avg Jaccard overlap:        {avg_overlap:.3f}")
    print()
    print(f"  When targets differ, outcomes:")
    print(f"    Both correct:               {diff_both_correct:>6}")
    print(f"    Only A correct:             {diff_a_only_correct:>6}")
    print(f"    Only B correct:             {diff_b_only_correct:>6}")
    print(f"    Both wrong:                 {diff_both_wrong:>6}")
    if diff_targets > 0:
        print(f"    A advantage (A only / diff):{100 * diff_a_only_correct / diff_targets:>6.1f}%")
        print(f"    B advantage (B only / diff):{100 * diff_b_only_correct / diff_targets:>6.1f}%")

    # ── 7. Disagreement analysis ──
    print()
    print(header("7. Answer Disagreement Analysis (common Qs)", W))
    agree_correct = 0
    agree_wrong = 0
    a_correct_b_wrong = 0
    b_correct_a_wrong = 0
    disagree_both_wrong = 0

    disagree_qids = []  # for detail mode

    for qid in common_qids:
        fa = feats_a[qid]
        fb = feats_b[qid]
        same_pred = (fa["pred"] == fb["pred"])
        if same_pred:
            if fa["correct"]:
                agree_correct += 1
            else:
                agree_wrong += 1
        else:
            if fa["correct"] and not fb["correct"]:
                a_correct_b_wrong += 1
                disagree_qids.append((qid, "A_wins"))
            elif not fa["correct"] and fb["correct"]:
                b_correct_a_wrong += 1
                disagree_qids.append((qid, "B_wins"))
            else:
                disagree_both_wrong += 1
                disagree_qids.append((qid, "both_wrong"))

    total_disagree = a_correct_b_wrong + b_correct_a_wrong + disagree_both_wrong
    print(f"  Agree & correct:              {agree_correct:>6}  ({100*agree_correct/nc:.1f}%)")
    print(f"  Agree & wrong:                {agree_wrong:>6}  ({100*agree_wrong/nc:.1f}%)")
    print(f"  Disagree (A correct only):    {a_correct_b_wrong:>6}  ({100*a_correct_b_wrong/nc:.1f}%)")
    print(f"  Disagree (B correct only):    {b_correct_a_wrong:>6}  ({100*b_correct_a_wrong/nc:.1f}%)")
    print(f"  Disagree (both wrong):        {disagree_both_wrong:>6}  ({100*disagree_both_wrong/nc:.1f}%)")
    if total_disagree > 0:
        print(f"  Net A advantage in disagree:  {a_correct_b_wrong - b_correct_a_wrong:>+6} questions")

    # ── 8. Hop 1 confidence vs final outcome ──
    print()
    print(header("8. Hop 1 Confidence vs Final Correctness (common Qs)", W))
    for label, feats in [("A", feats_a), ("B", feats_b)]:
        conf_buckets = defaultdict(lambda: {"correct": 0, "total": 0})
        for qid in common_qids:
            f = feats[qid]
            c = f["hop1_conf"] or "none"
            conf_buckets[c]["total"] += 1
            if f["correct"]:
                conf_buckets[c]["correct"] += 1
        print(f"  Experiment {label}:")
        for c in ["high", "medium", "low", "none"]:
            b = conf_buckets[c]
            if b["total"] > 0:
                acc = 100 * b["correct"] / b["total"]
                print(f"    hop1_conf={c:<7}  n={b['total']:<5}  final_accuracy={acc:.1f}%  {bar(acc/100, 15)}")
        print()

    # ── 9. Visual bar comparison ──
    print(header("9. Visual Summary", W))
    acc_a = agg_a["accuracy"]
    acc_b = agg_b["accuracy"]
    print(f"  A accuracy: {bar(acc_a)} {100*acc_a:.1f}%")
    print(f"  B accuracy: {bar(acc_b)} {100*acc_b:.1f}%")
    print()
    h1a = agg_a["hop1_solve_rate"]
    h1b = agg_b["hop1_solve_rate"]
    print(f"  A hop1-solve: {bar(h1a)} {100*h1a:.1f}%")
    print(f"  B hop1-solve: {bar(h1b)} {100*h1b:.1f}%")
    print()
    act_a = agg_a["avg_activation_rate"]
    act_b = agg_b["avg_activation_rate"]
    print(f"  A activation: {bar(act_a)} {act_a:.3f}")
    print(f"  B activation: {bar(act_b)} {act_b:.3f}")
    print(separator(width=W))

    # ── Detail mode ──
    if show_detail and disagree_qids:
        print()
        print(header("DETAIL: Disagreement Breakdown (first 30)", W))
        print(f"  {'QID':<12} {'Result':<12} {'A pred':>7} {'B pred':>7} {'Ans':>5} {'A hops':>7} {'B hops':>7} {'A h1conf':>9} {'B h1conf':>9} {'Tgt overlap':>12}")
        print(f"  {'─'*12} {'─'*12} {'─'*7} {'─'*7} {'─'*5} {'─'*7} {'─'*7} {'─'*9} {'─'*9} {'─'*12}")
        for qid, outcome in disagree_qids[:30]:
            fa = feats_a[qid]
            fb = feats_b[qid]
            ta = fa["hop1_target_times"]
            tb = fb["hop1_target_times"]
            if ta or tb:
                inter = len(ta & tb)
                union = len(ta | tb)
                jac = f"{inter/union:.2f}" if union > 0 else "--"
            else:
                jac = "--"
            print(f"  {qid:<12} {outcome:<12} {fa['pred'] or '-':>7} {fb['pred'] or '-':>7} {fa['answer'] or '-':>5} {fa['total_hops']:>7} {fb['total_hops']:>7} {fa['hop1_conf'] or '-':>9} {fb['hop1_conf'] or '-':>9} {jac:>12}")

        # Show hop distribution for disagreements
        print()
        print("  Disagreement hop-count breakdown:")
        for outcome_label in ["A_wins", "B_wins", "both_wrong"]:
            subset = [qid for qid, o in disagree_qids if o == outcome_label]
            if not subset:
                continue
            hops_a = [feats_a[q]["total_hops"] for q in subset]
            hops_b = [feats_b[q]["total_hops"] for q in subset]
            avg_a = sum(hops_a) / len(hops_a)
            avg_b = sum(hops_b) / len(hops_b)
            print(f"    {outcome_label:<12}  n={len(subset):<4}  avg_hops A={avg_a:.2f}  B={avg_b:.2f}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare initial tree initialization quality between two experiment runs."
    )
    parser.add_argument("dir_a", help="Path to first experiment directory (contains by_qid/)")
    parser.add_argument("dir_b", help="Path to second experiment directory (contains by_qid/)")
    parser.add_argument("--detail", action="store_true",
                        help="Show per-question detail for disagreements")
    args = parser.parse_args()

    name_a = os.path.basename(os.path.normpath(args.dir_a))
    parent_a = os.path.basename(os.path.dirname(os.path.normpath(args.dir_a)))
    label_a = f"{parent_a}/{name_a}"

    name_b = os.path.basename(os.path.normpath(args.dir_b))
    parent_b = os.path.basename(os.path.dirname(os.path.normpath(args.dir_b)))
    label_b = f"{parent_b}/{name_b}"

    print(f"Loading A: {args.dir_a} ...")
    results_a = load_results(args.dir_a)
    print(f"  Loaded {len(results_a)} questions")

    print(f"Loading B: {args.dir_b} ...")
    results_b = load_results(args.dir_b)
    print(f"  Loaded {len(results_b)} questions")

    common_qids = sorted(set(results_a.keys()) & set(results_b.keys()))
    only_a = sorted(set(results_a.keys()) - set(results_b.keys()))
    only_b = sorted(set(results_b.keys()) - set(results_a.keys()))

    print(f"  Common: {len(common_qids)}  |  Only in A: {len(only_a)}  |  Only in B: {len(only_b)}")

    # Extract features
    feats_a = {qid: extract_features(results_a[qid]) for qid in results_a}
    feats_b = {qid: extract_features(results_b[qid]) for qid in results_b}

    # Aggregate over ALL questions in each experiment
    agg_a = compute_aggregate(feats_a)
    agg_b = compute_aggregate(feats_b)

    print_comparison(
        label_a, agg_a,
        label_b, agg_b,
        feats_a, feats_b,
        common_qids,
        show_detail=args.detail,
    )


if __name__ == "__main__":
    main()
