#!/usr/bin/env python3
"""
Video-MME Subset Selector for Ablation Experiments
===================================================

Selects an optimal subset (~100-150 questions) from the full 900-question
Video-MME evaluation that best guides finding the right agent combination.

Selection criteria:
  1. All Video-MME categories covered (domain, sub_category, task_type)
  2. Bias toward hard questions (wrong answers overrepresented)
  3. Include some easy questions (to detect regressions)
  4. Diverse error patterns across failure modes
  5. Target size: ~100-150 questions

Error pattern taxonomy:
  - "hop1_high_conf_wrong"  : Answered at hop 1 with high confidence, but wrong
                              (bad initial retrieval or flawed reasoning)
  - "mid_hop_wrong"         : Answered at hops 2-4, but wrong
                              (partial search didn't help)
  - "exhaustive_wrong"      : Reached 5 hops via tree_search_hop5, still wrong
                              (thorough search + still wrong = hard problem)
  - "fallback_wrong"        : Method is tree_search_fallback (never found answer)
                              (complete retrieval failure)
  - "medium_conf_wrong"     : Final confidence is medium and wrong (close calls)
  - "easy_correct"          : Correct at hop 1 with high confidence
  - "hard_correct"          : Correct but needed 3+ hops
  - "correct_other"         : All other correct answers

Usage:
  python select_subset.py [--results-dir DIR] [--category-dir DIR]
                          [--output FILE] [--target-size N] [--seed S]
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> dict:
    """Load all per-question result JSONs."""
    data = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        qid = fname.replace(".json", "")
        with open(os.path.join(results_dir, fname)) as f:
            data[qid] = json.load(f)
    return data


def load_categories(category_dir: str) -> dict:
    """Load all per-question category/metadata JSONs."""
    data = {}
    for fname in sorted(os.listdir(category_dir)):
        if not fname.endswith(".json"):
            continue
        qid = fname.replace(".json", "")
        with open(os.path.join(category_dir, fname)) as f:
            data[qid] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Error-pattern classification
# ---------------------------------------------------------------------------

def classify_error_pattern(result: dict) -> str:
    """Classify a single result into an error-pattern bucket.

    Returns one of:
      hop1_high_conf_wrong, mid_hop_wrong, exhaustive_wrong, fallback_wrong,
      medium_conf_wrong, easy_correct, hard_correct, correct_other
    """
    correct = result.get("correct", False)
    method = result.get("method", "")
    hops = result.get("total_hops", 0)
    confidence = result.get("confidence", "unknown")

    if not correct:
        # Prioritise the most distinctive failure modes
        if method == "tree_search_fallback":
            return "fallback_wrong"
        if hops == 1 and confidence == "high":
            return "hop1_high_conf_wrong"
        if confidence == "medium":
            return "medium_conf_wrong"
        if hops >= 5:
            return "exhaustive_wrong"
        # hops 2-4 wrong
        return "mid_hop_wrong"
    else:
        if hops == 1 and confidence == "high":
            return "easy_correct"
        if hops >= 3:
            return "hard_correct"
        return "correct_other"


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------

def build_selection(
    results: dict,
    categories: dict,
    target_size: int = 130,
    seed: int = 42,
) -> list[str]:
    """Build a stratified subset of question IDs.

    Strategy (applied in order):

      Phase 1 - Mandatory category coverage (~42 questions):
        Guarantee at least 1 question from each sub_category (30 unique)
        and each task_type (12 unique).  Uses wrong answers when possible
        and tries to double-cover (pick a question that covers both a
        sub_category and a task_type that still need coverage).

      Phase 2 - Error-pattern quotas (~82 more questions):
        Fill explicit quotas for each of the 8 error patterns.  Picks
        are spread across sub-categories via round-robin to maintain
        category diversity.  This phase explicitly includes correct-answer
        patterns (easy_correct, hard_correct, correct_other) to provide
        regression canaries.

      Phase 3 - Final size adjustment:
        If over target, trim the least diagnostic patterns first
        (correct_other, then easy_correct), while preserving mandatory
        coverage.  If under target, add more wrong answers from the
        most under-represented sub-categories.
    """
    rng = random.Random(seed)

    # Pre-compute per-question metadata
    qid_meta = {}
    for qid in results:
        r = results[qid]
        c = categories.get(qid, {})
        qid_meta[qid] = {
            "correct": r.get("correct", False),
            "confidence": r.get("confidence", "unknown"),
            "total_hops": r.get("total_hops", 0),
            "method": r.get("method", ""),
            "question_type": r.get("question_type", "unknown"),
            "used_visual": r.get("used_visual", False),
            "domain": c.get("domain", "unknown"),
            "sub_category": c.get("sub_category", "unknown"),
            "task_type": c.get("task_type", "unknown"),
            "error_pattern": classify_error_pattern(r),
        }

    selected = set()
    # Track which coverage requirements are still unfulfilled
    all_subcats = set(m["sub_category"] for m in qid_meta.values())
    all_tasks = set(m["task_type"] for m in qid_meta.values())
    covered_subcats = set()
    covered_tasks = set()

    # ---- Phase 1: Mandatory category coverage ----
    # First pass: try to pick questions that cover both a missing sub_category
    # AND a missing task_type simultaneously.
    all_qids = list(qid_meta.keys())
    rng.shuffle(all_qids)

    # Sort: wrong answers first (to bias toward hard), then by how many
    # uncovered dimensions they fill (greedy set-cover heuristic).
    def coverage_score(qid):
        m = qid_meta[qid]
        sc_new = 1 if m["sub_category"] not in covered_subcats else 0
        tt_new = 1 if m["task_type"] not in covered_tasks else 0
        wrong_bonus = 1 if not m["correct"] else 0
        return (sc_new + tt_new, wrong_bonus)

    while covered_subcats != all_subcats or covered_tasks != all_tasks:
        # Re-sort candidates by coverage value
        candidates = [q for q in all_qids if q not in selected]
        candidates.sort(key=coverage_score, reverse=True)
        if not candidates:
            break
        best = candidates[0]
        if coverage_score(best)[0] == 0:
            # Nothing left to cover
            break
        selected.add(best)
        m = qid_meta[best]
        covered_subcats.add(m["sub_category"])
        covered_tasks.add(m["task_type"])

    phase1_count = len(selected)

    # ---- Phase 2: Error-pattern quotas ----
    # Allocate remaining budget across error patterns.
    # Target: ~70% wrong, ~30% correct in final subset.
    # With target_size=130 and ~42 from phase 1 (mostly wrong),
    # we have ~88 slots.  Target ~25 correct total.
    remaining_budget = target_size - len(selected)

    # Count how many of each pattern we already have from phase 1
    phase1_pattern_counts = Counter(
        qid_meta[q]["error_pattern"] for q in selected
    )

    # Desired total counts per pattern (sums to ~target_size)
    error_quotas = {
        # Wrong patterns (~70% of total = ~91)
        "hop1_high_conf_wrong": 22,   # Confident but wrong at first hop
        "mid_hop_wrong": 18,          # Partial search didn't help
        "exhaustive_wrong": 20,       # 5 hops still wrong (hard problems)
        "fallback_wrong": 16,         # Complete retrieval failure
        "medium_conf_wrong": 12,      # Close calls
        # Correct patterns (~30% of total = ~39)
        "easy_correct": 18,           # Regression canaries (must stay correct)
        "hard_correct": 14,           # Hard but solvable (guides us to keep wins)
        "correct_other": 10,          # Moderate correct
    }

    for pattern, target_count in error_quotas.items():
        already_have = phase1_pattern_counts.get(pattern, 0)
        needed = max(0, target_count - already_have)
        if needed == 0:
            continue

        # Build pool of candidates for this pattern, not yet selected
        pool = [
            q for q in qid_meta
            if q not in selected and qid_meta[q]["error_pattern"] == pattern
        ]
        if not pool:
            continue

        # Spread picks across sub-categories via round-robin
        by_subcat = defaultdict(list)
        for q in pool:
            by_subcat[qid_meta[q]["sub_category"]].append(q)
        # Shuffle within each sub-category
        for sc in by_subcat:
            rng.shuffle(by_subcat[sc])

        subcats = sorted(by_subcat.keys())
        rng.shuffle(subcats)
        added = 0
        idx = 0
        max_iters = needed * len(subcats) + len(subcats)  # safety bound
        iters = 0
        while added < needed and iters < max_iters:
            sc = subcats[idx % len(subcats)]
            if by_subcat[sc]:
                selected.add(by_subcat[sc].pop(0))
                added += 1
            idx += 1
            iters += 1
            # Check if all pools exhausted
            if all(len(by_subcat[s]) == 0 for s in subcats):
                break

    # ---- Phase 3: Final size adjustment ----
    # If under target, fill with wrong answers from under-represented sub_categories
    if len(selected) < target_size:
        subcat_counts = Counter(qid_meta[q]["sub_category"] for q in selected)
        # Find under-represented sub-categories
        remaining = [q for q in qid_meta if q not in selected]
        rng.shuffle(remaining)
        # Sort by how under-represented their sub_category is
        remaining.sort(key=lambda q: subcat_counts.get(qid_meta[q]["sub_category"], 0))
        # Bias: prefer wrong answers
        wrong_remaining = [q for q in remaining if not qid_meta[q]["correct"]]
        correct_remaining = [q for q in remaining if qid_meta[q]["correct"]]
        for q in wrong_remaining:
            if len(selected) >= target_size:
                break
            selected.add(q)
        for q in correct_remaining:
            if len(selected) >= target_size:
                break
            selected.add(q)

    # If over target, trim least-diagnostic patterns first, but protect
    # mandatory coverage (phase 1 picks).
    if len(selected) > target_size:
        # Build set of phase-1-essential qids (those that are the sole
        # representative of a sub_category or task_type)
        essential = set()
        for sc in all_subcats:
            sc_qids = [q for q in selected if qid_meta[q]["sub_category"] == sc]
            if len(sc_qids) == 1:
                essential.add(sc_qids[0])
        for tt in all_tasks:
            tt_qids = [q for q in selected if qid_meta[q]["task_type"] == tt]
            if len(tt_qids) == 1:
                essential.add(tt_qids[0])

        trim_order = [
            "correct_other", "easy_correct", "mid_hop_wrong",
            "hard_correct", "hop1_high_conf_wrong",
        ]
        for pattern in trim_order:
            candidates = [
                q for q in selected
                if qid_meta[q]["error_pattern"] == pattern and q not in essential
            ]
            rng.shuffle(candidates)
            while len(selected) > target_size and candidates:
                selected.discard(candidates.pop())

    return sorted(selected), qid_meta


# ---------------------------------------------------------------------------
# Output and statistics
# ---------------------------------------------------------------------------

def print_statistics(selected: list[str], qid_meta: dict, categories: dict):
    """Print detailed statistics about the selected subset."""
    total = len(selected)
    correct = sum(1 for q in selected if qid_meta[q]["correct"])
    wrong = total - correct

    print("=" * 72)
    print(f"  SUBSET SELECTION SUMMARY")
    print(f"  Total selected: {total}  (from 900)")
    print(f"  Correct: {correct} ({correct/total*100:.1f}%)  |  Wrong: {wrong} ({wrong/total*100:.1f}%)")
    print("=" * 72)

    # --- Error pattern distribution ---
    print("\n--- Error Pattern Distribution ---")
    pattern_counts = Counter(qid_meta[q]["error_pattern"] for q in selected)
    pattern_order = [
        "hop1_high_conf_wrong", "mid_hop_wrong", "exhaustive_wrong",
        "fallback_wrong", "medium_conf_wrong",
        "easy_correct", "hard_correct", "correct_other",
    ]
    for p in pattern_order:
        c = pattern_counts.get(p, 0)
        pct = c / total * 100
        bar = "#" * int(pct)
        print(f"  {p:<25s} {c:>4d}  ({pct:5.1f}%)  {bar}")

    # --- Domain distribution ---
    print("\n--- Domain Distribution ---")
    print(f"  {'Domain':<25s} {'Subset':>6s} {'Full':>6s} {'Sub%':>6s} {'Full%':>6s}")
    domain_full = Counter(qid_meta[q]["domain"] for q in qid_meta)
    domain_sub = Counter(qid_meta[q]["domain"] for q in selected)
    for d in sorted(domain_full.keys()):
        sf = domain_full[d]
        ss = domain_sub.get(d, 0)
        print(f"  {d:<25s} {ss:>6d} {sf:>6d} {ss/total*100:>5.1f}% {sf/900*100:>5.1f}%")

    # --- Sub-category distribution ---
    print("\n--- Sub-Category Distribution ---")
    print(f"  {'Sub-Category':<25s} {'Sel':>4s} {'Full':>4s} {'Sel%':>6s} {'Full%':>6s} {'SelAcc':>7s} {'FullAcc':>8s}")
    subcat_full = Counter(qid_meta[q]["sub_category"] for q in qid_meta)
    subcat_sub = Counter(qid_meta[q]["sub_category"] for q in selected)
    for sc in sorted(subcat_full.keys()):
        sf = subcat_full[sc]
        ss = subcat_sub.get(sc, 0)
        # Accuracy in subset vs full
        sel_correct = sum(1 for q in selected if qid_meta[q]["sub_category"] == sc and qid_meta[q]["correct"])
        full_correct = sum(1 for q in qid_meta if qid_meta[q]["sub_category"] == sc and qid_meta[q]["correct"])
        sel_acc = sel_correct / ss * 100 if ss else 0
        full_acc = full_correct / sf * 100 if sf else 0
        print(f"  {sc:<25s} {ss:>4d} {sf:>4d} {ss/total*100:>5.1f}% {sf/900*100:>5.1f}% {sel_acc:>6.1f}% {full_acc:>7.1f}%")

    # --- Task type distribution ---
    print("\n--- Task Type Distribution ---")
    print(f"  {'Task Type':<25s} {'Sel':>4s} {'Full':>4s} {'Sel%':>6s} {'FullAcc':>8s}")
    task_full = Counter(qid_meta[q]["task_type"] for q in qid_meta)
    task_sub = Counter(qid_meta[q]["task_type"] for q in selected)
    for tt in sorted(task_full.keys(), key=lambda x: -task_full[x]):
        sf = task_full[tt]
        ss = task_sub.get(tt, 0)
        full_correct = sum(1 for q in qid_meta if qid_meta[q]["task_type"] == tt and qid_meta[q]["correct"])
        full_acc = full_correct / sf * 100 if sf else 0
        print(f"  {tt:<25s} {ss:>4d} {sf:>4d} {ss/total*100:>5.1f}% {full_acc:>7.1f}%")

    # --- Confidence distribution ---
    print("\n--- Confidence Distribution ---")
    conf_sub = Counter(qid_meta[q]["confidence"] for q in selected)
    conf_full = Counter(qid_meta[q]["confidence"] for q in qid_meta)
    for c in ["high", "medium", "low"]:
        ss = conf_sub.get(c, 0)
        sf = conf_full.get(c, 0)
        print(f"  {c:<10s}  subset: {ss:>4d} ({ss/total*100:>5.1f}%)  full: {sf:>4d} ({sf/900*100:>5.1f}%)")

    # --- Hop distribution ---
    print("\n--- Hop Count Distribution ---")
    hop_sub = Counter(qid_meta[q]["total_hops"] for q in selected)
    hop_full = Counter(qid_meta[q]["total_hops"] for q in qid_meta)
    for h in sorted(hop_full.keys()):
        ss = hop_sub.get(h, 0)
        sf = hop_full.get(h, 0)
        print(f"  {h} hops:  subset: {ss:>4d} ({ss/total*100:>5.1f}%)  full: {sf:>4d} ({sf/900*100:>5.1f}%)")

    # --- Coverage verification ---
    print("\n--- Coverage Verification ---")
    full_domains = set(qid_meta[q]["domain"] for q in qid_meta)
    sel_domains = set(qid_meta[q]["domain"] for q in selected)
    full_subcats = set(qid_meta[q]["sub_category"] for q in qid_meta)
    sel_subcats = set(qid_meta[q]["sub_category"] for q in selected)
    full_tasks = set(qid_meta[q]["task_type"] for q in qid_meta)
    sel_tasks = set(qid_meta[q]["task_type"] for q in selected)

    print(f"  Domains:        {len(sel_domains)}/{len(full_domains)} covered", end="")
    missing = full_domains - sel_domains
    print(f"  MISSING: {missing}" if missing else "  [ALL COVERED]")

    print(f"  Sub-categories: {len(sel_subcats)}/{len(full_subcats)} covered", end="")
    missing = full_subcats - sel_subcats
    print(f"  MISSING: {missing}" if missing else "  [ALL COVERED]")

    print(f"  Task types:     {len(sel_tasks)}/{len(full_tasks)} covered", end="")
    missing = full_tasks - sel_tasks
    print(f"  MISSING: {missing}" if missing else "  [ALL COVERED]")

    # --- Method distribution ---
    print("\n--- Method Distribution ---")
    method_sub = Counter(qid_meta[q]["method"] for q in selected)
    method_full = Counter(qid_meta[q]["method"] for q in qid_meta)
    for m in sorted(method_full.keys()):
        ss = method_sub.get(m, 0)
        sf = method_full.get(m, 0)
        print(f"  {m:<25s}  subset: {ss:>4d} ({ss/total*100:>5.1f}%)  full: {sf:>4d} ({sf/900*100:>5.1f}%)")

    # --- Diagnostic value summary ---
    print("\n--- Diagnostic Value Summary ---")
    wrong_pct = wrong / total * 100
    print(f"  Wrong-answer ratio: {wrong_pct:.1f}% (target: ~65-75%)")
    unique_patterns = len([p for p in pattern_counts if pattern_counts[p] > 0])
    print(f"  Error patterns represented: {unique_patterns}/{len(pattern_order)}")
    print(f"  Easy canaries (easy_correct): {pattern_counts.get('easy_correct', 0)}")
    print(f"  If subset accuracy matches full accuracy within ~5%, the subset is representative.")
    print()


def write_output(
    output_path: str,
    selected: list[str],
    qid_meta: dict,
    results: dict,
    categories: dict,
):
    """Write selected subset as JSONL with metadata."""
    with open(output_path, "w") as f:
        for qid in selected:
            meta = qid_meta[qid]
            cat = categories.get(qid, {})
            entry = {
                "question_id": qid,
                "video_id": cat.get("video_id", qid.rsplit("-", 1)[0]),
                "correct": meta["correct"],
                "confidence": meta["confidence"],
                "total_hops": meta["total_hops"],
                "method": meta["method"],
                "error_pattern": meta["error_pattern"],
                "question_type": meta["question_type"],
                "used_visual": meta["used_visual"],
                "domain": meta["domain"],
                "sub_category": meta["sub_category"],
                "task_type": meta["task_type"],
                "question": cat.get("question", ""),
                "options": cat.get("options", []),
                "answer": cat.get("answer", ""),
                "pred": results[qid].get("pred", ""),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(selected)} entries to {output_path}")

    # Also write a simple list of question IDs (useful for filtering)
    ids_path = output_path.replace(".jsonl", "_ids.txt")
    with open(ids_path, "w") as f:
        for qid in selected:
            f.write(qid + "\n")
    print(f"Wrote question ID list to {ids_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select an optimal Video-MME subset for ablation experiments."
    )
    parser.add_argument(
        "--results-dir",
        default="/lustre/youngbeom/find_solver_please/output/video_mme_full_visual/v1/by_qid/",
        help="Directory containing per-question result JSONs",
    )
    parser.add_argument(
        "--category-dir",
        default="/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/long/",
        help="Directory containing per-question category/metadata JSONs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path (default: <results-dir>/../ablation_subset.jsonl)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=130,
        help="Target number of questions in the subset (default: 130)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.output is None:
        parent = str(Path(args.results_dir).parent)
        args.output = os.path.join(parent, "ablation_subset.jsonl")

    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"  Loaded {len(results)} results")

    print(f"Loading categories from: {args.category_dir}")
    categories = load_categories(args.category_dir)
    print(f"  Loaded {len(categories)} category records")
    print()

    selected, qid_meta = build_selection(
        results, categories,
        target_size=args.target_size,
        seed=args.seed,
    )

    print_statistics(selected, qid_meta, categories)

    write_output(args.output, selected, qid_meta, results, categories)


if __name__ == "__main__":
    main()
