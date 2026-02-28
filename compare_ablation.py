#!/usr/bin/env python3
"""
Ablation Experiment Comparison — 130-question subset 기준
==========================================================

6개 ablation 실험 + VideoLucy baseline 비교표.

Usage:
    python compare_ablation.py
    python compare_ablation.py --detail          # task_type/domain별 상세
    python compare_ablation.py --diff baseline_exact videolucy_prompt  # 2개 비교

Output:
    Overall accuracy, per-axis effect, agreement patterns
"""

import os
import json
import argparse
from collections import defaultdict, Counter
from glob import glob


# ============================================================
# Configuration
# ============================================================

SUBSET_IDS_PATH = "output/video_mme_full_visual/v1/ablation_subset_ids.txt"

# Experiment definitions: name → (config description, axes)
EXPERIMENTS = {
    # --- Round 1: C × D × F 기본 ablation ---
    "baseline_exact":       ("C0+D1+F1", {"C": "strict",   "D": "1-stage",   "F": "compact"}),
    "videolucy_prompt":     ("C2+D1+F1", {"C": "videolucy", "D": "1-stage",   "F": "compact"}),
    "strict_accumulate":    ("C0+D1+F0", {"C": "strict",   "D": "1-stage",   "F": "accumulate"}),
    "videolucy_accumulate": ("C2+D1+F0", {"C": "videolucy", "D": "1-stage",   "F": "accumulate"}),
    "text_only":            ("C2+D0+F0", {"C": "videolucy", "D": "text-only", "F": "accumulate"}),
    "twostage":             ("C2+D2+F0", {"C": "videolucy", "D": "two-stage", "F": "accumulate"}),
    # --- Round 2: text_only 심화 + no_caption ---
    "text_only_compact":    ("C2+D0+F1", {"C": "videolucy", "D": "text-only", "F": "compact"}),
    "text_only_strict":     ("C0+D0+F0", {"C": "strict",   "D": "text-only", "F": "accumulate"}),
    "no_caption":           ("C2+D1nc+F0",{"C": "videolucy","D": "1-stage-nc","F": "accumulate"}),
    # --- Round 3: A축 + B축 ---
    "no_query":             ("A0+D0+F0", {"A": "rule-only", "B": "exact",     "D": "text-only", "F": "accumulate"}),
    "llm_select":           ("A1+B3+D0", {"A": "LLM",      "B": "LLM-select","D": "text-only", "F": "accumulate"}),
    # --- Round 4: D3 always visual ---
    "always_visual":        ("C2+D3+F1", {"C": "videolucy", "D": "always-vis","F": "compact"}),
    "always_visual_acc":    ("C2+D3+F0", {"C": "videolucy", "D": "always-vis","F": "accumulate"}),
    # --- Round 5: C×D full grid (F0 고정) ---
    "strict_twostage":          ("C0+D2+F0", {"C": "strict",    "D": "two-stage", "F": "accumulate"}),
    "strict_always_visual":     ("C0+D3+F0", {"C": "strict",    "D": "always-vis","F": "accumulate"}),
    "relaxed_text_only":        ("C1+D0+F0", {"C": "relaxed",   "D": "text-only", "F": "accumulate"}),
    "relaxed_visual":           ("C1+D1+F0", {"C": "relaxed",   "D": "1-stage",   "F": "accumulate"}),
    "relaxed_twostage":         ("C1+D2+F0", {"C": "relaxed",   "D": "two-stage", "F": "accumulate"}),
    "relaxed_always_visual":    ("C1+D3+F0", {"C": "relaxed",   "D": "always-vis","F": "accumulate"}),
    "answerjudge_text_only":    ("C3+D0+F0", {"C": "VL+AJ",    "D": "text-only", "F": "accumulate"}),
    "answerjudge_visual":       ("C3+D1+F0", {"C": "VL+AJ",    "D": "1-stage",   "F": "accumulate"}),
    "answerjudge_twostage":     ("C3+D2+F0", {"C": "VL+AJ",    "D": "two-stage", "F": "accumulate"}),
    "answerjudge_always_visual":("C3+D3+F0", {"C": "VL+AJ",    "D": "always-vis","F": "accumulate"}),
    # --- Round 7: 미검증 축 검증 ---
    "c3_compact":               ("C3+D0+F1", {"C": "VL+AJ",    "D": "text-only", "F": "compact"}),
    "no_filter":                ("B0+C3+D0", {"B": "no-filter", "C": "VL+AJ",    "D": "text-only", "F": "accumulate"}),
    "no_filter_compact":        ("B0+C3+F1", {"B": "no-filter", "C": "VL+AJ",    "D": "text-only", "F": "compact"}),
    "visual_relaxed_rejudge":   ("C2+D1+Cv", {"C": "videolucy", "D": "1-stage",  "F": "accumulate", "Cv": "VJ-relaxed"}),
    # --- Round 8: Phase 0 (Coarse-First) ---
    "coarse_first":             ("G1+C3+F1", {"C": "VL+AJ",    "D": "text-only", "F": "compact",    "G": "coarse-first"}),
    # --- Round 9: Visual Context Enrichment ---
    "visual_enrich":            ("D5+C3+F1+G1", {"C": "VL+AJ", "D": "vis-enrich", "F": "compact",   "G": "coarse-first"}),
}

VIDEOLUCY_RESULTS = "output/videolucy-videomme-long/results.json"


# ============================================================
# Data loading
# ============================================================

def load_subset_ids():
    with open(SUBSET_IDS_PATH) as f:
        return set(line.strip() for line in f if line.strip())


def load_experiment(name):
    """Load results for an ablation experiment."""
    by_qid_dir = f"output/ablation_{name}/v1/by_qid"
    results = {}
    if os.path.isdir(by_qid_dir):
        for fname in os.listdir(by_qid_dir):
            if fname.endswith(".json"):
                qid = fname.replace(".json", "")
                with open(os.path.join(by_qid_dir, fname)) as f:
                    results[qid] = json.load(f)
    return results


def load_videolucy(subset_ids):
    """Load VideoLucy results filtered to subset."""
    if not os.path.isfile(VIDEOLUCY_RESULTS):
        return {}
    with open(VIDEOLUCY_RESULTS) as f:
        data = json.load(f)
    return {
        r["question_id"]: r for r in data
        if r.get("question_id") in subset_ids
    }


def load_metadata():
    """Load question metadata for task_type/domain."""
    meta_dir = "/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/long"
    metadata = {}
    for qf in glob(os.path.join(meta_dir, "**", "*.json"), recursive=True):
        try:
            with open(qf) as f:
                data = json.load(f)
            qlist = data if isinstance(data, list) else [data]
            for q in qlist:
                qid = str(q.get("question_id", ""))
                metadata[qid] = {
                    "task_type": q.get("task_type", ""),
                    "domain": q.get("domain", ""),
                    "sub_category": q.get("sub_category", ""),
                }
        except Exception:
            pass
    return metadata


# ============================================================
# Analysis
# ============================================================

def calc_acc(results, subset_ids):
    common = set(results.keys()) & subset_ids
    if not common:
        return 0, 0, 0.0
    correct = sum(1 for q in common if results[q].get("correct"))
    return correct, len(common), correct / len(common)


def print_overview(all_results, vl_results, subset_ids):
    """Main comparison table."""
    print("=" * 80)
    print("  ABLATION RESULTS — 130-question subset")
    print("=" * 80)

    # VideoLucy baseline
    vl_c, vl_t, vl_a = calc_acc(vl_results, subset_ids)

    print(f"\n  {'Experiment':<25s} {'Axes':<12s} {'Correct':>8s} {'Total':>6s} "
          f"{'Acc':>7s} {'vs BL':>7s} {'vs VL':>7s}")
    print(f"  {'-'*78}")

    # VideoLucy line
    print(f"  {'[VideoLucy baseline]':<25s} {'---':<12s} {vl_c:>8d} {vl_t:>6d} "
          f"{vl_a*100:>6.1f}% {'---':>7s} {'---':>7s}")
    print(f"  {'-'*78}")

    baseline_acc = None
    exp_results = {}

    for name, (axes_str, axes) in EXPERIMENTS.items():
        results = all_results.get(name, {})
        c, t, a = calc_acc(results, subset_ids)

        if t == 0:
            print(f"  {name:<25s} {axes_str:<12s} {'(not run)':>8s}")
            continue

        if baseline_acc is None:
            baseline_acc = a

        vs_bl = f"{(a - baseline_acc)*100:>+6.1f}%" if baseline_acc is not None else "---"
        vs_vl = f"{(a - vl_a)*100:>+6.1f}%" if vl_t > 0 else "---"

        print(f"  {name:<25s} {axes_str:<12s} {c:>8d} {t:>6d} "
              f"{a*100:>6.1f}% {vs_bl:>7s} {vs_vl:>7s}")
        exp_results[name] = (c, t, a)

    print()

    # Axis effect analysis
    if len(exp_results) >= 4:
        print_axis_effects(exp_results)


def print_axis_effects(exp_results):
    """Measure individual axis effects."""
    print("  " + "=" * 60)
    print("  AXIS EFFECTS (individual contribution)")
    print("  " + "=" * 60)

    def delta(a_name, b_name):
        if a_name not in exp_results or b_name not in exp_results:
            return None
        return (exp_results[b_name][2] - exp_results[a_name][2]) * 100

    # C축: strict → videolucy
    d = delta("baseline_exact", "videolucy_prompt")
    if d is not None:
        print(f"  C축 (Judge prompt):   baseline→videolucy = {d:>+.1f}%p")

    # F축: compact → accumulate
    d = delta("baseline_exact", "strict_accumulate")
    if d is not None:
        print(f"  F축 (History):        compact→accumulate = {d:>+.1f}%p")

    # C+F 시너지
    d_c = delta("baseline_exact", "videolucy_prompt")
    d_f = delta("baseline_exact", "strict_accumulate")
    d_cf = delta("baseline_exact", "videolucy_accumulate")
    if d_c is not None and d_f is not None and d_cf is not None:
        synergy = d_cf - (d_c + d_f)
        print(f"  C+F 시너지:           expected {d_c+d_f:>+.1f}, actual {d_cf:>+.1f} → synergy {synergy:>+.1f}%p")

    # D축 스펙트럼: text_only → 1-stage → two-stage
    d_t2v = delta("text_only", "videolucy_accumulate")
    d_v2ts = delta("videolucy_accumulate", "twostage")
    if d_t2v is not None:
        print(f"  D축 (Visual):         text→1-stage = {d_t2v:>+.1f}%p")
    if d_v2ts is not None:
        print(f"                        1-stage→two-stage = {d_v2ts:>+.1f}%p")

    print()


def print_detail(all_results, vl_results, subset_ids, metadata):
    """Per task_type and domain breakdown."""
    print("\n" + "=" * 100)
    print("  PER TASK_TYPE BREAKDOWN")
    print("=" * 100)

    # Collect task_types
    task_types = sorted(set(
        metadata.get(q, {}).get("task_type", "?")
        for q in subset_ids
    ) - {"?"})

    # Header
    exp_names = [n for n in EXPERIMENTS if n in all_results and calc_acc(all_results[n], subset_ids)[1] > 0]
    header = f"  {'Task Type':<22s} {'VL':>5s}"
    for name in exp_names:
        short = name[:8]
        header += f" {short:>8s}"
    print(header)
    print(f"  {'-'*(22 + 6 + 9*len(exp_names))}")

    for tt in task_types:
        tt_ids = {q for q in subset_ids if metadata.get(q, {}).get("task_type") == tt}
        if not tt_ids:
            continue

        vl_c, vl_t, vl_a = calc_acc(vl_results, tt_ids)
        line = f"  {tt:<22s} {vl_a*100:>4.0f}%"

        for name in exp_names:
            c, t, a = calc_acc(all_results[name], tt_ids)
            if t == 0:
                line += f" {'---':>8s}"
            else:
                line += f" {a*100:>7.0f}%"
        print(line)

    # Domain
    print("\n" + "=" * 100)
    print("  PER DOMAIN BREAKDOWN")
    print("=" * 100)

    domains = sorted(set(
        metadata.get(q, {}).get("domain", "?")
        for q in subset_ids
    ) - {"?"})

    header = f"  {'Domain':<22s} {'VL':>5s}"
    for name in exp_names:
        short = name[:8]
        header += f" {short:>8s}"
    print(header)
    print(f"  {'-'*(22 + 6 + 9*len(exp_names))}")

    for dom in domains:
        dom_ids = {q for q in subset_ids if metadata.get(q, {}).get("domain") == dom}
        if not dom_ids:
            continue

        vl_c, vl_t, vl_a = calc_acc(vl_results, dom_ids)
        line = f"  {dom:<22s} {vl_a*100:>4.0f}%"

        for name in exp_names:
            c, t, a = calc_acc(all_results[name], dom_ids)
            if t == 0:
                line += f" {'---':>8s}"
            else:
                line += f" {a*100:>7.0f}%"
        print(line)

    print()


def print_diff(all_results, vl_results, subset_ids, metadata, name_a, name_b):
    """두 실험 간 문제별 차이 분석."""
    res_a = all_results.get(name_a, {})
    res_b = all_results.get(name_b, {})

    common = set(res_a.keys()) & set(res_b.keys()) & subset_ids
    if not common:
        print(f"No common results between {name_a} and {name_b}")
        return

    a_only = [q for q in common if res_a[q].get("correct") and not res_b[q].get("correct")]
    b_only = [q for q in common if res_b[q].get("correct") and not res_a[q].get("correct")]
    both_ok = [q for q in common if res_a[q].get("correct") and res_b[q].get("correct")]
    both_bad = [q for q in common if not res_a[q].get("correct") and not res_b[q].get("correct")]

    a_c = sum(1 for q in common if res_a[q].get("correct"))
    b_c = sum(1 for q in common if res_b[q].get("correct"))

    print(f"\n  Diff: {name_a} vs {name_b} ({len(common)} common questions)")
    print(f"  {name_a}: {a_c}/{len(common)} = {a_c/len(common)*100:.1f}%")
    print(f"  {name_b}: {b_c}/{len(common)} = {b_c/len(common)*100:.1f}%")
    print(f"\n  Both correct: {len(both_ok)}")
    print(f"  Both wrong:   {len(both_bad)}")
    print(f"  {name_a} only: {len(a_only)}")
    print(f"  {name_b} only: {len(b_only)}")

    def show_qids(qids, tag):
        if not qids:
            return
        print(f"\n  {tag}:")
        for q in sorted(qids)[:15]:
            ra, rb = res_a.get(q, {}), res_b.get(q, {})
            tt = metadata.get(q, {}).get("task_type", "")
            vl_ok = "VL✓" if vl_results.get(q, {}).get("correct") else "VL✗"
            print(f"    {q:<12s} A={ra.get('pred','?')} B={rb.get('pred','?')} "
                  f"GT={ra.get('answer','?')} [{tt}] {vl_ok}")

    show_qids(a_only, f"{name_a}만 맞춤")
    show_qids(b_only, f"{name_b}만 맞춤")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation experiment comparison")
    parser.add_argument("--detail", action="store_true", help="Show per-task_type/domain breakdown")
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"), help="Compare two experiments")
    args = parser.parse_args()

    subset_ids = load_subset_ids()
    metadata = load_metadata()

    # Load all experiments
    all_results = {}
    for name in EXPERIMENTS:
        results = load_experiment(name)
        if results:
            all_results[name] = results

    # Load VideoLucy
    vl_results = load_videolucy(subset_ids)

    # Print
    print_overview(all_results, vl_results, subset_ids)

    if args.detail:
        print_detail(all_results, vl_results, subset_ids, metadata)

    if args.diff:
        print_diff(all_results, vl_results, subset_ids, metadata,
                   args.diff[0], args.diff[1])


if __name__ == "__main__":
    main()
