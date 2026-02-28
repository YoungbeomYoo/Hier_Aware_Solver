#!/usr/bin/env python3
"""
결과 확인 + 베이스라인 비교 스크립트

Usage:
    # 전체 실험 현황
    python check_results.py

    # 특정 실험 상세 (task_type/domain별)
    python check_results.py output/video_mme_full_semantic/v2

    # 베이스라인과 비교
    python check_results.py output/video_mme_full_semantic/v2 --baseline output/videolucy-videomme-long/merged_summary.json

    # 두 실험 문제별 비교
    python check_results.py --compare output/video_mme_subset_tree_search_visual3/v1 output/video_mme_subset_tree_search_semantic/v2

    # 비디오별 / 틀린 문제
    python check_results.py output/video_mme_full_semantic/v2 --by_video
    python check_results.py output/video_mme_full_semantic/v2 --wrong
"""

import os
import json
import argparse
from glob import glob
from collections import defaultdict


# ── Question metadata (task_type, domain) ────────────────────────

def load_question_metadata(q_dirs=None):
    """원본 question JSON에서 task_type, domain 등 메타데이터 로딩."""
    if q_dirs is None:
        q_dirs = [
            "/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/long",
            "/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/medium",
            "/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/split_per_duration/short",
        ]

    metadata = {}
    for q_dir in q_dirs:
        if not os.path.isdir(q_dir):
            continue
        for qf in glob(os.path.join(q_dir, "**", "*.json"), recursive=True):
            try:
                with open(qf) as f:
                    data = json.load(f)
                qlist = data if isinstance(data, list) else [data]
                for q in qlist:
                    qid = q.get("question_id", os.path.basename(qf).replace(".json", ""))
                    metadata[str(qid)] = {
                        "task_type": q.get("task_type", ""),
                        "domain": q.get("domain", ""),
                        "duration": q.get("duration", ""),
                        "sub_category": q.get("sub_category", ""),
                    }
            except Exception:
                pass
    return metadata


# ── Result loading ───────────────────────────────────────────────

def load_results(by_qid_dir):
    """by_qid/ 폴더에서 모든 결과 로딩. 없으면 상위의 results.json도 시도."""
    results = []
    if os.path.isdir(by_qid_dir):
        for fname in sorted(os.listdir(by_qid_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(by_qid_dir, fname)) as f:
                    r = json.load(f)
                r.setdefault("question_id", fname.replace(".json", ""))
                results.append(r)
            except Exception:
                pass
        if results:
            return results

    # Fallback: results.json (flat list) in parent directory
    parent = os.path.dirname(by_qid_dir.rstrip("/"))
    results_json = os.path.join(parent, "results.json")
    if os.path.isfile(results_json):
        try:
            with open(results_json) as f:
                data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    # Normalize keys: gt_answer→answer, pred_answer→pred
                    if "gt_answer" in r and "answer" not in r:
                        r["answer"] = r["gt_answer"]
                    if "pred_answer" in r and "pred" not in r:
                        r["pred"] = r["pred_answer"]
                    results.append(r)
        except Exception:
            pass
    return results


def enrich_results(results, metadata):
    """결과에 task_type, domain 등 메타데이터 추가."""
    for r in results:
        qid = str(r.get("question_id", ""))
        if qid in metadata:
            for k, v in metadata[qid].items():
                if k not in r or not r[k]:
                    r[k] = v
    return results


# ── Summary printing ─────────────────────────────────────────────

def calc_accuracy(results):
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    return correct, total, correct / total if total > 0 else 0


def print_summary(results, label=""):
    total = len(results)
    if total == 0:
        print(f"  {label}: (no results)")
        return {}

    correct, total, acc = calc_accuracy(results)
    print(f"  {label}")
    print(f"    Accuracy: {correct}/{total} = {acc:.4f} ({acc*100:.1f}%)")

    # visual 사용
    vis = sum(1 for r in results if r.get("used_visual"))
    if vis > 0:
        print(f"    Visual used: {vis}/{total} ({vis/total*100:.1f}%)")

    # 평균 hops / time
    hops = [r["total_hops"] for r in results if r.get("total_hops")]
    times = [r["time"] for r in results if r.get("time")]
    if hops:
        print(f"    Avg hops: {sum(hops)/len(hops):.2f}")
    if times:
        print(f"    Avg time: {sum(times)/len(times):.1f}s, Total: {sum(times)/60:.1f}min")

    return {"correct": correct, "total": total, "accuracy": round(acc * 100, 2)}


def print_breakdown(results, key, label):
    """특정 키 기준으로 breakdown."""
    groups = defaultdict(list)
    for r in results:
        val = r.get(key, "unknown")
        if isinstance(val, list):
            for v in val:
                groups[v].append(r)
        else:
            groups[val].append(r)

    if len(groups) <= 1 and "unknown" in groups:
        return {}

    print(f"\n  {label}:")
    print(f"  {'Category':<30} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print(f"  {'-'*55}")

    breakdown = {}
    for cat in sorted(groups.keys()):
        rs = groups[cat]
        c, t, a = calc_accuracy(rs)
        print(f"  {cat:<30} {c:>8} {t:>6} {a:>7.1%}")
        breakdown[cat] = {"correct": c, "total": t, "accuracy": round(a * 100, 2)}

    return breakdown


# ── Baseline comparison ──────────────────────────────────────────

def compare_with_baseline(results, baseline_path):
    """베이스라인 summary.json과 비교."""
    with open(baseline_path) as f:
        baseline = json.load(f)

    bl_acc = baseline.get("accuracy", 0)
    bl_total = baseline.get("evaluated", baseline.get("total", 0))
    bl_correct = baseline.get("correct", 0)

    my_correct, my_total, my_acc = calc_accuracy(results)

    print(f"\n  {'='*65}")
    print(f"  Baseline Comparison")
    print(f"  {'='*65}")
    print(f"  {'':30} {'Baseline':>12} {'Ours':>12} {'Delta':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Overall':<30} {bl_acc:>11.2f}% {my_acc*100:>11.2f}% {(my_acc*100-bl_acc):>+9.2f}%")

    # task_type 비교
    bl_types = baseline.get("per_task_type", {})
    if bl_types:
        my_types = defaultdict(list)
        for r in results:
            tt = r.get("task_type", "unknown")
            my_types[tt].append(r)

        print(f"\n  {'Task Type':<30} {'Baseline':>12} {'Ours':>12} {'Delta':>10}")
        print(f"  {'-'*65}")

        all_types = sorted(set(list(bl_types.keys()) + list(my_types.keys())))
        for tt in all_types:
            if tt == "unknown":
                continue
            bl_info = bl_types.get(tt, {})
            bl_a = bl_info.get("accuracy", 0)
            bl_t = bl_info.get("total", 0)

            my_rs = my_types.get(tt, [])
            my_c, my_t, my_a = calc_accuracy(my_rs)

            if my_t == 0 and bl_t == 0:
                continue

            delta = (my_a * 100 - bl_a) if my_t > 0 else float("nan")
            bl_str = f"{bl_a:.1f}% ({bl_t})" if bl_t > 0 else "  -  "
            my_str = f"{my_a*100:.1f}% ({my_t})" if my_t > 0 else "  -  "
            d_str = f"{delta:+.1f}%" if my_t > 0 else "  -  "
            print(f"  {tt:<30} {bl_str:>12} {my_str:>12} {d_str:>10}")

    # domain 비교
    bl_domains = baseline.get("per_domain", {})
    if bl_domains:
        my_domains = defaultdict(list)
        for r in results:
            d = r.get("domain", "unknown")
            my_domains[d].append(r)

        print(f"\n  {'Domain':<30} {'Baseline':>12} {'Ours':>12} {'Delta':>10}")
        print(f"  {'-'*65}")

        all_domains = sorted(set(list(bl_domains.keys()) + list(my_domains.keys())))
        for dom in all_domains:
            if dom == "unknown":
                continue
            bl_info = bl_domains.get(dom, {})
            bl_a = bl_info.get("accuracy", 0)
            bl_t = bl_info.get("total", 0)

            my_rs = my_domains.get(dom, [])
            my_c, my_t, my_a = calc_accuracy(my_rs)

            if my_t == 0 and bl_t == 0:
                continue

            delta = (my_a * 100 - bl_a) if my_t > 0 else float("nan")
            bl_str = f"{bl_a:.1f}% ({bl_t})" if bl_t > 0 else "  -  "
            my_str = f"{my_a*100:.1f}% ({my_t})" if my_t > 0 else "  -  "
            d_str = f"{delta:+.1f}%" if my_t > 0 else "  -  "
            print(f"  {dom:<30} {bl_str:>12} {my_str:>12} {d_str:>10}")

    print(f"  {'='*65}")
    if my_total < bl_total:
        print(f"  Note: 현재 {my_total}/{bl_total} 문제만 완료 ({my_total/bl_total*100:.0f}%)")


# ── Two-experiment comparison ────────────────────────────────────

def compare_experiments(dir_a, dir_b, metadata=None):
    """두 실험 결과 문제별 비교."""
    results_a = {r["question_id"]: r for r in load_results(os.path.join(dir_a, "by_qid"))}
    results_b = {r["question_id"]: r for r in load_results(os.path.join(dir_b, "by_qid"))}

    common = set(results_a.keys()) & set(results_b.keys())
    if not common:
        print("  No common questions found!")
        return

    label_a = "/".join(dir_a.rstrip("/").split("/")[-2:])
    label_b = "/".join(dir_b.rstrip("/").split("/")[-2:])

    a_correct = sum(1 for q in common if results_a[q].get("correct"))
    b_correct = sum(1 for q in common if results_b[q].get("correct"))

    print(f"\n  Comparing {len(common)} common questions:")
    print(f"    A: {label_a}  →  {a_correct}/{len(common)} = {a_correct/len(common):.4f}")
    print(f"    B: {label_b}  →  {b_correct}/{len(common)} = {b_correct/len(common):.4f}")

    a_only = [q for q in common if results_a[q].get("correct") and not results_b[q].get("correct")]
    b_only = [q for q in common if results_b[q].get("correct") and not results_a[q].get("correct")]
    both_ok = [q for q in common if results_a[q].get("correct") and results_b[q].get("correct")]
    both_bad = [q for q in common if not results_a[q].get("correct") and not results_b[q].get("correct")]

    print(f"\n    Both correct: {len(both_ok)}")
    print(f"    Both wrong:   {len(both_bad)}")
    print(f"    A only:       {len(a_only)}")
    print(f"    B only:       {len(b_only)}")

    def _show(qids, tag):
        if not qids:
            return
        print(f"\n    {tag} (상위 10개):")
        for q in sorted(qids)[:10]:
            ra, rb = results_a.get(q, {}), results_b.get(q, {})
            vid = ra.get("video_id", rb.get("video_id", "?"))
            tt = metadata.get(q, {}).get("task_type", "") if metadata else ""
            tt_str = f"  [{tt}]" if tt else ""
            print(f"      {q:<12} A={ra.get('pred','?')} B={rb.get('pred','?')} GT={ra.get('answer', rb.get('answer','?'))}{tt_str}")

    _show(a_only, "A만 맞춘 문제")
    _show(b_only, "B만 맞춘 문제")


# ── Misc views ───────────────────────────────────────────────────

def print_wrong(results):
    wrong = [r for r in results if not r.get("correct")]
    print(f"\n  Wrong answers: {len(wrong)}/{len(results)}")
    print(f"  {'QID':<12} {'Video':<20} {'Pred':>5} {'GT':>5} {'TaskType':<25} {'Hops':>5}")
    print(f"  {'-'*75}")
    for r in wrong:
        print(f"  {r.get('question_id','?'):<12} "
              f"{r.get('video_id','?'):<20} "
              f"{r.get('pred','?'):>5} "
              f"{r.get('answer','?'):>5} "
              f"{r.get('task_type','?'):<25} "
              f"{r.get('total_hops','?'):>5}")


def print_by_video(results):
    by_vid = defaultdict(list)
    for r in results:
        by_vid[r.get("video_id", "unknown")].append(r)

    print(f"\n  {'Video':<25} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print(f"  {'-'*50}")
    for vid in sorted(by_vid.keys()):
        rs = by_vid[vid]
        c, t, a = calc_accuracy(rs)
        marker = " X" if c == 0 else ""
        print(f"  {vid:<25} {c:>8} {t:>6} {a:>7.1%}{marker}")


# ── Scan all experiments ─────────────────────────────────────────

def scan_experiments(base_dir="output"):
    experiments = []
    if not os.path.isdir(base_dir):
        return experiments
    for exp_name in sorted(os.listdir(base_dir)):
        exp_dir = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        # Check for versioned subdirs (v1, v2, ...)
        has_versions = False
        for v in sorted(os.listdir(exp_dir)):
            by_qid = os.path.join(exp_dir, v, "by_qid")
            if os.path.isdir(by_qid):
                n = len([f for f in os.listdir(by_qid) if f.endswith(".json")])
                if n > 0:
                    experiments.append((f"{exp_name}/{v}", os.path.join(exp_dir, v), n))
                    has_versions = True
        # Flat results.json (e.g., videolucy baseline)
        if not has_versions:
            results_json = os.path.join(exp_dir, "results.json")
            if os.path.isfile(results_json):
                try:
                    with open(results_json) as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        experiments.append((exp_name, exp_dir, len(data)))
                except Exception:
                    pass
    return experiments


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="실험 결과 확인 + 베이스라인 비교")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="실험 디렉토리 (e.g. output/video_mme_full_semantic/v2)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="베이스라인 summary JSON")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="두 실험 디렉토리 비교")
    parser.add_argument("--by_video", action="store_true")
    parser.add_argument("--by_type", action="store_true")
    parser.add_argument("--wrong", action="store_true")
    args = parser.parse_args()

    metadata = load_question_metadata()

    if args.compare:
        compare_experiments(args.compare[0], args.compare[1], metadata)
        return

    if args.output_dir:
        results = load_results(os.path.join(args.output_dir, "by_qid"))
        if not results:
            print(f"No results in {args.output_dir} (checked by_qid/ and results.json)")
            return

        results = enrich_results(results, metadata)

        print(f"\n{'='*65}")
        print_summary(results, label=args.output_dir)
        print_breakdown(results, "task_type", "Per Task Type")
        print_breakdown(results, "domain", "Per Domain")

        if args.baseline:
            compare_with_baseline(results, args.baseline)

        if args.by_video:
            print_by_video(results)
        if args.wrong:
            print_wrong(results)
        print()
        return

    # 전체 현황
    print(f"\n{'='*65}")
    print(f"  All Experiments")
    print(f"{'='*65}")

    experiments = scan_experiments("output")
    if not experiments:
        print("  No results found in output/")
        return

    for label, exp_dir, n_files in experiments:
        results = load_results(os.path.join(exp_dir, "by_qid"))
        print()
        print_summary(results, label=label)

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
