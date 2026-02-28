#!/usr/bin/env python3
"""
두 실험에서 각 문제를 풀 때 어떤 시간 구간을 참조했는지 비교

Usage:
    # 두 실험 비교 (visual vs semantic)
    python compare_coverage.py \
        output/video_mme_subset_tree_search_visual3/v1 \
        output/video_mme_subset_tree_search_semantic/v2

    # 특정 문제만
    python compare_coverage.py DIR_A DIR_B --qid 604-1

    # 겹침 분석만
    python compare_coverage.py DIR_A DIR_B --overlap_only
"""

import os
import json
import argparse
from collections import defaultdict


def load_results(by_qid_dir):
    results = {}
    if not os.path.isdir(by_qid_dir):
        return results
    for fname in sorted(os.listdir(by_qid_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(by_qid_dir, fname)) as f:
                r = json.load(f)
            qid = r.get("question_id", fname.replace(".json", ""))
            results[qid] = r
        except Exception:
            pass
    return results


def extract_time_ranges(result):
    """결과에서 모든 hop의 target_leaves 시간 구간 추출."""
    ranges = []
    for hc in result.get("hop_contexts", []):
        hop_num = hc.get("hop", "?")
        for leaf in hc.get("target_leaves", []):
            s = leaf.get("start_time")
            e = leaf.get("end_time")
            if s is not None and e is not None:
                ranges.append({
                    "start": float(s),
                    "end": float(e),
                    "hop": hop_num,
                    "caption": (leaf.get("caption", "") or "")[:80],
                })
    return ranges


def extract_semantic_info(result):
    """semantic_match 정보 추출."""
    sm = result.get("semantic_match", {})
    if not sm:
        return None
    return {
        "selected_l1": sm.get("selected_l1", []),
        "top_scores": sm.get("top_scores", [])[:5],
    }


def ranges_to_set(ranges, resolution=30):
    """시간 구간을 resolution초 단위 셀 집합으로 변환."""
    cells = set()
    for r in ranges:
        s = int(r["start"] / resolution)
        e = int(r["end"] / resolution)
        for c in range(s, e + 1):
            cells.add(c)
    return cells


def format_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def print_question_comparison(qid, ra, rb, label_a="A", label_b="B"):
    """한 문제에 대해 두 실험의 탐색 구간 비교."""
    ranges_a = extract_time_ranges(ra)
    ranges_b = extract_time_ranges(rb)

    vid = ra.get("video_id", rb.get("video_id", "?"))

    print(f"\n  Question: {qid}  (video: {vid})")
    print(f"  {label_a}: pred={ra.get('pred','?')} {'O' if ra.get('correct') else 'X'}  "
          f"hops={ra.get('total_hops','?')}  conf={ra.get('confidence','?')}")
    print(f"  {label_b}: pred={rb.get('pred','?')} {'O' if rb.get('correct') else 'X'}  "
          f"hops={rb.get('total_hops','?')}  conf={rb.get('confidence','?')}")
    print(f"  GT: {ra.get('answer', rb.get('answer', '?'))}")

    # 시간 구간 출력
    print(f"\n  {label_a} explored ({len(ranges_a)} segments):")
    for r in sorted(ranges_a, key=lambda x: x["start"]):
        print(f"    hop{r['hop']}: {format_time(r['start'])}-{format_time(r['end'])}  {r['caption']}")

    print(f"\n  {label_b} explored ({len(ranges_b)} segments):")
    for r in sorted(ranges_b, key=lambda x: x["start"]):
        print(f"    hop{r['hop']}: {format_time(r['start'])}-{format_time(r['end'])}  {r['caption']}")

    # 겹침 분석
    cells_a = ranges_to_set(ranges_a)
    cells_b = ranges_to_set(ranges_b)

    if cells_a or cells_b:
        overlap = cells_a & cells_b
        only_a = cells_a - cells_b
        only_b = cells_b - cells_a
        union = cells_a | cells_b

        iou = len(overlap) / len(union) if union else 0

        print(f"\n  Coverage overlap:")
        print(f"    {label_a} only: {len(only_a)} cells ({len(only_a)*30}s)")
        print(f"    {label_b} only: {len(only_b)} cells ({len(only_b)*30}s)")
        print(f"    Shared:  {len(overlap)} cells ({len(overlap)*30}s)")
        print(f"    IoU:     {iou:.2f}")

    # semantic match 정보 (B가 semantic인 경우)
    sm = extract_semantic_info(rb)
    if sm:
        print(f"\n  Semantic match (top-3 L1 nodes):")
        for sc in sm.get("top_scores", [])[:3]:
            idx = sc.get("idx", "?")
            score = sc.get("score", 0)
            matches = sc.get("top_matches", [])[:2]
            match_str = ", ".join(f"'{m.get('q_element','')}' -> '{m.get('matched_to','')}' ({m.get('similarity',0):.3f})"
                                  for m in matches)
            print(f"    L1[{idx}] score={score:.2f}: {match_str}")


def print_aggregate_overlap(results_a, results_b, label_a="A", label_b="B"):
    """전체 문제에 대한 탐색 구간 겹침 통계."""
    common = set(results_a.keys()) & set(results_b.keys())
    if not common:
        print("  No common questions!")
        return

    ious = []
    a_only_total = 0
    b_only_total = 0
    shared_total = 0

    diff_correct = {"a_only": [], "b_only": [], "both_correct": [], "both_wrong": []}

    for qid in sorted(common):
        ra, rb = results_a[qid], results_b[qid]
        cells_a = ranges_to_set(extract_time_ranges(ra))
        cells_b = ranges_to_set(extract_time_ranges(rb))
        union = cells_a | cells_b
        overlap = cells_a & cells_b

        iou = len(overlap) / len(union) if union else 1.0
        ious.append(iou)
        a_only_total += len(cells_a - cells_b)
        b_only_total += len(cells_b - cells_a)
        shared_total += len(overlap)

        ac = ra.get("correct", False)
        bc = rb.get("correct", False)
        if ac and not bc:
            diff_correct["a_only"].append((qid, iou))
        elif bc and not ac:
            diff_correct["b_only"].append((qid, iou))
        elif ac and bc:
            diff_correct["both_correct"].append((qid, iou))
        else:
            diff_correct["both_wrong"].append((qid, iou))

    n = len(common)
    ca = sum(1 for q in common if results_a[q].get("correct"))
    cb = sum(1 for q in common if results_b[q].get("correct"))

    print(f"\n{'='*65}")
    print(f"  Coverage Comparison: {n} common questions")
    print(f"{'='*65}")
    print(f"  {label_a}: {ca}/{n} correct ({ca/n*100:.1f}%)")
    print(f"  {label_b}: {cb}/{n} correct ({cb/n*100:.1f}%)")
    print(f"\n  Exploration overlap:")
    print(f"    Avg IoU: {sum(ious)/len(ious):.3f}")
    print(f"    {label_a}-only cells:  {a_only_total} ({a_only_total*30/60:.0f} min total)")
    print(f"    {label_b}-only cells:  {b_only_total} ({b_only_total*30/60:.0f} min total)")
    print(f"    Shared cells: {shared_total} ({shared_total*30/60:.0f} min total)")

    # IoU 분포
    low = sum(1 for x in ious if x < 0.3)
    mid = sum(1 for x in ious if 0.3 <= x < 0.7)
    high = sum(1 for x in ious if x >= 0.7)
    print(f"\n  IoU distribution:")
    print(f"    Low (<0.3):   {low} questions — 완전히 다른 구간 참조")
    print(f"    Mid (0.3-0.7): {mid} questions — 부분 겹침")
    print(f"    High (>=0.7): {high} questions — 비슷한 구간 참조")

    # 정답 차이 vs 탐색 겹침
    print(f"\n  Correctness vs Coverage:")
    for tag, items in diff_correct.items():
        if items:
            avg_iou = sum(x[1] for x in items) / len(items)
            print(f"    {tag:<15}: {len(items)} questions, avg IoU={avg_iou:.3f}")

    # 한쪽만 맞춘 문제 중 IoU가 낮은 것 (다른 구간을 봐서 달라진 것)
    for tag, label in [("a_only", f"{label_a}만 맞춤"), ("b_only", f"{label_b}만 맞춤")]:
        items = diff_correct[tag]
        if items:
            low_iou = sorted(items, key=lambda x: x[1])[:5]
            print(f"\n  {label} + 낮은 IoU (다른 구간 참조로 결과 달라짐):")
            for qid, iou in low_iou:
                ra, rb = results_a[qid], results_b[qid]
                print(f"    {qid:<12} IoU={iou:.2f}  "
                      f"A={ra.get('pred','?')} B={rb.get('pred','?')} GT={ra.get('answer','?')}  "
                      f"video={ra.get('video_id','?')}")


def main():
    parser = argparse.ArgumentParser(description="탐색 구간 비교")
    parser.add_argument("dir_a", help="실험 A 디렉토리")
    parser.add_argument("dir_b", help="실험 B 디렉토리")
    parser.add_argument("--qid", type=str, default=None, help="특정 문제만 비교")
    parser.add_argument("--overlap_only", action="store_true", help="겹침 통계만")
    parser.add_argument("--show_all", action="store_true", help="모든 문제 개별 비교")
    parser.add_argument("--diff_only", action="store_true", help="결과가 다른 문제만")
    args = parser.parse_args()

    label_a = "/".join(args.dir_a.rstrip("/").split("/")[-2:])
    label_b = "/".join(args.dir_b.rstrip("/").split("/")[-2:])

    results_a = load_results(os.path.join(args.dir_a, "by_qid"))
    results_b = load_results(os.path.join(args.dir_b, "by_qid"))

    if not results_a:
        print(f"No results in {args.dir_a}/by_qid/")
        return
    if not results_b:
        print(f"No results in {args.dir_b}/by_qid/")
        return

    common = sorted(set(results_a.keys()) & set(results_b.keys()))
    print(f"  A: {label_a} ({len(results_a)} results)")
    print(f"  B: {label_b} ({len(results_b)} results)")
    print(f"  Common: {len(common)} questions")

    # 특정 문제
    if args.qid:
        if args.qid not in results_a or args.qid not in results_b:
            print(f"  Question {args.qid} not found in both experiments")
            return
        print_question_comparison(args.qid, results_a[args.qid], results_b[args.qid], label_a, label_b)
        return

    # 전체 통계
    print_aggregate_overlap(results_a, results_b, label_a, label_b)

    # 개별 문제 출력
    if args.show_all or args.diff_only:
        for qid in common:
            ra, rb = results_a[qid], results_b[qid]
            if args.diff_only and ra.get("correct") == rb.get("correct"):
                continue
            print_question_comparison(qid, ra, rb, label_a, label_b)
            print(f"  {'-'*60}")


if __name__ == "__main__":
    main()
