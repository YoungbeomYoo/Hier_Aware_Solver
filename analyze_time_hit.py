#!/usr/bin/env python3
"""
LVBench target time hit 분석

각 문제의 ground-truth time_reference와 솔버가 탐색한 시간 구간을 비교하여
target time을 얼마나 커버했는지 분석.

Usage:
    python analyze_time_hit.py output/lvbench_full_semantic/v3
    python analyze_time_hit.py output/lvbench_full_semantic/v3 --detail
    python analyze_time_hit.py output/lvbench_full_semantic/v3 --missed  # hit 못한 것만
"""

import os
import json
import argparse
from collections import defaultdict


def parse_time_ref(time_str):
    """'MM:SS-MM:SS' or 'HH:MM:SS-HH:MM:SS' → (start_sec, end_sec)"""
    parts = time_str.strip().split("-")
    if len(parts) != 2:
        return None
    try:
        s = _to_sec(parts[0].strip())
        e = _to_sec(parts[1].strip())
        return (s, e)
    except Exception:
        return None


def _to_sec(t):
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0])


def format_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def load_time_references(question_path):
    """LVBench jsonl에서 uid → time_reference 매핑."""
    refs = {}
    with open(question_path) as f:
        for line in f:
            d = json.loads(line)
            for qa in d.get("qa", []):
                uid = str(qa.get("uid", ""))
                tr = qa.get("time_reference", "")
                if uid and tr:
                    parsed = parse_time_ref(tr)
                    if parsed:
                        refs[uid] = {
                            "raw": tr,
                            "start": parsed[0],
                            "end": parsed[1],
                            "question_type": qa.get("question_type", []),
                        }
    return refs


def extract_explored_ranges(result):
    """결과 JSON에서 솔버가 탐색한 시간 구간 추출."""
    ranges = []
    for hc in result.get("hop_contexts", []):
        for leaf in hc.get("target_leaves", []):
            s = leaf.get("start_time")
            e = leaf.get("end_time")
            if s is not None and e is not None:
                ranges.append((float(s), float(e)))
    return ranges


def check_overlap(target_start, target_end, explored_ranges):
    """target time과 탐색 구간의 겹침 분석."""
    target_len = max(target_end - target_start, 1)

    # target을 1초 단위 셀로 변환
    target_cells = set(range(int(target_start), int(target_end) + 1))

    # explored를 1초 단위 셀로 변환
    explored_cells = set()
    for s, e in explored_ranges:
        for c in range(int(s), int(e) + 1):
            explored_cells.add(c)

    overlap = target_cells & explored_cells
    coverage = len(overlap) / len(target_cells) if target_cells else 0

    # 가장 가까운 탐색 구간까지의 거리
    min_dist = float("inf")
    if explored_ranges:
        target_mid = (target_start + target_end) / 2
        for s, e in explored_ranges:
            # 겹치면 거리 0
            if s <= target_end and e >= target_start:
                min_dist = 0
                break
            # 안 겹치면 최소 거리
            dist = min(abs(s - target_end), abs(e - target_start))
            min_dist = min(min_dist, dist)

    return {
        "coverage": coverage,
        "overlap_sec": len(overlap),
        "target_sec": len(target_cells),
        "explored_sec": len(explored_cells),
        "min_distance": min_dist if min_dist != float("inf") else -1,
        "hit": coverage > 0,  # 조금이라도 겹침
        "full_hit": coverage >= 0.5,  # 50% 이상 커버
    }


def main():
    parser = argparse.ArgumentParser(description="LVBench target time hit 분석")
    parser.add_argument("result_dir", help="결과 디렉토리 (by_qid/ 포함)")
    parser.add_argument("--detail", action="store_true", help="개별 문제 출력")
    parser.add_argument("--missed", action="store_true", help="hit 못한 문제만 출력")
    parser.add_argument("--correct_only", action="store_true", help="맞춘 문제만")
    parser.add_argument("--wrong_only", action="store_true", help="틀린 문제만")
    args = parser.parse_args()

    # LVBench time reference 로드
    question_path = "/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl"
    time_refs = load_time_references(question_path)
    print(f"  LVBench time references: {len(time_refs)} questions")

    # 결과 로드
    by_qid_dir = os.path.join(args.result_dir, "by_qid")
    if not os.path.isdir(by_qid_dir):
        print(f"  No by_qid/ in {args.result_dir}")
        return

    results = {}
    for fname in sorted(os.listdir(by_qid_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(by_qid_dir, fname)) as f:
                r = json.load(f)
            qid = str(r.get("question_id", fname.replace(".json", "")))
            results[qid] = r
        except Exception:
            pass

    print(f"  Results loaded: {len(results)} questions")
    print()

    # 분석
    stats = {
        "total": 0,
        "hit": 0,
        "full_hit": 0,
        "correct_hit": 0,
        "correct_miss": 0,
        "wrong_hit": 0,
        "wrong_miss": 0,
        "coverages": [],
        "distances_miss": [],
    }
    by_qtype = defaultdict(lambda: {"total": 0, "hit": 0, "correct": 0, "correct_hit": 0})
    detail_rows = []

    for qid in sorted(results.keys()):
        r = results[qid]
        if qid not in time_refs:
            continue

        ref = time_refs[qid]
        explored = extract_explored_ranges(r)
        overlap_info = check_overlap(ref["start"], ref["end"], explored)

        correct = r.get("correct", False)
        if args.correct_only and not correct:
            continue
        if args.wrong_only and correct:
            continue

        stats["total"] += 1
        stats["coverages"].append(overlap_info["coverage"])

        if overlap_info["hit"]:
            stats["hit"] += 1
        if overlap_info["full_hit"]:
            stats["full_hit"] += 1

        if correct and overlap_info["hit"]:
            stats["correct_hit"] += 1
        elif correct and not overlap_info["hit"]:
            stats["correct_miss"] += 1
        elif not correct and overlap_info["hit"]:
            stats["wrong_hit"] += 1
        else:
            stats["wrong_miss"] += 1

        if not overlap_info["hit"] and overlap_info["min_distance"] >= 0:
            stats["distances_miss"].append(overlap_info["min_distance"])

        # per question type
        for qt in ref.get("question_type", ["unknown"]):
            by_qtype[qt]["total"] += 1
            if overlap_info["hit"]:
                by_qtype[qt]["hit"] += 1
            if correct:
                by_qtype[qt]["correct"] += 1
                if overlap_info["hit"]:
                    by_qtype[qt]["correct_hit"] += 1

        detail_rows.append({
            "qid": qid,
            "correct": correct,
            "pred": r.get("pred", "?"),
            "answer": r.get("answer", "?"),
            "target": ref["raw"],
            "target_sec": f"{format_time(ref['start'])}-{format_time(ref['end'])}",
            "coverage": overlap_info["coverage"],
            "hit": overlap_info["hit"],
            "min_dist": overlap_info["min_distance"],
            "explored_count": len(explored),
            "hops": r.get("total_hops", "?"),
            "video_id": r.get("video_id", "?"),
        })

    # === Summary ===
    n = stats["total"]
    if n == 0:
        print("  No matching results found.")
        return

    print(f"{'='*65}")
    print(f"  Target Time Hit Analysis — {n} questions")
    print(f"{'='*65}")

    hit_rate = stats["hit"] / n * 100
    full_rate = stats["full_hit"] / n * 100
    avg_cov = sum(stats["coverages"]) / n * 100

    print(f"  Any overlap (hit):    {stats['hit']}/{n} ({hit_rate:.1f}%)")
    print(f"  >=50% coverage:       {stats['full_hit']}/{n} ({full_rate:.1f}%)")
    print(f"  Avg target coverage:  {avg_cov:.1f}%")

    print(f"\n  Correctness × Hit:")
    print(f"    Correct + Hit:   {stats['correct_hit']:>4}  (정답 맞춤 + target 봄)")
    print(f"    Correct + Miss:  {stats['correct_miss']:>4}  (정답 맞춤 + target 안봄 — lucky?)")
    print(f"    Wrong + Hit:     {stats['wrong_hit']:>4}  (틀림 + target 봤는데 틀림)")
    print(f"    Wrong + Miss:    {stats['wrong_miss']:>4}  (틀림 + target 안봄)")

    total_correct = stats["correct_hit"] + stats["correct_miss"]
    if total_correct > 0:
        print(f"\n  맞춘 문제 중 target hit rate: {stats['correct_hit']}/{total_correct} ({stats['correct_hit']/total_correct*100:.1f}%)")
    total_wrong = stats["wrong_hit"] + stats["wrong_miss"]
    if total_wrong > 0:
        print(f"  틀린 문제 중 target hit rate: {stats['wrong_hit']}/{total_wrong} ({stats['wrong_hit']/total_wrong*100:.1f}%)")

    # Miss distance
    if stats["distances_miss"]:
        dists = sorted(stats["distances_miss"])
        avg_d = sum(dists) / len(dists)
        med_d = dists[len(dists)//2]
        print(f"\n  Miss한 문제의 target까지 거리:")
        print(f"    평균: {avg_d:.0f}초 ({avg_d/60:.1f}분)")
        print(f"    중앙값: {med_d:.0f}초 ({med_d/60:.1f}분)")
        print(f"    <30초: {sum(1 for d in dists if d < 30)}")
        print(f"    <1분:  {sum(1 for d in dists if d < 60)}")
        print(f"    <5분:  {sum(1 for d in dists if d < 300)}")

    # Per question type
    print(f"\n  Per Question Type:")
    print(f"    {'Type':<30} {'Hit%':>6} {'Acc%':>6} {'AccIfHit':>8} {'Total':>6}")
    print(f"    {'-'*56}")
    for qt in sorted(by_qtype.keys(), key=lambda x: -by_qtype[x]["total"]):
        info = by_qtype[qt]
        t = info["total"]
        hr = info["hit"] / t * 100 if t else 0
        ar = info["correct"] / t * 100 if t else 0
        ahr = info["correct_hit"] / info["hit"] * 100 if info["hit"] else 0
        print(f"    {qt:<30} {hr:>5.1f}% {ar:>5.1f}% {ahr:>7.1f}% {t:>5}")

    # Coverage distribution
    covs = stats["coverages"]
    zero = sum(1 for c in covs if c == 0)
    low = sum(1 for c in covs if 0 < c < 0.3)
    mid = sum(1 for c in covs if 0.3 <= c < 0.7)
    high = sum(1 for c in covs if 0.7 <= c < 1.0)
    full = sum(1 for c in covs if c >= 1.0)
    print(f"\n  Coverage distribution:")
    print(f"    0% (miss):     {zero:>4} ({zero/n*100:.1f}%)")
    print(f"    1-29%:         {low:>4} ({low/n*100:.1f}%)")
    print(f"    30-69%:        {mid:>4} ({mid/n*100:.1f}%)")
    print(f"    70-99%:        {high:>4} ({high/n*100:.1f}%)")
    print(f"    100% (full):   {full:>4} ({full/n*100:.1f}%)")

    # Detail output
    if args.detail or args.missed:
        print(f"\n{'='*65}")
        if args.missed:
            print(f"  Missed questions (no overlap with target):")
        else:
            print(f"  All questions:")
        print(f"{'='*65}")

        for row in sorted(detail_rows, key=lambda x: x["coverage"]):
            if args.missed and row["hit"]:
                continue
            mark = "O" if row["correct"] else "X"
            hit_str = f"cov={row['coverage']*100:.0f}%" if row["hit"] else f"MISS (dist={row['min_dist']:.0f}s)"
            print(f"  {row['qid']:<8} {mark} pred={row['pred']} gt={row['answer']}  "
                  f"target={row['target']:<15} {hit_str}  "
                  f"explored={row['explored_count']} segs, {row['hops']} hops  "
                  f"video={row['video_id']}")


if __name__ == "__main__":
    main()
