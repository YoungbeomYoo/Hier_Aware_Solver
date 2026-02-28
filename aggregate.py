#!/usr/bin/env python3
"""
Aggregate — 병렬 실행된 per-video 결과를 합쳐서 summary 생성

Usage:
    python aggregate.py --output_dir ./output/video_mme_tree_search_visual/v1

by_qid/ 폴더 안의 모든 .json 파일을 읽어서 summary.json 생성.
"""

import os
import json
import argparse


def aggregate(output_dir: str):
    by_qid_dir = os.path.join(output_dir, "by_qid")
    if not os.path.isdir(by_qid_dir):
        print(f"Error: {by_qid_dir} not found")
        return

    results = []
    for fname in sorted(os.listdir(by_qid_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(by_qid_dir, fname)
        with open(fpath) as f:
            results.append(json.load(f))

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total > 0 else 0.0

    # By video
    by_video = {}
    for r in results:
        vid = r.get("video_id", "unknown")
        if vid not in by_video:
            by_video[vid] = {"total": 0, "correct": 0}
        by_video[vid]["total"] += 1
        if r.get("correct"):
            by_video[vid]["correct"] += 1

    for vid in by_video:
        v = by_video[vid]
        v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0

    # By question type
    by_type = {}
    for r in results:
        q_type = r.get("question_type", "unknown")
        if q_type not in by_type:
            by_type[q_type] = {"total": 0, "correct": 0}
        by_type[q_type]["total"] += 1
        if r.get("correct"):
            by_type[q_type]["correct"] += 1

    for qt in by_type:
        v = by_type[qt]
        v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_video": by_video,
        "by_question_type": by_type,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Aggregated {total} results from {len(by_video)} videos")
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"Summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate parallel results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory containing by_qid/")
    args = parser.parse_args()
    aggregate(args.output_dir)
