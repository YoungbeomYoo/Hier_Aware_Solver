#!/usr/bin/env python3
"""
Segment Selection Evaluation — LLM이 hierarchical summary로 target 구간을 잘 잡는가?

Flow:
1. Level_N ~ Level_1 summary + key_elements를 LLM에게 보여줌
2. "이 질문에 답하려면 어느 구간을 먼저 봐야 할까? 5개 골라"
3. 선택된 구간이 time_reference(GT)와 overlap하는지 측정

Usage:
    python eval_segment_selection.py \
        --qid_list output/lvbench_mini_qid_list.tsv \
        --output_dir output/eval_segment_selection
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SEGMENT_SELECT_PROMPT = """You are given a hierarchical summary of a long video and a question about it.

=== Video Overview ===
{context}

=== Question ===
{question}

=== Choices ===
{options_text}

Based on the video overview above, select the {n_select} most relevant time segments to examine in order to answer this question.

Output ONLY valid JSON:
{{
    "reasoning": "Brief explanation of why these segments are relevant",
    "segments": [
        {{"start": 30.0, "end": 60.0}},
        {{"start": 120.0, "end": 150.0}}
    ]
}}

Rules:
- Select exactly {n_select} segments from the Level_1 time ranges shown above.
- Use the exact start/end times from the overview (in seconds).
- Order by relevance (most relevant first).
- Focus on segments that are most likely to contain the answer."""


def parse_time_reference(time_ref: str):
    if not time_ref:
        return None
    parts = time_ref.split("-")
    if len(parts) != 2:
        return None
    def to_sec(s):
        tokens = [float(t) for t in s.strip().split(":")]
        if len(tokens) == 3: return tokens[0]*3600 + tokens[1]*60 + tokens[2]
        elif len(tokens) == 2: return tokens[0]*60 + tokens[1]
        return tokens[0]
    try:
        return (to_sec(parts[0]), to_sec(parts[1]))
    except:
        return None


def overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def build_coarse_context(tree):
    """Level_1 children(leaf)의 summary + key_elements — 개별 segment 단위.

    각 leaf의 30초 구간을 별도 항목으로 보여줌.
    상위 레벨 summary는 overview section에 포함.
    """
    lines = []

    # --- 상위 레벨 (Level_N ~ Level_2) overview ---
    level_names = sorted(
        [k for k in tree.keys() if k.startswith("Level_") and int(k.split("_")[1]) >= 2],
        key=lambda x: int(x.split("_")[1]),
        reverse=True,
    )
    if level_names:
        lines.append("=== Video Overview ===")
        for level_name in level_names:
            for node in tree.get(level_name, []):
                summary = node.get("summary", "")
                if not summary:
                    continue
                segs = node.get("time_segments", [])
                seg_strs = ["%.0fs-%.0fs" % (float(s[0]), float(s[1]))
                            for s in segs if isinstance(s, (list, tuple)) and len(s) >= 2]
                time_str = ", ".join(seg_strs[:5])
                if len(seg_strs) > 5:
                    time_str += f", ... ({len(seg_strs)} segments)"
                lines.append(f"[{level_name}] [{time_str}] {summary[:200]}")
        lines.append("")

    # --- Level_1 segments (개별 단위) ---
    lines.append("=== Segments ===")
    segments = []
    for l1_node in tree.get("Level_1", []):
        summary = l1_node.get("summary", "")
        ke = l1_node.get("key_elements", {})
        ke_brief = ""
        for field in ["actions", "objects", "persons"]:
            vals = ke.get(field, [])
            if vals:
                ke_brief += " | %s: %s" % (field, ", ".join(str(v) for v in vals[:4]))

        for s in l1_node.get("time_segments", []):
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                st, en = float(s[0]), float(s[1])
                segments.append((st, en, summary, ke_brief))

    # 시간순 정렬
    segments.sort(key=lambda x: x[0])
    for st, en, summary, ke_brief in segments:
        lines.append("[%.0fs-%.0fs] %s%s" % (st, en, summary, ke_brief))

    return "\n".join(lines), len(segments)


def load_questions(qid_list_path, question_path):
    qid_vid = {}
    with open(qid_list_path) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2:
                qid_vid[p[0]] = p[1]

    questions = {}
    with open(question_path) as f:
        for line in f:
            entry = json.loads(line)
            vid = entry["key"]
            for qa in entry.get("qa", []):
                uid = str(qa.get("uid", ""))
                if uid in qid_vid and qid_vid[uid] == vid:
                    raw_q = qa.get("question", "")
                    match = re.search(r"\n\(A\)", raw_q)
                    if match:
                        qt = raw_q[:match.start()].strip()
                        opts = re.findall(r"\([A-D]\)\s*[^\n]+", raw_q[match.start():])
                        opts = [re.sub(r"^\(([A-D])\)\s*", r"\1. ", o) for o in opts]
                    else:
                        qt = raw_q
                        opts = []
                    questions[uid] = {
                        "question_id": uid,
                        "video_id": vid,
                        "question": qt,
                        "options": opts,
                        "answer": qa.get("answer", ""),
                        "time_reference": qa.get("time_reference", ""),
                        "question_type": qa.get("question_type", []),
                    }
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid_list", required=True)
    parser.add_argument("--question_path", default="/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl")
    parser.add_argument("--mem_dir", default="/lustre/youngbeom/DyHiStreamMem/poc/results/LVBench/stage2_v9")
    parser.add_argument("--output_dir", default="./output/eval_segment_selection")
    parser.add_argument("--n_select", type=int, default=5)
    parser.add_argument("--question_id", type=str, default=None)
    parser.add_argument("--video_id", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "by_qid"), exist_ok=True)

    questions = load_questions(args.qid_list, args.question_path)

    # Single question mode
    if args.question_id:
        questions = {k: v for k, v in questions.items() if k == args.question_id}

    if not questions:
        print("No questions to process")
        return

    # Load model
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model_path = "/scratch2/youngbeom/ckpt/Qwen3-VL-8B-Instruct"
    print(f"Loading model: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    from components.vlm import TextOnlyLLM
    llm = TextOnlyLLM(model, processor)
    llm_fn = lambda prompt, max_tokens=400: llm.reason(prompt, max_tokens=max_tokens)

    # Process
    total = 0
    hits_at = {1: 0, 3: 0, 5: 0}
    by_type = {}

    for qid, qa in sorted(questions.items()):
        vid = qa["video_id"]
        out_path = os.path.join(args.output_dir, "by_qid", f"{qid}.json")

        # Skip if cached
        if os.path.exists(out_path):
            try:
                cached = json.load(open(out_path))
                if cached.get("segments"):
                    # Re-evaluate from cache
                    gt = parse_time_reference(qa["time_reference"])
                    if gt:
                        total += 1
                        for k in hits_at:
                            top = cached["segments"][:k]
                            if any(overlap(s["start"], s["end"], gt[0], gt[1]) for s in top):
                                hits_at[k] += 1
                        for qt in qa.get("question_type", ["unknown"]):
                            if qt not in by_type:
                                by_type[qt] = {k: 0 for k in hits_at}
                                by_type[qt]["total"] = 0
                            by_type[qt]["total"] += 1
                            for k in hits_at:
                                top = cached["segments"][:k]
                                if any(overlap(s["start"], s["end"], gt[0], gt[1]) for s in top):
                                    by_type[qt][k] += 1
                    print(f"  [{qid}] cached — skipping")
                    continue
            except:
                pass

        # Load memory
        mem_path = os.path.join(args.mem_dir, f"{vid}.json")
        if not os.path.exists(mem_path):
            print(f"  [{qid}] no memory for {vid}")
            continue
        mem = json.load(open(mem_path))
        tree = mem.get("streaming_memory_tree", {})
        if not tree:
            continue

        gt = parse_time_reference(qa["time_reference"])
        if gt is None:
            continue

        # Build context
        context, n_segs = build_coarse_context(tree)
        opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(qa["options"]))

        prompt = SEGMENT_SELECT_PROMPT.format(
            context=context,
            question=qa["question"],
            options_text=opt_text,
            n_select=args.n_select,
        )

        # LLM call
        try:
            result = llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"  [{qid}] LLM error: {e}")
            continue

        # Parse segments
        segments = []
        if isinstance(result, dict) and "segments" in result:
            for seg in result["segments"]:
                if isinstance(seg, dict) and "start" in seg and "end" in seg:
                    segments.append({
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                    })

        # Evaluate
        total += 1
        hit_info = {}
        for k in hits_at:
            top = segments[:k]
            hit = any(overlap(s["start"], s["end"], gt[0], gt[1]) for s in top)
            if hit:
                hits_at[k] += 1
            hit_info[f"hit@{k}"] = hit

        for qt in qa.get("question_type", ["unknown"]):
            if qt not in by_type:
                by_type[qt] = {k: 0 for k in hits_at}
                by_type[qt]["total"] = 0
            by_type[qt]["total"] += 1
            for k in hits_at:
                if hit_info.get(f"hit@{k}"):
                    by_type[qt][k] += 1

        # Save
        out = {
            "question_id": qid,
            "video_id": vid,
            "question": qa["question"],
            "time_reference": qa["time_reference"],
            "gt_range": list(gt),
            "segments": segments,
            "reasoning": result.get("reasoning", "") if isinstance(result, dict) else "",
            "hit_info": hit_info,
            "n_segments": len(segments),
        }
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        hit_str = " ".join(f"@{k}={'Y' if hit_info[f'hit@{k}'] else 'N'}" for k in hits_at)
        print(f"  [{qid}] {len(segments)} segs | {hit_str} | gt={gt[0]:.0f}-{gt[1]:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"Segment Selection — {total} questions")
    print(f"{'='*60}")
    for k in sorted(hits_at.keys()):
        print(f"  Hit@{k}: {hits_at[k]}/{total} = {hits_at[k]/max(total,1)*100:.1f}%")

    print(f"\nBy question type (Hit@{args.n_select}):")
    for qt in sorted(by_type.keys()):
        d = by_type[qt]
        t = d["total"]
        h = d[args.n_select]
        print(f"  {qt:30s}: {h}/{t} = {h/max(t,1)*100:.1f}%")

    # Save summary
    summary = {
        "total": total,
        "hits_at": {str(k): v for k, v in hits_at.items()},
        "by_type": by_type,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
