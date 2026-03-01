#!/usr/bin/env python3
"""
Segment Selection → Raw Caption → Answer (End-to-End)

Flow:
1. Exp2 segment selection 결과 로드 (이미 선택된 5개 구간)
2. 해당 구간의 leaf raw caption을 메모리에서 추출
3. Raw caption context로 질문에 답변
4. 정답률 측정

Usage:
    python eval_segment_answer.py \
        --qid_list output/lvbench_mini_qid_list.tsv \
        --output_dir output/eval_segment_answer
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


ANSWER_PROMPT = """You are given detailed descriptions of specific video segments and a question about the video.

=== Selected Video Segments ===
{context}

=== Question ===
{question}

=== Choices ===
{options_text}

Based on the video segment descriptions above, answer the question by selecting the best choice.

Output ONLY valid JSON:
{{
    "reasoning": "Brief explanation of your answer",
    "answer": "A"
}}

Rules:
- Select exactly one answer from A, B, C, D.
- Base your answer only on the information provided in the segments above.
- If the segments don't contain enough information, make your best guess."""


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


def get_leaf_captions(tree, selected_segments):
    """선택된 segment 시간 범위에 해당하는 leaf의 raw caption을 찾는다."""
    # 모든 leaf를 시간순으로 수집
    all_leaves = []
    for l1_node in tree.get("Level_1", []):
        for child in l1_node.get("children", []):
            st = float(child.get("start_time", 0))
            et = float(child.get("end_time", 0))
            caption = child.get("caption", "")
            summary = child.get("summary", "")
            all_leaves.append((st, et, caption, summary))
    all_leaves.sort(key=lambda x: x[0])

    # 선택된 segment에 overlap하는 leaf들의 caption 수집
    captions = []
    for seg in selected_segments:
        seg_st, seg_et = seg["start"], seg["end"]
        for st, et, caption, summary in all_leaves:
            if st < seg_et and seg_st < et:  # overlap
                captions.append((st, et, caption, summary))

    # 중복 제거 (같은 시간대의 leaf가 여러 segment에 걸칠 수 있음)
    seen = set()
    unique = []
    for st, et, cap, summ in captions:
        key = (st, et)
        if key not in seen:
            seen.add(key)
            unique.append((st, et, cap, summ))
    unique.sort(key=lambda x: x[0])
    return unique


def build_caption_context(leaves):
    """선택된 leaf들의 raw caption으로 context 생성."""
    lines = []
    for st, et, caption, summary in leaves:
        lines.append(f"[{st:.0f}s-{et:.0f}s] {caption}")
    return "\n\n".join(lines)


def build_summary_context(leaves):
    """선택된 leaf들의 summary로 context 생성 (비교용)."""
    lines = []
    for st, et, caption, summary in leaves:
        lines.append(f"[{st:.0f}s-{et:.0f}s] {summary}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid_list", required=True)
    parser.add_argument("--question_path", default="/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl")
    parser.add_argument("--mem_dir", default="/lustre/youngbeom/DyHiStreamMem/poc/results/LVBench/stage2_v9")
    parser.add_argument("--seg_result_dir", default="./output/eval_segment_selection/by_qid",
                        help="Exp2 segment selection results directory")
    parser.add_argument("--output_dir", default="./output/eval_segment_answer")
    parser.add_argument("--mode", choices=["caption", "summary"], default="caption",
                        help="caption: raw caption, summary: leaf summary")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "by_qid"), exist_ok=True)

    questions = load_questions(args.qid_list, args.question_path)
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
    correct = 0
    by_type = {}
    skipped = 0

    for qid, qa in sorted(questions.items()):
        vid = qa["video_id"]
        out_path = os.path.join(args.output_dir, "by_qid", f"{qid}.json")

        # Skip if cached
        if os.path.exists(out_path):
            try:
                cached = json.load(open(out_path))
                if cached.get("pred"):
                    total += 1
                    if cached.get("correct"):
                        correct += 1
                    for qt in qa.get("question_type", ["unknown"]):
                        if qt not in by_type:
                            by_type[qt] = {"c": 0, "t": 0}
                        by_type[qt]["t"] += 1
                        if cached.get("correct"):
                            by_type[qt]["c"] += 1
                    print(f"  [{qid}] cached — pred={cached['pred']} correct={cached['correct']}")
                    continue
            except:
                pass

        # Load Exp2 segment selection result
        seg_path = os.path.join(args.seg_result_dir, f"{qid}.json")
        if not os.path.exists(seg_path):
            print(f"  [{qid}] no segment selection result — skipping")
            skipped += 1
            continue

        seg_result = json.load(open(seg_path))
        selected_segments = seg_result.get("segments", [])
        if not selected_segments:
            print(f"  [{qid}] no segments selected — skipping")
            skipped += 1
            continue

        # Load memory
        mem_path = os.path.join(args.mem_dir, f"{vid}.json")
        if not os.path.exists(mem_path):
            print(f"  [{qid}] no memory for {vid}")
            skipped += 1
            continue
        mem = json.load(open(mem_path))
        tree = mem.get("streaming_memory_tree", {})
        if not tree:
            skipped += 1
            continue

        # Get captions for selected segments
        leaves = get_leaf_captions(tree, selected_segments)
        if not leaves:
            print(f"  [{qid}] no matching leaves found")
            skipped += 1
            continue

        if args.mode == "caption":
            context = build_caption_context(leaves)
        else:
            context = build_summary_context(leaves)

        opt_text = "\n".join(qa["options"]) if qa["options"] else "A. Option A\nB. Option B\nC. Option C\nD. Option D"

        prompt = ANSWER_PROMPT.format(
            context=context,
            question=qa["question"],
            options_text=opt_text,
        )

        # LLM call
        try:
            result = llm_fn(prompt, max_tokens=400)
        except Exception as e:
            print(f"  [{qid}] LLM error: {e}")
            skipped += 1
            continue

        # Parse answer
        pred = None
        if isinstance(result, dict) and "answer" in result:
            ans_str = str(result["answer"]).upper().strip()
            m = re.search(r"[ABCD]", ans_str)
            if m:
                pred = m.group(0)

        if not pred:
            # Fallback: try to find answer in reasoning
            if isinstance(result, dict):
                text = json.dumps(result)
            else:
                text = str(result)
            m = re.search(r"\b([ABCD])\b", text)
            pred = m.group(1) if m else "A"

        # Check correctness
        gt = qa["answer"].strip().upper()
        is_correct = (pred == gt)

        total += 1
        if is_correct:
            correct += 1

        for qt in qa.get("question_type", ["unknown"]):
            if qt not in by_type:
                by_type[qt] = {"c": 0, "t": 0}
            by_type[qt]["t"] += 1
            if is_correct:
                by_type[qt]["c"] += 1

        # Save
        out = {
            "question_id": qid,
            "video_id": vid,
            "question": qa["question"],
            "answer_gt": gt,
            "pred": pred,
            "correct": is_correct,
            "mode": args.mode,
            "n_leaves_used": len(leaves),
            "context_chars": len(context),
            "segments_used": [{"start": l[0], "end": l[1]} for l in leaves],
            "reasoning": result.get("reasoning", "") if isinstance(result, dict) else "",
            "question_type": qa.get("question_type", []),
        }
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"  [{qid}] pred={pred} gt={gt} {'O' if is_correct else 'X'} | {len(leaves)} leaves | {len(context)/1000:.1f}K chars | mode={args.mode}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Segment → Answer ({args.mode}) — {total} questions ({skipped} skipped)")
    print(f"{'='*60}")
    print(f"  Accuracy: {correct}/{total} = {correct/max(total,1)*100:.1f}%")

    print(f"\nBy question type:")
    for qt in sorted(by_type.keys()):
        d = by_type[qt]
        print(f"  {qt:30s}: {d['c']}/{d['t']} = {d['c']/max(d['t'],1)*100:.1f}%")

    # Save summary
    summary = {
        "mode": args.mode,
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "skipped": skipped,
        "by_type": by_type,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
