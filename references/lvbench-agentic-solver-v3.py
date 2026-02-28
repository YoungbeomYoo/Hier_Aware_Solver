"""
LVBench Agentic Solver v3 — Rule-Based Filter + Budgeted Leaf Hops

6-Step Flow:
  1. Question Decomposition → extract key cues
  2. Flatten all leaves from memory tree
  3. Rule-based keyword filter (no LLM) → marked/unmarked leaves
  4. Budgeted hop loop: LLM selects 10 leaves → solvability → frame VLM
  5. Width/Depth frame exploration
  6. Expansion fallback to unmarked leaves
"""

import os
import json
import re
import math
import gc
import argparse
from glob import glob
from tqdm import tqdm

# -------------------------
# Default Paths
# -------------------------
MEMORY_DIR = '/lustre/youngbeom/DyHiStreamMem/poc/results/LVBench/stage2_debug_v8'
QUESTION_PATH = '/lustre/youngbeom/DyHiStreamMem/datasets/LVBench/video_info.meta.jsonl'
OUTPUT_PATH = '/lustre/youngbeom/DyHiStreamMem/vqa/video-mme/output/lvbench_agentic3'
MODEL_PATH = '/scratch2/youngbeom/ckpt/Qwen3-VL-8B-Instruct'
VIDEO_ROOT = '/scratch2/youngbeom/datasets/LVBench/all_videos'
CAPTION_DIR = '/lustre/youngbeom/yeeun/LVBench_caption_gpt5mini_rename'

# -------------------------
# Shared Utilities
# -------------------------
def safe_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_memory_files(memory_dir, prefer_synced=True):
    idx = {}
    json_files = sorted(glob(os.path.join(memory_dir, "*.json")))
    for fp in json_files:
        basename = os.path.splitext(os.path.basename(fp))[0]
        is_synced = basename.endswith("_synced")
        vid = basename.replace("_synced", "") if is_synced else basename
        if vid.startswith("unsynced_"):
            continue
        if vid in idx and not is_synced:
            continue
        if prefer_synced and is_synced:
            idx[vid] = read_json(fp)
        elif vid not in idx:
            idx[vid] = read_json(fp)
    print(f"Loaded {len(idx)} memory files from {memory_dir}")
    return idx


def load_raw_captions(caption_dir, video_id):
    """Load raw captions for a video. Returns {start_time_float: caption_text}."""
    caption_path = os.path.join(caption_dir, f"{video_id}.json")
    if not os.path.exists(caption_path):
        return {}
    captions = read_json(caption_path)
    caption_map = {}
    for entry in captions:
        st = float(entry.get("start", -1))
        cap = entry.get("caption", "")
        if st >= 0 and cap:
            caption_map[st] = cap
    return caption_map


def load_lvbench_questions(question_path):
    idx = {}
    with open(question_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            video_key = entry.get("key", "")
            qa_list = entry.get("qa", [])
            for qa in qa_list:
                qa["video_key"] = video_key
                qa["video_type"] = entry.get("type", "")
            if video_key:
                idx[video_key] = qa_list
    total_qa = sum(len(v) for v in idx.values())
    print(f"Loaded {len(idx)} videos, {total_qa} total QAs from {question_path}")
    return idx


def parse_lvbench_question(qa):
    raw_q = qa.get("question", "")
    match = re.search(r'\n\(A\)', raw_q)
    if match:
        question_text = raw_q[:match.start()].strip()
        options_text = raw_q[match.start():].strip()
        options = re.findall(r'\([A-D]\)\s*[^\n]+', options_text)
        options = [re.sub(r'^\(([A-D])\)\s*', r'\1. ', o) for o in options]
    else:
        question_text = raw_q
        options = []
    return question_text, options


def parse_time_reference(time_ref_str):
    if not time_ref_str:
        return None
    parts = re.split(r'\s*-\s*', time_ref_str.strip())
    if len(parts) != 2:
        return None
    def _to_sec(t):
        segs = t.split(':')
        if len(segs) == 3:
            return int(segs[0]) * 3600 + int(segs[1]) * 60 + float(segs[2])
        elif len(segs) == 2:
            return int(segs[0]) * 60 + float(segs[1])
        return float(t)
    try:
        return (_to_sec(parts[0]), _to_sec(parts[1]))
    except Exception:
        return None


# -------------------------
# JSON extraction from LLM response
# -------------------------
def extract_json_from_response(response):
    start = response.find('{')
    if start == -1:
        return {}
    depth, in_string, escape = 0, False, False
    end = -1
    for i, c in enumerate(response[start:], start):
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            pass
    result = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', response[start:]):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r'"(\w+)"\s*:\s*(-?\d+)', response[start:]):
        if m.group(1) not in result:
            result[m.group(1)] = int(m.group(2))
    for m in re.finditer(r'"(\w+)"\s*:\s*\[([^\]]*)\]', response[start:]):
        key = m.group(1)
        if key not in result:
            items = m.group(2).strip()
            if items:
                try:
                    result[key] = json.loads(f"[{items}]")
                except json.JSONDecodeError:
                    result[key] = [x.strip().strip('"') for x in items.split(',')]
    for m in re.finditer(r'"(\w+)"\s*:\s*(true|false)', response[start:]):
        key = m.group(1)
        if key not in result:
            result[key] = m.group(2) == "true"
    return result


# -------------------------
# AgenticSolver v3
# -------------------------
class AgenticSolverV3:
    def __init__(self, model, processor, max_hops=5, max_frames=32,
                 max_memory_chars=120000, video_root=VIDEO_ROOT, dry_run=False,
                 image_token_size=256, leaf_budget=10, depth_budget=5,
                 caption_dir=CAPTION_DIR):
        self.model = model
        self.processor = processor
        self.max_hops = max_hops
        self.max_frames = max_frames
        self.max_memory_chars = max_memory_chars
        self.video_root = video_root
        self.dry_run = dry_run
        self.image_token_size = image_token_size
        self.leaf_budget = leaf_budget
        self.depth_budget = depth_budget
        self.caption_dir = caption_dir

    # ================================================================
    # LLM inference helpers
    # ================================================================

    def _llm_text_reasoning(self, prompt_text, max_tokens=256):
        if self.dry_run:
            print(f"[DRY_RUN _llm_text_reasoning] prompt ({len(prompt_text)} chars):")
            print(prompt_text[:600] + "..." if len(prompt_text) > 600 else prompt_text)
            return {}

        import torch

        messages = [
            {"role": "system", "content": "You are a logical AI assistant that outputs strictly in JSON format."},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(text=text, images=None, videos=None, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0].strip()

        del inputs, output_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        print(f"  [LLM raw] {repr(response[:300])}")
        parsed = extract_json_from_response(response)
        if parsed:
            return parsed
        print(f"  [LLM] JSON parsing failed, returning {{}}")
        return {}

    def _vlm_inference(self, frames_np, memory_context, question, options, hop_history=None, max_tokens=200):
        if self.dry_run:
            print(f"[DRY_RUN _vlm_inference] {len(frames_np)} frames, memory({len(memory_context)} chars)")
            return {"answer": "", "confidence": "low", "observation": "dry_run", "raw_response": ""}

        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        opt_text = "\n".join(options)

        history_text = ""
        if hop_history:
            history_text = "[Previous Observations]\n"
            for h in hop_history:
                history_text += (
                    f"Hop {h['hop']}: {h.get('type', 'unknown')}\n"
                    f"  Observation: {h.get('observation', 'N/A')}\n"
                    f"  Previous answer: {h.get('answer', 'N/A')} (confidence: {h.get('confidence', 'N/A')})\n\n"
                )
            history_text += "Based on ALL observations so far, answer the question.\n"

        frame_list = [Image.fromarray(f) for f in frames_np]

        VLM_MEMORY_BUDGET = 20000
        if len(memory_context) > VLM_MEMORY_BUDGET:
            memory_context = memory_context[:VLM_MEMORY_BUDGET] + "\n... [truncated for memory budget]"

        pixel_count = self.image_token_size * 32 * 32

        user_content = []
        if history_text:
            user_content.append({"type": "text", "text": history_text})

        user_content.append({"type": "text", "text": f"The following {len(frame_list)} frames are from the relevant video segments:"})
        for img in frame_list:
            user_content.append({
                "type": "image", "image": img,
                "min_pixels": pixel_count, "max_pixels": pixel_count,
            })

        user_content.append({"type": "text", "text": f"\n[Video Memory Context]\n{memory_context}"})
        user_content.append({"type": "text", "text": (
            f"\nQuestion: {question}\nOptions:\n{opt_text}\n\n"
            "Select the best answer. Output ONLY valid JSON:\n"
            '{"answer": "B", "confidence": "high", "observation": "Key visual details that support the answer"}\n'
            '- "high": Clear visual/textual evidence supports your answer.\n'
            '- "low": Uncertain, describe what additional information would help.'
        )})

        sys_prompt = (
            "You are an expert video analyzer. "
            "Analyze the video frames and memory context to answer the multiple choice question. "
            "Output ONLY valid JSON with keys: answer, confidence, observation."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": user_content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos_vis = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=text, images=images, videos=videos_vis, return_tensors="pt",
        ).to(self.model.device)

        print(f"  [VLM] input_ids: {inputs.input_ids.shape[1]} tokens, {len(frame_list)} frames")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                use_cache=True, top_p=None, num_beams=1, top_k=None, temperature=None,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0].strip()

        del inputs, output_ids, generated_ids_trimmed, user_content, frame_list, images, videos_vis
        torch.cuda.empty_cache()

        print(f"  [VLM raw] {repr(response[:300])}")
        parsed = extract_json_from_response(response)

        answer_str = parsed.get("answer", "")
        confidence = parsed.get("confidence", "low")
        observation = parsed.get("observation", "")

        if not answer_str:
            answer_str = response.strip()
            confidence = "low"
        if confidence not in ("high", "low"):
            confidence = "low"

        return {
            "answer": answer_str,
            "confidence": confidence,
            "observation": observation,
            "raw_response": response,
        }

    # ================================================================
    # Step 1: Question Decomposition
    # ================================================================

    def decompose_query(self, question_text, options):
        opt_str = "\n".join(options)
        prompt = f"""Analyze the following video question and choices.
Extract 3 to 5 highly specific keywords (cues) that act as search triggers for finding the relevant video segment.
Focus on: Prominent Objects, Specific Persons, Specific Actions, State Changes, Numbers/Statistics.

[Question]
{question_text}

[Choices]
{opt_str}

Output ONLY valid JSON:
{{
    "cues": ["keyword1", "keyword2", "keyword3"],
    "target_action": "Brief description of what to look for in the video"
}}"""
        result = self._llm_text_reasoning(prompt, max_tokens=150)
        cues = result.get("cues", [])
        if not cues:
            words = re.findall(r'\b[A-Z][a-z]+\b|\b\w{4,}\b', question_text)
            cues = list(set(words))[:5]
        return {
            "cues": cues,
            "target_action": result.get("target_action", ""),
        }

    # ================================================================
    # Step 2: Flatten all leaves from memory tree
    # ================================================================

    def flatten_all_leaves(self, tree):
        """Extract every leaf clip from the memory tree.

        Walks Level_1 nodes (which directly contain leaf clips as children).
        Falls back to recursive extraction if Level_1 is missing.
        Deduplicates by (start_time, end_time), sorts by start_time.
        """
        seen = set()
        leaves = []

        def _add_leaf(leaf, parent_summary=""):
            if "start_time" not in leaf:
                return
            leaf_id = (float(leaf["start_time"]), float(leaf["end_time"]))
            if leaf_id in seen:
                return
            seen.add(leaf_id)
            leaves.append({
                "leaf": leaf,
                "leaf_id": leaf_id,
                "parent_summary": parent_summary,
            })

        def _extract_recursive(node, parent_summary=""):
            children = node.get("children", [])
            if not children:
                _add_leaf(node, parent_summary)
                return
            node_summary = node.get("summary", "")
            if isinstance(children[0], dict) and "level" in children[0]:
                for child in children:
                    _extract_recursive(child, node_summary)
            else:
                for child in children:
                    _add_leaf(child, node_summary)

        # Primary: walk Level_1 directly (all leaves are children of L1 nodes)
        if "Level_1" in tree:
            for node in tree["Level_1"]:
                parent_summary = node.get("summary", "")
                for child in node.get("children", []):
                    _add_leaf(child, parent_summary)

        # Safety: if Level_1 missed some, also walk higher levels recursively
        for lvl_key in sorted(tree.keys()):
            if lvl_key == "Level_1":
                continue
            for node in tree[lvl_key]:
                _extract_recursive(node)

        leaves.sort(key=lambda x: x["leaf_id"][0])
        return leaves

    # ================================================================
    # Step 3: Rule-based keyword filter (no LLM cost)
    # ================================================================

    def rule_based_filter(self, all_leaves, cues):
        """Match cues against leaf key_elements, summary, caption.

        Returns (marked_leaves, unmarked_leaves).
        marked_leaves are enriched with match_count and matched_cues.
        """
        cues_lower = [c.lower().strip() for c in cues if c.strip()]
        marked = []
        unmarked = []

        for entry in all_leaves:
            leaf = entry["leaf"]

            # Build searchable text fragments
            searchable = []
            searchable.append(leaf.get("summary", "").lower())
            searchable.append(leaf.get("caption", "").lower())
            searchable.append(entry.get("parent_summary", "").lower())

            ke = leaf.get("key_elements", {})
            for category in ["persons", "actions", "objects", "locations", "text_ocr", "attributes"]:
                for val in ke.get(category, []):
                    searchable.append(str(val).lower())

            # Check each cue
            matched_cues = []
            for cue in cues_lower:
                for text in searchable:
                    if cue in text:
                        matched_cues.append(cue)
                        break

            if matched_cues:
                enriched = dict(entry)
                enriched["match_count"] = len(matched_cues)
                enriched["matched_cues"] = matched_cues
                marked.append(enriched)
            else:
                unmarked.append(entry)

        # Sort: most cue matches first, then by start_time
        marked.sort(key=lambda x: (-x["match_count"], x["leaf_id"][0]))

        return marked, unmarked

    # ================================================================
    # Step 4: LLM leaf selection (budgeted)
    # ================================================================

    def _format_leaf_for_selection(self, idx, leaf_entry):
        """Format a single leaf for the LLM selection prompt (compact 1-line)."""
        leaf = leaf_entry["leaf"]
        st, et = leaf.get("start_time", 0), leaf.get("end_time", 0)
        summary = leaf.get("summary", "N/A")

        matched_cues = leaf_entry.get("matched_cues", [])
        match_str = f" | Matched: [{', '.join(matched_cues)}]" if matched_cues else ""

        ke = leaf.get("key_elements", {})
        ke_highlights = []
        for cat in ["persons", "text_ocr", "actions", "objects"]:
            items = ke.get(cat, [])
            if items:
                ke_highlights.extend(items[:3])
        ke_str = ", ".join(ke_highlights[:8]) if ke_highlights else ""

        return f"ID {idx} | {st:.0f}s-{et:.0f}s | {summary[:200]} | {ke_str}{match_str}"

    def llm_select_leaves(self, candidate_leaves, cues, question, options,
                          hop_history, budget=10):
        """LLM picks `budget` most promising leaves from candidates."""
        if len(candidate_leaves) <= budget:
            return list(range(len(candidate_leaves)))

        cues_str = ", ".join(cues)
        opt_text = "\n".join(options)

        leaf_descriptions = "\n".join(
            self._format_leaf_for_selection(i, entry)
            for i, entry in enumerate(candidate_leaves)
        )

        # Build history context
        history_text = ""
        if hop_history:
            for h in hop_history:
                if h.get("type") == "frame_inference" and h.get("observation"):
                    history_text += f"- Hop {h['hop']}: {h['observation'][:150]}\n"
                elif h.get("type") == "leaf_solvability" and h.get("reasoning"):
                    history_text += f"- Previous check: {h['reasoning'][:150]}\n"

        history_section = f"\n[Previous Observations]\n{history_text}" if history_text else ""

        prompt = f"""You are selecting video segments to examine for answering a question.
Pick the {budget} most promising segments based on how well they match the search cues and question.

[Question]: {question}
[Search Cues]: {cues_str}
{history_section}

[Candidate Segments]
{leaf_descriptions}

Select exactly {budget} segment IDs. Prioritize:
1. Segments with more matched cues
2. Segments whose summary/key elements directly relate to the question
3. Segments covering different time ranges (temporal diversity)

Output ONLY valid JSON:
{{"selected_ids": [0, 1, 2], "reasoning": "brief explanation"}}"""

        result = self._llm_text_reasoning(prompt, max_tokens=200)
        selected_ids = result.get("selected_ids", [])
        valid_ids = [i for i in selected_ids if isinstance(i, int) and 0 <= i < len(candidate_leaves)]

        if not valid_ids:
            # Fallback: top-N by match_count (already sorted)
            valid_ids = list(range(min(budget, len(candidate_leaves))))

        return valid_ids[:budget]

    # ================================================================
    # Leaf batch context & intervals
    # ================================================================

    def _collect_leaf_batch_context(self, leaf_entries, max_chars=None):
        """Build detailed text context from a flat list of leaf entries."""
        if max_chars is None:
            max_chars = self.max_memory_chars

        entries = []
        for entry in leaf_entries:
            leaf = entry["leaf"]
            st = leaf.get("start_time", 0)
            et = leaf.get("end_time", 0)

            summary = leaf.get("summary", "")
            caption = leaf.get("caption", "")
            ke = leaf.get("key_elements", {})

            summary_line = f"[{st:.0f}s-{et:.0f}s] {summary}"
            lines = [summary_line]
            if caption:
                lines.append(f"  Caption: {caption}")

            for cat, label in [("text_ocr", "On-screen text"), ("persons", "Persons"),
                               ("actions", "Actions"), ("objects", "Objects"),
                               ("locations", "Locations"), ("attributes", "Attributes")]:
                items = ke.get(cat, [])
                if items:
                    lines.append(f"  {label}: {', '.join(str(v) for v in items)}")

            matched = entry.get("matched_cues", [])
            if matched:
                lines.append(f"  [Matched cues: {', '.join(matched)}]")

            entries.append((st, "\n".join(lines), summary_line))

        entries.sort(key=lambda x: x[0])

        full_context = "\n\n".join(e[1] for e in entries)
        if len(full_context) <= max_chars:
            return full_context

        summary_context = "\n".join(e[2] for e in entries)
        if len(summary_context) <= max_chars:
            return summary_context

        return summary_context[:max_chars] + "\n... [truncated]"

    def _get_leaf_intervals_from_entries(self, leaf_entries):
        """Extract time intervals from a flat list of leaf entries."""
        intervals = []
        for entry in leaf_entries:
            leaf = entry["leaf"]
            if "start_time" in leaf:
                intervals.append((float(leaf["start_time"]), float(leaf["end_time"])))
        intervals.sort()
        return intervals

    # ================================================================
    # Solvability Check
    # ================================================================

    def check_solvability(self, leaf_context, question, options, hop_history=None,
                          batch_num=None, total_batches=None):
        opt_text = "\n".join(options)

        history_section = ""
        if hop_history:
            history_lines = []
            for h in hop_history:
                htype = h.get("type", "unknown")
                if htype == "leaf_solvability" and h.get("reasoning"):
                    history_lines.append(f"- Previous memory check: solvable={h['solvable']}, reasoning: {h['reasoning'][:200]}")
                elif htype == "frame_inference" and h.get("observation"):
                    history_lines.append(f"- Visual observation: {h['observation'][:200]} (answer={h.get('answer')}, confidence={h.get('confidence')})")
            if history_lines:
                history_section = "\n### Previous Reasoning History\n" + "\n".join(history_lines) + "\n"

        batch_note = ""
        if batch_num is not None and total_batches is not None:
            batch_note = f" (Batch {batch_num}/{total_batches} — you are seeing a subset of the video)"

        prompt = f"""### Role
You are a strict "Evidence-Based QA Agent". Your task is to answer the user's question based ONLY on the provided memory context.

### Instructions
1. Read the question and the memory context carefully.
2. Determine if the memory context contains sufficient specific information to answer the question completely and accurately.
   - Do NOT use outside knowledge.
   - Do NOT guess or hallucinate details not present in the context.
   - If the context lacks specific details the question asks for (e.g., exact values, player numbers, on-screen statistics), you MUST mark it as unsolvable.
   - Pay special attention to "On-screen text" entries — they contain text visible in the video (scores, player names, statistics).
3. Return a JSON object.
{history_section}
### Memory Context{batch_note}
{leaf_context}

### Question
{question}

### Options
{opt_text}

### Output Format (JSON only)
{{
    "reasoning": "Step-by-step reasoning on whether the context supports the answer.",
    "solvable": true/false,
    "needs_depth": true/false,
    "answer": "A or B or C or D (letter only). If solvable is false, set to null."
}}
- "needs_depth": true if you found partial evidence in some segments and need more visual detail to confirm."""

        result = self._llm_text_reasoning(prompt, max_tokens=512)
        solvable = result.get("solvable", False)
        answer = result.get("answer", None)
        reasoning = result.get("reasoning", "")
        needs_depth = result.get("needs_depth", False)

        if answer and isinstance(answer, str):
            m = re.search(r'[ABCD]', answer.upper())
            answer = m.group(0) if m else None

        return {
            "solvable": bool(solvable),
            "answer": answer,
            "reasoning": reasoning,
            "needs_depth": bool(needs_depth),
        }

    # ================================================================
    # Frame Loading (reused from v2)
    # ================================================================

    def load_targeted_frames(self, video_path, intervals, max_frames=None):
        if max_frames is None:
            max_frames = self.max_frames

        if not intervals or not os.path.exists(video_path):
            return None, []

        import numpy as np
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        vid_frames = len(vr)

        intervals = sorted(intervals)
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + 1.0:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        durations = [max(0.1, e - s) for s, e in merged]
        total_dur = sum(durations)
        n_per = [max(1, int(round(max_frames * (d / total_dur)))) for d in durations]

        while sum(n_per) > max_frames:
            idx = n_per.index(max(n_per))
            n_per[idx] -= 1
        while sum(n_per) < max_frames:
            idx = durations.index(max(durations))
            n_per[idx] += 1

        all_idxs = []
        for (s, e), n in zip(merged, n_per):
            if n <= 0:
                continue
            start_f = max(0, int(s * fps))
            end_f = min(vid_frames - 1, int(e * fps))
            if end_f <= start_f:
                end_f = start_f + 1
            idxs = np.linspace(start_f, end_f, n).astype(int)
            idxs = np.clip(idxs, 0, vid_frames - 1)
            all_idxs.append(idxs)

        if not all_idxs:
            return None, []

        frame_idxs = np.concatenate(all_idxs)
        _, unique_mask = np.unique(frame_idxs, return_index=True)
        frame_idxs = frame_idxs[np.sort(unique_mask)]

        frames = vr.get_batch(frame_idxs).asnumpy()
        frame_seconds = (frame_idxs / fps).tolist()

        return frames, frame_seconds

    # ================================================================
    # Time Reference Coverage Analysis
    # ================================================================

    def _compute_coverage_from_intervals(self, intervals, time_ref_str):
        """Core coverage computation from a list of (start, end) intervals."""
        time_ref = parse_time_reference(time_ref_str)
        if time_ref is None:
            return {"time_ref_seconds": None, "coverage_ratio": None, "precision": None, "hit": None}

        ref_start, ref_end = time_ref
        if ref_end < ref_start:
            ref_end = ref_start

        if not intervals:
            return {
                "time_ref_seconds": [ref_start, ref_end],
                "active_segments": [],
                "coverage_ratio": 0.0,
                "precision": 0.0,
                "hit": False,
            }

        ref_duration = max(ref_end - ref_start, 1.0)
        overlap_duration = 0.0
        hit = False

        for seg_start, seg_end in intervals:
            overlap_start = max(ref_start, seg_start)
            overlap_end = min(ref_end, seg_end)
            if overlap_start < overlap_end:
                overlap_duration += (overlap_end - overlap_start)
                hit = True
            elif ref_start == ref_end and seg_start <= ref_start <= seg_end:
                hit = True
                overlap_duration = 1.0

        coverage_ratio = min(overlap_duration / ref_duration, 1.0)

        total_active_duration = sum(max(0, e - s) for s, e in intervals)
        if total_active_duration > 0:
            precision = min(overlap_duration / total_active_duration, 1.0)
        else:
            precision = 0.0

        return {
            "time_ref_seconds": [ref_start, ref_end],
            "active_segments": intervals[:20],
            "coverage_ratio": round(coverage_ratio, 4),
            "precision": round(precision, 4),
            "hit": hit,
        }

    def compute_coverage_from_entries(self, leaf_entries, time_ref_str):
        intervals = self._get_leaf_intervals_from_entries(leaf_entries)
        return self._compute_coverage_from_intervals(intervals, time_ref_str)

    # ================================================================
    # Main solve loop
    # ================================================================

    def solve(self, video_id, memory_dict, qa, tree):
        uid = qa.get("uid", "unknown")
        question_text, options = parse_lvbench_question(qa)
        gt_answer = qa.get("answer", "")
        time_ref = qa.get("time_reference", "")

        print(f"\n{'='*60}")
        print(f"[Solving UID: {uid} | GT: {gt_answer} | Time Ref: {time_ref}]")
        print(f"Q: {question_text}")

        hop_history = []
        best_answer = None
        best_confidence = "low"
        used_visual = False

        # ---- Step 1: Question Decomposition ----
        print("\n  [Step 1] Question Decomposition")
        decomp = self.decompose_query(question_text, options)
        cues = decomp["cues"]
        print(f"  Cues: {cues}")
        print(f"  Target: {decomp['target_action']}")

        hop_history.append({
            "hop": 1,
            "type": "decomposition",
            "cues": cues,
            "target_action": decomp["target_action"],
        })

        # ---- Step 2: Flatten all leaves + inject raw captions ----
        print("\n  [Step 2] Flatten leaves")
        all_leaves = self.flatten_all_leaves(tree)
        caption_map = load_raw_captions(self.caption_dir, video_id)
        caption_injected = 0
        for entry in all_leaves:
            leaf = entry["leaf"]
            st = float(leaf.get("start_time", -1))
            if st in caption_map:
                leaf["caption"] = caption_map[st]
                caption_injected += 1
        print(f"  Total leaves: {len(all_leaves)}, raw captions injected: {caption_injected}/{len(caption_map)}")

        # ---- Step 2.5: Time-reference pre-filter ----
        time_range = parse_time_reference(time_ref)
        if time_range:
            ref_start, ref_end = time_range
            time_filtered = [e for e in all_leaves
                             if float(e["leaf"]["end_time"]) > ref_start
                             and float(e["leaf"]["start_time"]) < ref_end]
            if time_filtered:
                print(f"  [Time filter] {time_ref} → {ref_start:.0f}s-{ref_end:.0f}s → {len(time_filtered)}/{len(all_leaves)} leaves")
                all_leaves = time_filtered
            else:
                print(f"  [Time filter] No leaves in range {ref_start:.0f}s-{ref_end:.0f}s, using all")
        else:
            print(f"  [Time filter] No time reference, using all {len(all_leaves)} leaves")

        # ---- Step 3: Rule-based keyword filter ----
        print("\n  [Step 3] Rule-based keyword filter")
        marked_leaves, unmarked_leaves = self.rule_based_filter(all_leaves, cues)
        print(f"  Marked: {len(marked_leaves)}, Unmarked: {len(unmarked_leaves)}")
        if marked_leaves:
            top3 = marked_leaves[:3]
            for e in top3:
                print(f"    [{e['leaf_id'][0]:.0f}s-{e['leaf_id'][1]:.0f}s] matches={e['match_count']} cues={e['matched_cues']}")

        # Coverage after rule filter (marked leaves only)
        filter_coverage = self.compute_coverage_from_entries(marked_leaves, time_ref) if time_ref else {}

        hop_history.append({
            "hop": 2,
            "type": "rule_filter",
            "total_leaves": len(all_leaves),
            "marked_count": len(marked_leaves),
            "unmarked_count": len(unmarked_leaves),
            "time_ref_coverage": filter_coverage,
        })
        if filter_coverage:
            print(f"  Filter coverage: hit={filter_coverage.get('hit')}, "
                  f"coverage={filter_coverage.get('coverage_ratio')}, precision={filter_coverage.get('precision')}")

        # ---- State tracking ----
        examined_ids = set()
        all_examined_entries = []
        expansion_used = False
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")

        # Estimate total batches for context
        total_marked = len(marked_leaves)
        est_batches = math.ceil(total_marked / self.leaf_budget) if total_marked > 0 else 1

        # ---- Hop Loop ----
        for hop_num in range(1, self.max_hops + 1):
            print(f"\n  [Hop {hop_num}]")

            # Determine candidate pool
            candidates = [e for e in marked_leaves if e["leaf_id"] not in examined_ids]
            source = "marked"

            if not candidates:
                if not expansion_used and unmarked_leaves:
                    candidates = [e for e in unmarked_leaves if e["leaf_id"] not in examined_ids]
                    source = "unmarked_expansion"
                    expansion_used = True
                    print(f"  Marked exhausted. Expanding to {len(candidates)} unmarked leaves.")

                if not candidates:
                    print(f"  No more candidates. Stopping.")
                    break

            # LLM selects best N from candidates
            if len(candidates) <= self.leaf_budget:
                selected_ids = list(range(len(candidates)))
            else:
                selected_ids = self.llm_select_leaves(
                    candidates, cues, question_text, options,
                    hop_history, budget=self.leaf_budget
                )

            current_batch = [candidates[i] for i in selected_ids]
            batch_leaf_ids = {e["leaf_id"] for e in current_batch}
            examined_ids.update(batch_leaf_ids)
            all_examined_entries.extend(current_batch)

            batch_num = hop_num
            print(f"  Selected {len(current_batch)} leaves from {len(candidates)} {source} candidates")
            for e in current_batch[:3]:
                print(f"    [{e['leaf_id'][0]:.0f}s-{e['leaf_id'][1]:.0f}s] {e['leaf'].get('summary', '')[:80]}")

            # ---- Phase A: Solvability check ----
            print(f"  [Hop {hop_num}, Phase A] Solvability Check")
            batch_context = self._collect_leaf_batch_context(current_batch)
            print(f"  Batch context: {len(batch_context)} chars")

            solv = self.check_solvability(
                batch_context, question_text, options, hop_history,
                batch_num=batch_num, total_batches=est_batches
            )
            print(f"  Solvable: {solv['solvable']} | Answer: {solv['answer']} | Needs depth: {solv['needs_depth']}")
            print(f"  Reasoning: {solv['reasoning'][:200]}")

            hop_coverage = self.compute_coverage_from_entries(all_examined_entries, time_ref) if time_ref else {}

            hop_history.append({
                "hop": len(hop_history) + 1,
                "type": "leaf_solvability",
                "source": source,
                "batch_size": len(current_batch),
                "batch_time_ranges": [(e["leaf_id"][0], e["leaf_id"][1]) for e in current_batch],
                "solvable": solv["solvable"],
                "answer": solv["answer"],
                "reasoning": solv["reasoning"],
                "needs_depth": solv["needs_depth"],
                "time_ref_coverage": hop_coverage,
            })

            if solv["solvable"] and solv["answer"]:
                best_answer = solv["answer"]
                best_confidence = "high"
                print(f"  >> Solved from memory! Answer: {best_answer}")
                break

            # ---- Phase B: Frame Loading + VLM ----
            if self.max_frames > 0:
                if self.dry_run:
                    print(f"  [Hop {hop_num}, Phase B] [DRY_RUN] Would load frames from {video_path}")
                elif not os.path.exists(video_path):
                    print(f"  >> Video file not found: {video_path}")
                else:
                    # Decide width vs depth mode
                    use_depth = solv.get("needs_depth", False)
                    frame_entries = current_batch

                    if use_depth and len(current_batch) > self.depth_budget:
                        print(f"  [Hop {hop_num}, Phase B] DEPTH mode — selecting {self.depth_budget} from {len(current_batch)}")
                        depth_ids = self.llm_select_leaves(
                            current_batch, cues, question_text, options,
                            hop_history, budget=self.depth_budget
                        )
                        frame_entries = [current_batch[i] for i in depth_ids]
                    elif use_depth:
                        print(f"  [Hop {hop_num}, Phase B] DEPTH mode — using all {len(current_batch)} (< depth_budget)")
                    else:
                        print(f"  [Hop {hop_num}, Phase B] WIDTH mode — {len(current_batch)} leaves")

                    intervals = self._get_leaf_intervals_from_entries(frame_entries)

                    if intervals:
                        frames_np, frame_secs = self.load_targeted_frames(video_path, intervals)

                        if frames_np is not None:
                            print(f"  Loaded {len(frame_secs)} frames for {len(frame_entries)} leaves")
                            frame_context = self._collect_leaf_batch_context(frame_entries)
                            vlm_result = self._vlm_inference(
                                frames_np, frame_context, question_text, options, hop_history
                            )
                            used_visual = True

                            hop_history.append({
                                "hop": len(hop_history) + 1,
                                "type": "frame_inference",
                                "mode": "depth" if use_depth else "width",
                                "n_frames": len(frame_secs),
                                "n_leaves": len(frame_entries),
                                "answer": vlm_result["answer"],
                                "confidence": vlm_result["confidence"],
                                "observation": vlm_result["observation"],
                            })

                            ans = vlm_result["answer"]
                            if ans:
                                m = re.search(r'[ABCD]', ans.upper())
                                ans = m.group(0) if m else None

                            if vlm_result["confidence"] == "high" and ans:
                                best_answer = ans
                                best_confidence = "high"
                                print(f"  >> High confidence from VLM! Answer: {best_answer}")
                                break
                            elif ans:
                                best_answer = ans
                                best_confidence = vlm_result["confidence"]

                            del frames_np
                            gc.collect()
            else:
                print(f"  [Hop {hop_num}, Phase B] Skipped (max_frames=0)")

        # ---- Forced answer fallback ----
        pred = best_answer
        if not pred:
            for h in reversed(hop_history):
                if h.get("answer"):
                    m = re.search(r'[ABCD]', str(h["answer"]).upper())
                    if m:
                        pred = m.group(0)
                        break

        if not pred and not self.dry_run:
            print("\n  [Fallback] Forcing answer from all examined leaves...")
            all_context = self._collect_leaf_batch_context(all_examined_entries)
            opt_text = "\n".join(options)
            force_prompt = f"""You MUST answer the following multiple choice question. Pick the best option even if uncertain.

[Memory Context]
{all_context}

[Question]
{question_text}

[Options]
{opt_text}

Output ONLY valid JSON: {{"answer": "A"}}"""
            force_result = self._llm_text_reasoning(force_prompt, max_tokens=20)
            raw = force_result.get("answer", "") or str(force_result)
            m = re.search(r'[ABCD]', raw.upper())
            if m:
                pred = m.group(0)
                print(f"  Forced answer: {pred}")

        is_correct = (pred == gt_answer) if gt_answer else None

        time_coverage = self.compute_coverage_from_entries(all_examined_entries, time_ref)

        print(f"\n  [Final] Pred: {pred} | GT: {gt_answer} | Correct: {is_correct}")
        print(f"  Time Ref Coverage: {time_coverage}")
        print(f"  Examined: {len(examined_ids)} leaves out of {len(all_leaves)} total")

        return {
            "uid": uid,
            "video_id": video_id,
            "question": question_text,
            "options": options,
            "pred": pred,
            "answer": gt_answer,
            "correct": is_correct,
            "confidence": best_confidence,
            "hop_history": hop_history,
            "total_hops": len(hop_history),
            "used_visual": used_visual,
            "time_reference": time_ref,
            "time_ref_coverage": time_coverage,
            "question_type": qa.get("question_type", []),
            "total_leaves": len(all_leaves),
            "marked_leaves": len(marked_leaves),
            "examined_leaves": len(examined_ids),
        }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="LVBench Agentic Solver v3")
    parser.add_argument("--memory_dir", type=str, default=MEMORY_DIR)
    parser.add_argument("--question_path", type=str, default=QUESTION_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--video_root", type=str, default=VIDEO_ROOT)
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--max_memory_chars", type=int, default=120000)
    parser.add_argument("--image_token_size", type=int, default=256,
                        help="Number of image tokens per frame (pixels = token_size * 32 * 32)")
    parser.add_argument("--leaf_budget", type=int, default=10,
                        help="Number of leaves per hop (width mode)")
    parser.add_argument("--depth_budget", type=int, default=5,
                        help="Number of leaves for depth mode (more frames per leaf)")
    parser.add_argument("--caption_dir", type=str, default=CAPTION_DIR,
                        help="Directory with raw caption JSON files per video")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    by_qid_dir = os.path.join(args.output_path, "by_qid")
    os.makedirs(by_qid_dir, exist_ok=True)

    # 1) Load memory & questions
    mem_index = load_memory_files(args.memory_dir)
    qa_index = load_lvbench_questions(args.question_path)

    target_videos = set(mem_index.keys()) & set(qa_index.keys())
    if not target_videos:
        print("No matching videos. Exiting.")
        return

    total_questions = sum(len(qa_index[vid]) for vid in target_videos)
    print(f"Found {len(target_videos)} matching videos with {total_questions} questions.")

    # 2) Load model
    model, processor = None, None
    if not args.dry_run:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print("Loading model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        model.eval()
        print("Model loaded.")

    solver = AgenticSolverV3(
        model=model,
        processor=processor,
        max_hops=args.max_hops,
        max_frames=args.max_frames,
        max_memory_chars=args.max_memory_chars,
        video_root=args.video_root,
        dry_run=args.dry_run,
        image_token_size=args.image_token_size,
        leaf_budget=args.leaf_budget,
        depth_budget=args.depth_budget,
        caption_dir=args.caption_dir,
    )

    # 3) Solve
    results = []
    correct, total = 0, 0
    coverage_hits, coverage_total = 0, 0

    for vid in tqdm(sorted(target_videos), desc="Videos"):
        memory_dict = mem_index[vid]
        tree = memory_dict.get("streaming_memory_tree", {})
        qa_list = qa_index[vid]

        for qa in tqdm(qa_list, desc=f"  QA ({vid})", leave=False):
            rec = solver.solve(vid, memory_dict, qa, tree)
            results.append(rec)

            if rec["answer"]:
                total += 1
                if rec["correct"]:
                    correct += 1

            cov = rec.get("time_ref_coverage", {})
            if cov.get("hit") is not None:
                coverage_total += 1
                if cov["hit"]:
                    coverage_hits += 1

            # Save per-QID
            qid_file = os.path.join(by_qid_dir, f"{safe_filename(rec['uid'])}.json")
            with open(qid_file, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        print(f"\n[Dry run] Would solve {total_questions} questions.")
        return

    # 4) Summary
    acc = correct / total * 100 if total > 0 else 0
    cov_rate = coverage_hits / coverage_total * 100 if coverage_total > 0 else 0

    avg_coverage_ratio = 0.0
    avg_precision = 0.0
    n_cov = 0
    for r in results:
        cov = r.get("time_ref_coverage", {})
        if cov.get("coverage_ratio") is not None:
            avg_coverage_ratio += cov["coverage_ratio"]
            avg_precision += cov["precision"]
            n_cov += 1
    if n_cov > 0:
        avg_coverage_ratio /= n_cov
        avg_precision /= n_cov

    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct ({acc:.2f}%)")
    print(f"Time Ref Hit Rate: {coverage_hits}/{coverage_total} ({cov_rate:.1f}%)")
    print(f"Avg Coverage Ratio: {avg_coverage_ratio:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")

    # Leaf stats
    avg_marked = sum(r.get("marked_leaves", 0) for r in results) / len(results) if results else 0
    avg_examined = sum(r.get("examined_leaves", 0) for r in results) / len(results) if results else 0
    print(f"Avg marked leaves per question: {avg_marked:.1f}")
    print(f"Avg examined leaves per question: {avg_examined:.1f}")

    # Count visual usage & solvable
    visual_count = sum(1 for r in results if r.get("used_visual"))
    solvable_count = sum(
        1 for r in results
        if any(h.get("solvable") for h in r.get("hop_history", []) if h.get("type") == "leaf_solvability")
    )
    print(f"Solved from memory only: {solvable_count}/{len(results)}")
    print(f"Used visual frames: {visual_count}/{len(results)}")

    # Per-hop time ref coverage stats
    hop_type_stats = {}
    for r in results:
        for h in r.get("hop_history", []):
            cov = h.get("time_ref_coverage", {})
            if not cov or cov.get("hit") is None:
                continue
            htype = h.get("type", "unknown")
            if htype not in hop_type_stats:
                hop_type_stats[htype] = {"hits": 0, "total": 0, "sum_coverage": 0.0, "sum_precision": 0.0, "count": 0}
            s = hop_type_stats[htype]
            s["total"] += 1
            if cov["hit"]:
                s["hits"] += 1
            if cov.get("coverage_ratio") is not None:
                s["sum_coverage"] += cov["coverage_ratio"]
                s["sum_precision"] += cov["precision"]
                s["count"] += 1

    if hop_type_stats:
        print(f"\n--- Per-Hop Time Ref Coverage ---")
        for htype, s in sorted(hop_type_stats.items()):
            hit_pct = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
            avg_cov = s["sum_coverage"] / s["count"] if s["count"] > 0 else 0
            avg_prec = s["sum_precision"] / s["count"] if s["count"] > 0 else 0
            print(f"  {htype}: hit={s['hits']}/{s['total']} ({hit_pct:.1f}%), "
                  f"avg_coverage={avg_cov:.4f}, avg_precision={avg_prec:.4f}")

    # Per-hop stats for JSON
    hop_type_summary = {}
    for htype, s in hop_type_stats.items():
        hop_type_summary[htype] = {
            "hit_rate": s["hits"] / s["total"] * 100 if s["total"] > 0 else 0,
            "hits": s["hits"],
            "total": s["total"],
            "avg_coverage": s["sum_coverage"] / s["count"] if s["count"] > 0 else 0,
            "avg_precision": s["sum_precision"] / s["count"] if s["count"] > 0 else 0,
        }

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "time_ref_hit_rate": cov_rate,
        "avg_coverage_ratio": avg_coverage_ratio,
        "avg_precision": avg_precision,
        "avg_marked_leaves": avg_marked,
        "avg_examined_leaves": avg_examined,
        "per_hop_coverage": hop_type_summary,
        "solvable_from_memory": solvable_count,
        "used_visual": visual_count,
        "memory_dir": args.memory_dir,
        "question_path": args.question_path,
        "video_root": args.video_root,
        "max_hops": args.max_hops,
        "max_frames": args.max_frames,
        "leaf_budget": args.leaf_budget,
        "depth_budget": args.depth_budget,
        "predictions": results,
    }
    summary_file = os.path.join(args.output_path, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print(f"Per-QID results saved to: {by_qid_dir}")


if __name__ == "__main__":
    # try:
    #     import debugpy
    #     if not debugpy.is_client_connected():
    #         debugpy.listen(("0.0.0.0", 1234))
    #         print("Waiting for debugger to attach...")
    #         debugpy.wait_for_client()
    #     debugpy.configure(subProcess=True)
    # except ImportError:
    #     pass
    main()
