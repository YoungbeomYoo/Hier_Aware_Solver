import os
from PIL import Image
import os.path as osp
import time
import json
import gc
import numpy as np

from models.base_model import BaseVLM

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch

import sys
import copy
import math

import re

from decord import VideoReader, cpu


class Model(BaseVLM):
    def __init__(self, args, config, run_output_dir):
        super().__init__(args, config, run_output_dir, backend="decord")
        self.gen_kwargs = {
            "do_sample": False,
            "top_p": None,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 2,
            "top_k": None,
            "temperature": None,
        }

        pretrained = self.config["model_cache"]
        device_map = "auto"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(pretrained)

        self.model.eval()

    # ================================================================
    # Utility methods (same as v3)
    # ================================================================

    @staticmethod
    def _time_to_secs(t):
        """MM:SS 또는 HH:MM:SS 문자열을 float 초로 변환"""
        try:
            parts = str(t).split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(t)
        except Exception:
            return 0.0

    @staticmethod
    def _secs_to_time_str(s):
        """float 초를 HH:MM:SS 또는 MM:SS 문자열로 변환"""
        s = float(s)
        h = int(s) // 3600
        m = (int(s) % 3600) // 60
        sec = s - h * 3600 - m * 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{sec:05.2f}"
        return f"{m:02d}:{sec:04.1f}"

    def _extract_time_ranges(self, text):
        """텍스트에서 타임스탬프를 추출하여 초 단위 (start, end) 튜플 리스트로 반환"""
        ranges = []

        # 우선순위 1: <TIME HH:MM:SS.ms video N> to/and <TIME HH:MM:SS.ms video N> 형식
        tag_pattern = r'<TIME\s+([\d:\.]+)\s+[^>]+>\s*(?:to|and)\s*<TIME\s+([\d:\.]+)\s+[^>]+>'
        for start_str, end_str in re.findall(tag_pattern, text):
            ranges.append((self._time_to_secs(start_str), self._time_to_secs(end_str)))

        if ranges:
            return ranges

        # 폴백: "HH:MM:SS - HH:MM:SS" 또는 "MM:SS ~ MM:SS" 형식
        time_pattern = r'(?:(?:[0-9]{2}:)?(?:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?))'
        range_pattern = fr'({time_pattern})\s*(?:-|~|to)\s*({time_pattern})'
        for start_str, end_str in re.findall(range_pattern, text):
            ranges.append((self._time_to_secs(start_str), self._time_to_secs(end_str)))

        return ranges

    def _extract_json_from_response(self, response):
        """응답에서 JSON 추출. depth-tracking + 잘린 JSON 복구."""
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
        return result

    def _llm_text_reasoning(self, prompt_text, max_tokens=256):
        """프레임 없이 Qwen3-VL을 텍스트 LLM으로 사용"""
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
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

        del inputs, output_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        print(f"[DEBUG _llm_text_reasoning] raw response: {repr(response[:500])}")

        parsed = self._extract_json_from_response(response)
        if parsed:
            print(f"[DEBUG _llm_text_reasoning] parsed JSON: {parsed}")
            return parsed
        print(f"[DEBUG _llm_text_reasoning] JSON parsing FAILED, returning {{}}")
        return {}

    # ================================================================
    # Memory retrieval methods (same as v3)
    # ================================================================

    def _get_bottom_up_context_all(self, raw_memory_dict, target_ranges):
        """[Track A] 타겟 시간 구간에 해당하는 모든 Substep을 메모리에서 수집"""
        extracted_context = ""
        found_any = False
        context_by_range = {tr: [] for tr in target_ranges}

        for _, mem_content in raw_memory_dict.items():
            if not isinstance(mem_content, dict) or "memory" not in mem_content:
                continue
            memory = mem_content["memory"]
            goal = memory.get("goal", "Unknown Goal")
            extracted_context += f"=== Global Context ===\n[Overall Goal]: {goal}\n\n"

            for step_dict in memory.get("steps", []):
                for _, step_data in step_dict.items():
                    step_summary = step_data.get("step", "")
                    for sub in step_data.get("substeps", []):
                        sub_start, sub_end = self._time_to_secs(sub["start"]), self._time_to_secs(sub["end"])
                        for tr in target_ranges:
                            t_start, t_end = tr
                            if max(t_start, sub_start) < min(t_end, sub_end):
                                context_by_range[tr].append(
                                    f"  [Phase: {step_summary}]\n"
                                    f"  -> {self._secs_to_time_str(sub_start)} ~ {self._secs_to_time_str(sub_end)}: {sub['substep']}"
                                )
                                found_any = True

        if not found_any:
            return ""

        for tr, evidences in context_by_range.items():
            if evidences:
                extracted_context += f"▶ Target Time {self._secs_to_time_str(tr[0])} ~ {self._secs_to_time_str(tr[1])}:\n"
                extracted_context += "\n".join(list(dict.fromkeys(evidences))) + "\n\n"

        return extracted_context

    def _get_bottom_up_hierarchy_path(self, raw_memory_dict, target_ranges):
        """[Track A] 매칭된 substep들의 계층 경로 반환"""
        paths = []
        for _, mem_content in raw_memory_dict.items():
            if not isinstance(mem_content, dict) or "memory" not in mem_content:
                continue
            memory = mem_content["memory"]
            goal = memory.get("goal", "")
            for step_dict in memory.get("steps", []):
                for _, step_data in step_dict.items():
                    step_summary = step_data.get("step", "")
                    for sub in step_data.get("substeps", []):
                        sub_start = self._time_to_secs(sub["start"])
                        sub_end = self._time_to_secs(sub["end"])
                        for tr in target_ranges:
                            if max(tr[0], sub_start) < min(tr[1], sub_end):
                                paths.append({
                                    "goal": goal,
                                    "step": step_summary,
                                    "substep": sub.get("substep", ""),
                                    "time": f"{sub['start']} ~ {sub['end']}",
                                    "matched_range": f"{self._secs_to_time_str(tr[0])} ~ {self._secs_to_time_str(tr[1])}",
                                })
                                break
        return paths

    def decompose_query(self, question_text, choices):
        """STEP 1: 질문 분해 → 검색 Cue 추출"""
        choices_str = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
        prompt = f"""Analyze the following video question and choices.
Extract 3 to 5 highly specific keywords (cues) that act as search triggers.
Focus on 'Prominent Objects', 'Specific Actions', and 'State Changes'.

[Question]
{question_text}

[Choices]
{choices_str}

Output ONLY valid JSON:
{{
    "cues": ["keyword1", "keyword2", "keyword3"],
    "target_action": "Brief description of the action to find"
}}"""
        return self._llm_text_reasoning(prompt, max_tokens=150)

    def hierarchical_search(self, memory_tree_dict, cues):
        """STEP 2: Cue 기반 Top-Down 계층 탐색 (single hop)"""
        if isinstance(memory_tree_dict, str):
            try:
                memory_tree_dict = json.loads(memory_tree_dict) if memory_tree_dict else {}
            except json.JSONDecodeError:
                memory_tree_dict = {}

        level_1_nodes = []
        tree_goal = ""
        for _, mem_content in memory_tree_dict.items():
            if not isinstance(mem_content, dict):
                continue
            memory = mem_content.get("memory", {})
            tree_goal = memory.get("goal", "")
            for step_obj in memory.get("steps", []):
                for _, step_content in step_obj.items():
                    step_name = step_content.get("step", "")
                    for substep in step_content.get("substeps", []):
                        level_1_nodes.append({
                            "time_segments": f"{substep.get('start', '')} ~ {substep.get('end', '')}",
                            "summary": substep.get("substep", ""),
                            "start": substep.get("start", ""),
                            "end": substep.get("end", ""),
                            "step_name": step_name,
                        })

        if not level_1_nodes:
            return json.dumps(memory_tree_dict, ensure_ascii=False), "unknown", {}

        node_summaries = "\n".join([
            f"ID: {i} | Time: {n['time_segments']} | Summary: {n['summary']}"
            for i, n in enumerate(level_1_nodes)
        ])

        prompt = f"""You are a Hierarchical Memory Navigator.
Your goal is to find the most relevant video segment branch based on the search cues.

[Search Cues]
{cues}

[Level 1 Memory Nodes]
{node_summaries}

Select the ID of the ONE node that most likely contains the answer. If multiple, select the best one.
Keep "thought" under 20 words.
Output ONLY valid JSON:
{{
    "thought": "One sentence reason.",
    "selected_id": <int>
}}"""
        decision = self._llm_text_reasoning(prompt, max_tokens=512)
        selected_id = decision.get("selected_id", 0)

        if selected_id < 0 or selected_id >= len(level_1_nodes):
            selected_id = 0

        target_branch = level_1_nodes[selected_id]
        localized_memory = json.dumps(target_branch, ensure_ascii=False, indent=2)
        target_time_segments = target_branch['time_segments']

        hierarchy_path = {
            "goal": tree_goal,
            "step": target_branch.get("step_name", ""),
            "substep": target_branch.get("summary", ""),
            "time": target_time_segments,
            "llm_thought": decision.get("thought", ""),
        }

        return localized_memory, target_time_segments, hierarchy_path

    # ================================================================
    # V4 NEW: Video metadata + Targeted frame loading
    # ================================================================

    def _get_video_metadata(self, q0):
        """temporal_divisor와 비디오 정보를 프레임 로드 없이 계산"""
        vid_ids, start_secs, end_secs, input_keys = self.get_vids_info_from_question(q0)
        temporal_divisor, loaded_secs, total_secs = self.compute_temporal_divisor(
            vid_ids, start_secs, end_secs
        )
        return vid_ids, start_secs, end_secs, input_keys, temporal_divisor, total_secs

    def _resolve_target_intervals(self, route_evidence, temporal_divisor, input_start_secs):
        """
        라우팅 결과를 절대 비디오 시간(초)으로 변환.
        Track A: matched_time_ranges_abs 는 이미 절대 시간 → 그대로 사용
        Track B: selected_segment 는 메모리 상대 시간 → 역변환 필요
          역변환: abs_time = mem_time * temporal_divisor + input_start_secs
        """
        intervals = []
        track = route_evidence.get("track", "")

        if track == "A":
            for pair in route_evidence.get("matched_time_ranges_abs", []):
                intervals.append((float(pair[0]), float(pair[1])))
        elif track == "B":
            segment_str = route_evidence.get("selected_segment", "")
            parsed = self._extract_time_ranges(segment_str)
            for mem_start, mem_end in parsed:
                abs_start = mem_start * temporal_divisor + input_start_secs
                abs_end = mem_end * temporal_divisor + input_start_secs
                intervals.append((abs_start, abs_end))

        return intervals

    def _load_targeted_frames(self, vid_id, intervals, max_frames):
        """
        지정된 절대 시간 구간들에서만 프레임을 로드.
        max_frames 수를 구간 길이에 비례하여 배분.
        """
        if not intervals:
            return None, []

        full_video_fn = self.get_full_video_fn(vid_id)
        vr = VideoReader(full_video_fn, ctx=cpu(0))
        fps = vr.get_avg_fps()
        vid_frames = len(vr)

        # 겹치거나 인접한 구간 병합 (1초 이내 갭)
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + 1.0:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        # 구간 길이에 비례하여 프레임 수 배분
        durations = [max(0.1, e - s) for s, e in merged]
        total_dur = sum(durations)
        n_per_interval = [max(1, int(round(max_frames * (d / total_dur)))) for d in durations]

        # 총합을 max_frames에 맞춤
        while sum(n_per_interval) > max_frames:
            idx = n_per_interval.index(max(n_per_interval))
            n_per_interval[idx] -= 1
        while sum(n_per_interval) < max_frames:
            idx = durations.index(max(durations))
            n_per_interval[idx] += 1

        # 구간별 프레임 인덱스 생성
        all_idxs = []
        for (s, e), n in zip(merged, n_per_interval):
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
        # 중복 제거 (순서 유지)
        _, unique_mask = np.unique(frame_idxs, return_index=True)
        frame_idxs = frame_idxs[np.sort(unique_mask)]

        frames = vr.get_batch(frame_idxs).asnumpy()
        frame_seconds = (frame_idxs / fps).tolist()

        return frames, frame_seconds

    # ================================================================
    # Main orchestrator
    # ================================================================

    def run_video(self, vids_comb_id, questions):
        q0 = next(iter(questions.values()))

        # ==============================================================
        # PHASE 1: Metadata + Memory (프레임 로드 없음)
        # ==============================================================
        vid_ids, start_secs, end_secs, input_keys, temporal_divisor, total_video_seconds = (
            self._get_video_metadata(q0)
        )

        raw_memory_str = self.format_memory(q0, temporal_divisor)
        raw_memory_dict = json.loads(raw_memory_str) if raw_memory_str else {}

        # input_start_seconds 추출 (Track B 역변환에 필요)
        input_start_secs = 0
        for k, v in q0["inputs"].items():
            if "start_time" in v:
                from models.base_model import secs_from_time_str
                input_start_secs = secs_from_time_str(v["start_time"])
                break

        max_targeted_frames = self.config.get("max_targeted_frames", 64)

        results = {}
        for q_id, question_data in questions.items():
            print(f"\n[Solving {q_id}]")
            start = time.time()

            question_text = question_data["question"]
            choices = question_data["choices"]

            # ==============================================================
            # PHASE 2: Track A/B Routing (텍스트 전용, 프레임 불필요)
            # ==============================================================
            full_text_to_check = question_text + " " + " ".join(choices)
            explicit_time_ranges_abs = self._extract_time_ranges(full_text_to_check)
            explicit_time_ranges = [(s / temporal_divisor, e / temporal_divisor) for s, e in explicit_time_ranges_abs]

            localized_memory = ""
            route_evidence = {}

            if explicit_time_ranges:
                # Track A: Bottom-Up Spreading Activation
                print(f"  -> [Track A] Time cues found: {explicit_time_ranges_abs}. Using Direct Indexing.")
                localized_memory = self._get_bottom_up_context_all(raw_memory_dict, explicit_time_ranges)
                hierarchy_paths = self._get_bottom_up_hierarchy_path(raw_memory_dict, explicit_time_ranges)
                route_evidence = {
                    "track": "A",
                    "matched_time_ranges_abs": [[round(s, 3), round(e, 3)] for s, e in explicit_time_ranges_abs],
                    "hierarchy_path": hierarchy_paths,
                }

                if not localized_memory:
                    print("  -> Target time not found in memory, falling back to Track B.")

            if not explicit_time_ranges or not localized_memory:
                # Track B: Top-Down Hierarchical Search
                print("  -> [Track B] No time cues. Using Top-Down Search (single hop).")
                parsed_query = self.decompose_query(question_text, choices)
                cues = parsed_query.get("cues", [])
                localized_memory, selected_segment, hierarchy_path = self.hierarchical_search(raw_memory_dict, cues)
                route_evidence = {
                    "track": "B",
                    "cues": cues,
                    "target_action": parsed_query.get("target_action", ""),
                    "selected_segment": selected_segment,
                    "hierarchy_path": hierarchy_path,
                }

            # ==============================================================
            # PHASE 3: Targeted Frame Loading (V4 핵심)
            # ==============================================================
            target_intervals_abs = self._resolve_target_intervals(
                route_evidence, temporal_divisor, input_start_secs
            )

            frames_np = None
            frame_seconds = []
            vid_id = vid_ids[0]

            if target_intervals_abs:
                frames_np, frame_seconds = self._load_targeted_frames(
                    vid_id, target_intervals_abs, max_targeted_frames
                )
                print(f"  -> [Targeted Loading] {len(frame_seconds)} frames from {len(target_intervals_abs)} interval(s)")

            # Fallback: 타겟 구간을 못 찾은 경우 전체 비디오에서 균등 샘플링
            if frames_np is None:
                print(f"  -> [Fallback] No targeted intervals, uniform sampling from full video.")
                frames_np = self.load_single_video(
                    vid_id, start_secs[0], end_secs[0], temporal_divisor
                )
                frame_seconds = []

            # 타겟 프레임으로 video_contents 구성 (구간 설명 포함)
            video_contents = []
            frame_list = [Image.fromarray(f) for f in frames_np]
            if target_intervals_abs:
                intervals_str = ", ".join([
                    f"{self._secs_to_time_str(s)} ~ {self._secs_to_time_str(e)}"
                    for s, e in target_intervals_abs
                ])
                video_contents.append({"type": "text", "text": (
                    f"The following {len(frame_list)} frames are visual evidence "
                    f"from the memory-retrieved segment ({intervals_str}). "
                    f"Use these frames to verify and answer the question."
                )})
            video_contents.append({"type": "text", "text": f"{input_keys[0]}: "})
            for img in frame_list:
                video_contents.append({"type": "image", "image": img})
            n_frames_loaded = len(frame_list)
            del frames_np, frame_list
            gc.collect()

            # ==============================================================
            # PHASE 4: VLM Inference
            # ==============================================================
            q, _ = self.formulate_question(question=question_data, temporal_divisor=temporal_divisor)
            current_contents = video_contents + [
                {"type": "text", "text": f"{localized_memory}"},
                {"type": "text", "text": f"{q}"},
            ]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.config["prompts"]["sys"]}]},
                {"role": "user", "content": current_contents},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos_vis = process_vision_info(messages, image_patch_size=16)
            inputs = self.processor(
                text=text, images=images, videos=videos_vis, return_tensors="pt", do_resize=False
            ).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **self.gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

            del inputs, output_ids, generated_ids_trimmed, video_contents
            torch.cuda.empty_cache()

            predicted_idx = self.parse_response(response, len(choices))

            stop = time.time()
            total_time = stop - start

            correct = 1.0 if predicted_idx == question_data.get("correct_idx", -1) else 0.0
            routing_label = "Track A (Direct)" if route_evidence.get("track") == "A" else "Track B (Agentic)"

            # 로그 출력
            print(f"  -> [Route] {routing_label}")
            if route_evidence.get("track") == "A":
                for p in route_evidence.get("hierarchy_path", []):
                    print(f"     [Queried range] {p['matched_range']}")
                    print(f"       Goal: {p['goal']}")
                    print(f"         └─ Step: {p['step']}")
                    print(f"              └─ Substep [{p['time']}]: {p['substep']}")
            else:
                hp = route_evidence.get("hierarchy_path", {})
                print(f"     [Cues] {route_evidence.get('cues', [])}")
                print(f"       Goal: {hp.get('goal', '')}")
                print(f"         └─ Step: {hp.get('step', '')}")
                print(f"              └─ Substep [{hp.get('time', '')}]: {hp.get('substep', '')}")
                print(f"     [LLM thought] {hp.get('llm_thought', '')}")
            print(f"  -> [Frames] {n_frames_loaded} loaded (max_targeted_frames={max_targeted_frames})")
            print(f"  -> [VLM Response] {response!r}  |  Predicted Index: {predicted_idx}  |  Correct: {correct}")

            entry = {
                "id": q_id,
                "time": total_time,
                "answer": predicted_idx,
                "routing": routing_label,
                "route_evidence": route_evidence,
                "target_intervals_abs": [[round(s, 3), round(e, 3)] for s, e in target_intervals_abs] if target_intervals_abs else [],
                "n_frames_loaded": n_frames_loaded,
                "localized_memory": localized_memory,
                "vlm_raw_response": response,
                "correct": correct,
            }

            with open(osp.join(self.run_output_dir, f"{q_id}.json"), "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=4)
            results[q_id] = entry

            gc.collect()

        return results

    def run_eval(self):
        results = {}

        for vids_comb_id, questions in self.questions_by_vid.items():
            if self.args.check_input_only:
                self.run_video_check_input(vids_comb_id, questions)
            else:
                vid_results = self.run_video(vids_comb_id, questions)
                for k, v in vid_results.items():
                    results[k] = v

        return results
