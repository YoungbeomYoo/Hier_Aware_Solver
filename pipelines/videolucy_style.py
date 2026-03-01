"""
VideoLucy-style Pipeline — VideoLucy 프롬프트 + 우리 memory tree

Flow (VideoLucy 원본과 동일):
1. Phase 1 (Coarse Answer): Hierarchical summary → 답변 시도 → confident면 종료
2. Phase 2 (Fine Search, max 2회):
   a. Question Type Judge → time filtering
   b. LLM이 time period 1개 선택
   c. 해당 구간 leaf detail 가져와서 coarse + fine으로 재답변
   d. confident면 종료
3. Phase 3 (Forced Answer): must answer
"""

from __future__ import annotations

import re
import ast
import copy

import torch
from pipelines.base import BasePipeline


class VideoLucyStylePipeline(BasePipeline):
    """VideoLucy prompts + our pre-built memory tree."""

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        max_iterations = self.config.get("max_fine_iterations", 2)

        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")
        tree = memory.get("streaming_memory_tree", {})

        # ============================================================
        # Phase 1: Coarse Answer
        # ============================================================
        coarse_memory = self._build_coarse_memory(tree)
        video_duration = (
            coarse_memory[-1]["time_period"][1] - coarse_memory[0]["time_period"][0]
            if coarse_memory else 0
        )

        coarse_prompt = self._answer_with_coarse_memory_prompt(
            coarse_memory, question, options,
        )
        coarse_raw = self._generate_text(coarse_prompt, max_new_tokens=512)
        coarse_answer = self._parse_answer(coarse_raw)

        if coarse_answer and coarse_answer.get("Confidence"):
            pred = self._extract_option(coarse_answer.get("Answer", ""), options)
            correct = self.adapter.check_correct(pred, answer)
            print(f"    [Phase1] Coarse confident: pred={pred} correct={correct}")
            return self._make_result(
                pred, answer, correct, "phase1_coarse",
                coarse_answer=coarse_answer,
            )

        print(f"    [Phase1] Not confident, proceeding to fine search")

        # ============================================================
        # Time-based filtering (VideoLucy 차용)
        # ============================================================
        time_flag = not self._contains_ordinal_number(question)
        filtered_coarse = coarse_memory

        if time_flag:
            type_prompt = self._question_type_judge_prompt(
                coarse_memory, question, options,
            )
            type_raw = self._generate_text(type_prompt, max_new_tokens=512)
            type_judge = self._parse_answer(type_raw)

            if type_judge and type_judge.get("Flag"):
                related_periods = type_judge.get("Time Period", [])
                filtered_coarse = self._filter_coarse_memory(
                    coarse_memory, related_periods,
                )
                print(f"    [TypeJudge] Filtered to {len(filtered_coarse)}/{len(coarse_memory)} periods")
            else:
                print(f"    [TypeJudge] No filtering (Flag=False or parse fail)")
        else:
            print(f"    [TypeJudge] Ordinal detected, skip time filtering")

        # ============================================================
        # Phase 2: Fine Search Loop
        # ============================================================
        fine_history = {
            "time_periods": [],
            "entire_memories": [],
            "divided_memories": [],
        }

        for iteration in range(max_iterations):
            print(f"    [Phase2] Iteration {iteration + 1}/{max_iterations}")

            # Select time period
            time_prompt = self._get_single_related_time_prompt(
                filtered_coarse,
                fine_history["entire_memories"],
                fine_history["divided_memories"],
                question, options,
                excluded_periods=fine_history["time_periods"],
                duration=video_duration,
            )
            time_raw = self._generate_text(time_prompt, max_new_tokens=512)
            time_result = self._parse_answer(time_raw)

            if not time_result or "Time Period" not in time_result:
                print(f"    [Phase2] Time selection failed, skipping")
                continue

            period = time_result["Time Period"]
            if isinstance(period, list) and len(period) == 1:
                period = period[0]

            print(f"    [Phase2] Selected period: {period}")

            # Get fine detail from tree
            entire, divided = self._get_fine_detail(tree, period)

            fine_history["time_periods"].append(list(period))
            fine_history["entire_memories"].append(entire)
            fine_history["divided_memories"].append(divided)

            # Answer with coarse + fine
            fine_prompt = self._answer_with_coarse_and_fine_prompt(
                filtered_coarse,
                fine_history["entire_memories"],
                fine_history["divided_memories"],
                question, options,
                duration=video_duration,
            )
            fine_raw = self._generate_text(fine_prompt, max_new_tokens=512)
            fine_answer = self._parse_answer(fine_raw)

            if fine_answer and fine_answer.get("Confidence"):
                pred = self._extract_option(fine_answer.get("Answer", ""), options)
                correct = self.adapter.check_correct(pred, answer)
                print(f"    [Phase2] Fine confident (iter {iteration+1}): "
                      f"pred={pred} correct={correct}")
                return self._make_result(
                    pred, answer, correct,
                    f"phase2_fine_iter{iteration + 1}",
                    coarse_answer=coarse_answer,
                    fine_answer=fine_answer,
                    fine_iterations=iteration + 1,
                    fine_periods=fine_history["time_periods"],
                )

        # ============================================================
        # Phase 3: Forced Answer
        # ============================================================
        print(f"    [Phase3] Forced answer")

        forced_prompt = self._must_answer_prompt(
            coarse_memory,  # full coarse (not filtered)
            fine_history["entire_memories"],
            fine_history["divided_memories"],
            question, options,
            duration=video_duration,
        )
        forced_raw = self._generate_text(forced_prompt, max_new_tokens=512)
        forced_answer = self._parse_answer(forced_raw)

        if forced_answer:
            pred = self._extract_option(
                forced_answer.get("Answer", ""), options,
            )
        else:
            pred = "A"  # ultimate fallback

        correct = self.adapter.check_correct(pred, answer)
        print(f"    [Phase3] Forced: pred={pred} correct={correct}")

        return self._make_result(
            pred, answer, correct, "phase3_forced",
            coarse_answer=coarse_answer,
            fine_answer=forced_answer,
            fine_iterations=max_iterations,
            fine_periods=fine_history["time_periods"],
        )

    # ==================== Memory Construction ====================

    def _build_coarse_memory(self, tree: dict) -> list:
        """Hierarchical overview: Level_N~Level_1 모든 summary를 VideoLucy coarse format으로."""
        entries = []

        level_names = sorted(
            [k for k in tree.keys() if k.startswith("Level_")],
            key=lambda x: int(x.split("_")[1]),
            reverse=True,
        )

        for level_name in level_names:
            for node in tree.get(level_name, []):
                summary = node.get("summary", "")
                if not summary:
                    continue

                segs = node.get("time_segments", [])
                if not segs:
                    continue

                flat_times = []
                for s in segs:
                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                        flat_times.extend([float(s[0]), float(s[1])])
                if not flat_times:
                    continue

                start = min(flat_times)
                end = max(flat_times)

                entries.append({
                    "time_period": [start, end],
                    "general_memory": summary,
                    "_level": level_name,
                })

        entries.sort(key=lambda x: (x["time_period"][0], -x["time_period"][1]))
        print(f"    [Coarse] Built {len(entries)} entries from tree")
        return entries

    def _get_fine_detail(self, tree: dict, period) -> tuple:
        """선택된 time period의 fine detail을 tree에서 추출.

        Returns:
            (entire_memories, divided_memories)
            - entire: [{time_period, general_memory}] — Level_1 summary
            - divided: [{time_period, general_memory}] — leaf captions
        """
        if isinstance(period, (list, tuple)) and len(period) >= 2:
            target_start = float(period[0])
            target_end = float(period[1])
        else:
            return [], []

        entire = []
        divided = []

        for l1_node in tree.get("Level_1", []):
            segs = l1_node.get("time_segments", [])
            if not segs:
                continue

            flat_times = []
            for s in segs:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    flat_times.extend([float(s[0]), float(s[1])])
            if not flat_times:
                continue

            node_start = min(flat_times)
            node_end = max(flat_times)

            # Check overlap with target period
            if node_end <= target_start or node_start >= target_end:
                continue

            # This L1 node overlaps — use as "entire"
            l1_summary = l1_node.get("summary", "")
            if l1_summary:
                entire.append({
                    "time_period": [node_start, node_end],
                    "general_memory": l1_summary,
                })

            # Get children (leaves) within the target period as "divided"
            for child in l1_node.get("children", []):
                st = float(child.get("start_time", 0))
                et = float(child.get("end_time", 0))
                if st >= target_start and et <= target_end:
                    caption = child.get("caption", child.get("summary", ""))
                    if caption:
                        divided.append({
                            "time_period": [st, et],
                            "general_memory": caption,
                        })

        divided.sort(key=lambda x: x["time_period"][0])
        print(f"    [FineDetail] period={period} → "
              f"{len(entire)} entire, {len(divided)} divided")
        return entire, divided

    # ==================== VideoLucy Prompts ====================
    # Copied from /lustre/youngbeom/VideoLucy/LLMs/utils.py

    @staticmethod
    def _answer_with_coarse_memory_prompt(memory, question, options):
        prompt = ("The following provides a rough description of what's shown "
                  "in the video during different time periods:\n")
        for i, mem in enumerate(memory):
            caption = mem["general_memory"]
            st = mem["time_period"][0]
            et = mem["time_period"][1]
            prompt += (f"{i+1}. Time Period: from {st}s to {et}s. "
                       f"Content Description: {caption}\n\n")

        prompt += ("Note that since these descriptions are not very complete and "
                   "detailed, some key information in the video segments of each "
                   "time period may not all appear in these content descriptions.\n")
        prompt += ("Now, a question has been raised regarding the content "
                   f"descriptions of this video.\n{question}\n")
        for opt in options:
            prompt += opt + "\n"

        prompt += ("Please read the given video content descriptions and the "
                   "question in depth, and determine whether you can accurately "
                   "answer the given question solely based on the currently "
                   "provided descriptions.\n")
        prompt += ("If you can answer it with absolute confidence, please answer "
                   "this question and provide the time periods you are referring to. "
                   "The answer you provide must have completely and absolutely "
                   "objective support in the video descriptions. "
                   "Do not make inferences arbitrarily.\n")
        if options:
            prompt += ("Please note that there is only one option that can answer "
                       "this question. The answer you provide must include the "
                       "English letters of the options [A, B, C, D].\n")
        prompt += ("If you think the current content descriptions of the video is "
                   "still insufficient to accurately answer the question, "
                   "please do not answer it and give me your reason.\n")
        prompt += ('Please output in a strictly standardized dictionary format '
                   'containing four key-value pairs:\n')
        prompt += ('"Confidence": A boolean value. Set it to True if you are '
                   'certain about the answer, and False if not.\n')
        prompt += ('"Answer": A string. This string must be enclosed in double '
                   'quotes. When "Confidence" is True, fill in the answer content; '
                   'when "Confidence" is False, fill in "No Answer".\n')
        prompt += ('"Time Period": A list. When "Confidence" is True, fill in the '
                   'list with time periods corresponding to the answer, each in '
                   'the format of a tuple (start time, end time); when "Confidence" '
                   'is False, fill in "No Time".\n')
        prompt += ('"Reason": A String. This string must be enclosed in double '
                   'quotes. Show me your reasoning about your judgment. '
                   'You need to ensure and check that your reasoning must be able '
                   'to absolutely support your answer.\n')
        prompt += "Note that no additional comments should be added within the dictionary."
        prompt += ('You must note that if an ordinal number appears in the provided '
                   'question, in the vast majority of cases, you should not simply '
                   'assume that this ordinal number represents the ordinal of the '
                   'provided time period. You need to focus on understanding the '
                   'specific meaning of this ordinal number within the question '
                   'based on all the content descriptions.\n')
        return prompt

    @staticmethod
    def _question_type_judge_prompt(memory, question, options):
        prompt = ("The following provides a rough description of what's shown "
                  "in the video during different time periods:\n")
        for i, mem in enumerate(memory):
            caption = mem["general_memory"]
            st = mem["time_period"][0]
            et = mem["time_period"][1]
            prompt += (f"{i+1}. Time Period: from {st}s to {et}s. "
                       f"Content Description: {caption}\n\n")

        prompt += ("Now, a question has been raised regarding this video.\n"
                   f"{question}\n")
        for opt in options:
            prompt += opt + "\n"

        prompt += "Please read the given video content descriptions and the question in depth.\n"
        prompt += ("Since most of these descriptions are rather rough and some "
                   "detailed information is lost, my task is to try my best to find "
                   "the time periods related to the given question, and then provide "
                   "more detailed descriptions of the video content of these time periods.\n")
        prompt += "In order to assist me in completing my task, your task is to:\n"
        prompt += ("Based on the provided rough video descriptions, determine whether "
                   "the given question allows me to provide a more confident answer by "
                   "further observing the video content of two time periods.\n")
        prompt += ("If so, you should find out the time periods related to the question "
                   "as much as possible and provide these relevant time periods so that "
                   "I can review the content information of these video segments again "
                   "to obtain more information and answer the question better.\n")
        prompt += ("For example, since there is no need for an overall understanding of "
                   "large video segments, the following questions can obtain more accurate "
                   "answers by carefully re-observing the video segments of two time periods:\n")
        prompt += "(i) What color is Putin's tie between the interview with Antony Blinkoen and interview with Marie Yovanovitch?\n"
        prompt += "(ii) How does the goalkeeper prevent Liverpool's shot from scoring at 81:38 in the video?\n"
        prompt += "(iii) Who smashes the magic mirror?\n"
        prompt += ("On the contrary, for example, because an overall understanding of "
                   "large video segments is required, it is difficult to obtain more "
                   "accurate answers to the following questions by merely observing two video segments:\n")
        prompt += "(i) What happens in the second half of the game?\n"
        prompt += "(ii) What is the video about?\n"
        prompt += "(iii) Which places has the protagonist of this video been to in total?\n"
        prompt += ('You should output in a strictly standardized dictionary format '
                   'containing three key-value pairs:\n')
        prompt += ('"Flag": A bool. If you are very confident that you can provide '
                   'the time periods according to the above requirements, set it as True. '
                   'Otherwise, set it as False.\n')
        prompt += ('"Time Period": A list. If "Flag" is True, fill in the list with '
                   'the most relevant two time periods, in the tuple format '
                   '(start time, end time). If "Flag" is False, fill in "No Time Periods."\n')
        prompt += ('"Reason": A String. This string must be enclosed in double quotes. '
                   'Show me your reasons for the time periods you provided.\n')
        prompt += "No additional comments should be added within the dictionary."
        return prompt

    @staticmethod
    def _get_single_related_time_prompt(coarse_memory, entire_list, divided_list,
                                         question, options, excluded_periods, duration):
        coarse_copy = copy.deepcopy(coarse_memory)
        fine_time_periods = []
        for entire in entire_list:
            if entire:
                st = entire[0]["time_period"][0]
                et = entire[-1]["time_period"][1]
                fine_time_periods.append([st, et])

        # Build total memories (coarse - already fine + fine)
        saved = [m for m in coarse_copy
                 if list(m["time_period"]) not in fine_time_periods]
        total = saved
        for entire in entire_list:
            total += entire
        total = sorted(total, key=lambda x: x["time_period"][0])

        prompt = f"There is currently a video with a total duration of {duration} seconds.\n"
        prompt += ("The following gives a general description of what is shown "
                   "in the video during certain time periods:\n")
        for i, mem in enumerate(total):
            caption = mem["general_memory"]
            st = mem["time_period"][0]
            et = mem["time_period"][1]
            prompt += (f"{i+1}. Time Period: from {st}s to {et}s. "
                       f"Content Description: {caption}\n\n")
            if [st, et] in fine_time_periods:
                idx = fine_time_periods.index([st, et])
                if idx < len(divided_list):
                    divs = divided_list[idx]
                    prompt += (f"Note that for the video within this time period "
                               f"from {st} seconds to {et} seconds, "
                               f"there is the following more detailed description:\n")
                    for j, d in enumerate(divs):
                        d_st = d["time_period"][0]
                        d_et = d["time_period"][1]
                        prompt += (f"    ({j+1}). Time Period: from {d_st}s to {d_et}s. "
                                   f"Content Description: {d['general_memory']}\n")

        prompt += (f"Now, a question has been raised regarding this entire video "
                   f"which has a duration of {duration} seconds.\n{question}\n")
        for opt in options:
            prompt += opt + "\n"

        prompt += "Please read the given video content descriptions and the question in depth.\n"
        prompt += "You do not need to answer this question.\n"
        prompt += ("Your first task is to identify, based on the video content in each "
                   "time period, the single time period that is most relevant to the "
                   "question and that you think requires further elaboration of its "
                   "video content details to make the answer to this question more explicit.\n")

        if excluded_periods:
            prompt += ("Notably, you only need to select the most relevant one from "
                       "the time periods other than the following time periods:\n")
            for p in excluded_periods:
                prompt += f"({p[0]},{p[1]});\n"

        prompt += ("In addition, assume there is now a caption model that can describe "
                   "a given video according to your instruction.\n")
        prompt += ("Your second task is to consider what detailed content in the video "
                   "of the time period you have selected you want the model to focus on "
                   "describing, and provide your instruction.\n")
        prompt += ("For example, assume that the entire video segment is about an "
                   "offensive play in a certain football game, and you want to focus "
                   "on the passing situation of the football during this offensive play. "
                   "The instruction you give to the model could be:\n")
        prompt += ("Please observe all the details in this video very carefully and "
                   "provide a detailed and objective description of what is shown in "
                   "the video. If this video is about an offensive play in a football "
                   "match, you should focus particularly on the passing situation of "
                   "the football during this offensive play.\n")
        prompt += ("Note that you should organize your instruction by strictly "
                   "referring to the language expressions in the above example.\n")
        prompt += ('You should output in a strictly standardized dictionary format '
                   'containing three key-value pairs:\n')
        prompt += ('"Time Period": A list. Fill in the list with the single most '
                   'relevant time period, in the tuple format (start time, end time).\n')
        prompt += ('"Instruction": A String. This string must be enclosed in double '
                   'quotes. Show me the instruction you want to give to the caption '
                   'model for the second task.\n')
        prompt += ('"Reason": A String. This string must be enclosed in double quotes. '
                   'Show me your reasons for the time period and instruction you provided.\n')
        prompt += "No additional comments should be added within the dictionary."
        prompt += ('You must note that if an ordinal number appears in the provided '
                   'question, in the vast majority of cases, you should not simply '
                   'assume that this ordinal number represents the ordinal of the '
                   'provided time period. You need to focus on understanding the '
                   'specific meaning of this ordinal number within the question '
                   'based on all the content descriptions.\n')
        return prompt

    @staticmethod
    def _answer_with_coarse_and_fine_prompt(coarse_memory, entire_list, divided_list,
                                             question, options, duration):
        coarse_copy = copy.deepcopy(coarse_memory)
        fine_time_periods = []
        for entire in entire_list:
            if entire:
                st = entire[0]["time_period"][0]
                et = entire[-1]["time_period"][1]
                fine_time_periods.append([st, et])

        saved = [m for m in coarse_copy
                 if list(m["time_period"]) not in fine_time_periods]
        total = saved
        for entire in entire_list:
            total += entire
        total = sorted(total, key=lambda x: x["time_period"][0])

        prompt = f"There is currently a video with a total duration of {duration} seconds.\n"
        prompt += ("The following gives a general description of what is shown "
                   "in the video during certain time periods:\n")
        for i, mem in enumerate(total):
            caption = mem["general_memory"]
            st = mem["time_period"][0]
            et = mem["time_period"][1]
            prompt += (f"{i+1}. Time Period: from {st}s to {et}s. "
                       f"Content Description: {caption}\n\n")
            if [st, et] in fine_time_periods:
                idx = fine_time_periods.index([st, et])
                if idx < len(divided_list):
                    divs = divided_list[idx]
                    prompt += (f"Note that for the video within this time period "
                               f"from {st} seconds to {et} seconds, "
                               f"there is the following more detailed description:\n")
                    for j, d in enumerate(divs):
                        d_st = d["time_period"][0]
                        d_et = d["time_period"][1]
                        prompt += (f"    ({j+1}). Time Period: from {d_st}s to {d_et}s. "
                                   f"Content Description: {d['general_memory']}\n")

        prompt += (f"Now, a question has been raised regarding the content "
                   f"descriptions of this video.\n{question}\n")
        for opt in options:
            prompt += opt + "\n"

        prompt += ("Please read the given video content descriptions and the "
                   "question in depth, and determine whether you can accurately "
                   "answer the given question solely based on the currently "
                   "provided descriptions.\n")
        prompt += ("If you can answer it with absolute confidence, please answer "
                   "this question and provide the time periods of the video content "
                   "you are referring to. The answer you provided must have completely "
                   "and absolutely objective support in the video descriptions. "
                   "Do not make inferences arbitrarily.\n")
        if options:
            prompt += ("Please note that there is only one option that can answer "
                       "this question. The answer you provide must include the "
                       "English letters of the options [A, B, C, D].\n")
        prompt += ("If you think the current content descriptions of the video are "
                   "still insufficient to accurately answer the question, please do "
                   "not answer it and give me your reason.\n")
        prompt += ('Please output in a strictly standardized dictionary format '
                   'containing four key-value pairs:\n')
        prompt += ('"Confidence": A boolean value. Set it to True if you are '
                   'certain about the answer, and False if not.\n')
        prompt += ('"Answer": A string. This string must be enclosed in double '
                   'quotes. When "Confidence" is True, fill in the answer content; '
                   'when "Confidence" is False, fill in "No Answer".\n')
        prompt += ('"Time Period": A list. When "Confidence" is True, fill in the '
                   'list with time periods corresponding to the answer, each in '
                   'the format of a tuple (start time, end time); when "Confidence" '
                   'is False, fill in "No Time".\n')
        prompt += ('"Reason": A String. This string must be enclosed in double '
                   'quotes. Show me your reasoning about your judgment. '
                   'You need to ensure and check that your reasoning must be able '
                   'to absolutely support your answer.\n')
        prompt += "No additional comments should be added within the dictionary."
        prompt += ('You must note that if an ordinal number appears in the provided '
                   'question, in the vast majority of cases, you should not simply '
                   'assume that this ordinal number represents the ordinal of the '
                   'provided time period. You need to focus on understanding the '
                   'specific meaning of this ordinal number within the question '
                   'based on all the content descriptions.\n')
        return prompt

    @staticmethod
    def _must_answer_prompt(coarse_memory, entire_list, divided_list,
                            question, options, duration):
        coarse_copy = copy.deepcopy(coarse_memory)
        fine_time_periods = []
        for entire in entire_list:
            if entire:
                st = entire[0]["time_period"][0]
                et = entire[-1]["time_period"][1]
                fine_time_periods.append([st, et])

        saved = [m for m in coarse_copy
                 if list(m["time_period"]) not in fine_time_periods]
        total = saved
        for entire in entire_list:
            total += entire
        total = sorted(total, key=lambda x: x["time_period"][0])

        prompt = f"There is currently a video with a total duration of {duration} seconds.\n"
        prompt += ("The following gives a general description of what is shown "
                   "in the video during certain time periods:\n")
        for i, mem in enumerate(total):
            caption = mem["general_memory"]
            st = mem["time_period"][0]
            et = mem["time_period"][1]
            prompt += (f"{i+1}. Time Period: from {st}s to {et}s. "
                       f"Content Description: {caption}\n\n")
            if [st, et] in fine_time_periods:
                idx = fine_time_periods.index([st, et])
                if idx < len(divided_list):
                    divs = divided_list[idx]
                    prompt += (f"Note that for the video within this time period "
                               f"from {st} seconds to {et} seconds, "
                               f"there is the following more detailed and accurate description:\n")
                    for j, d in enumerate(divs):
                        d_st = d["time_period"][0]
                        d_et = d["time_period"][1]
                        prompt += (f"    ({j+1}). Time Period: from {d_st}s to {d_et}s. "
                                   f"Content Description: {d['general_memory']}\n")

        prompt += (f"Now, a question has been raised regarding the content "
                   f"descriptions of this video.\n{question}\n")
        for opt in options:
            prompt += opt + "\n"

        prompt += ("Please read and understand the given video content and "
                   "question in depth. ")
        if options:
            prompt += ("Strictly based on the video content, select the single best "
                       "option. You must choose an option from these provided options. "
                       "The answer you provide must include the English letters of "
                       "the options [A, B, C, D].\n")
        else:
            prompt += "You must provide a best answer for this question.\n"

        prompt += ("Please note that if an ordinal number appears in the provided "
                   "question, in most cases, the meaning of this ordinal number is "
                   "not related to the ordinal of the provided time period. "
                   "You need to focus on analyzing the meaning of this ordinal number.\n")
        prompt += ('Please output in a strictly standardized dictionary format '
                   'containing three key-value pair:\n')
        prompt += ('"Answer": A string. This string must be enclosed in double '
                   'quotes. The best answer for the question.\n')
        prompt += ('"Time Period": A list. Fill in the list with time periods '
                   'corresponding to the best answer, each in the format of a '
                   'tuple (start time, end time).\n')
        prompt += ('"Reason": A String. This string must be enclosed in double '
                   'quotes. Show me your reasoning about your judgment. '
                   'You need to ensure and check that your reasoning must be able '
                   'to absolutely support your answer.\n')
        prompt += "No additional comments should be added within the dictionary."
        return prompt

    # ==================== LLM Generation ====================

    def _generate_text(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate raw text using model/processor (no JSON parsing).

        Accesses the underlying model/processor from TextOnlyLLM.
        """
        llm = self.components.get("llm")
        if not llm:
            return ""

        model = llm.model
        processor = llm.processor

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for video question answering. "
                    "Follow the output format instructions exactly."
                ),
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=text, images=None, videos=None, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
            generated_ids = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, output_ids)
            ]
            response = processor.batch_decode(
                generated_ids, skip_special_tokens=True,
            )[0].strip()

        del inputs, output_ids, generated_ids
        torch.cuda.empty_cache()

        return response

    # ==================== Utilities ====================

    @staticmethod
    def _parse_answer(answer_string: str) -> dict | None:
        """VideoLucy parse_answer: LLM output에서 dict 추출."""
        if not answer_string:
            return None
        # Remove think tags
        answer_string = answer_string.split("</think>")[-1]
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, answer_string)
        if not matches:
            print(f"    [Parse] No dict pattern found")
            return None

        last_dict_str = matches[-1]
        last_dict_str = (last_dict_str
                         .replace("false", "False")
                         .replace("true", "True")
                         .replace("\n", ""))
        try:
            return ast.literal_eval(last_dict_str)
        except (SyntaxError, ValueError):
            print(f"    [Parse] Failed to parse: {last_dict_str[:100]}")
            return None

    @staticmethod
    def _extract_option(answer_text: str, options: list) -> str:
        """Answer text에서 A/B/C/D 추출."""
        if not answer_text:
            return "A"
        m = re.search(r"[ABCD]", answer_text.upper())
        return m.group(0) if m else "A"

    @staticmethod
    def _contains_ordinal_number(text: str) -> bool:
        ordinals = [
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth',
            'seventh', 'eighth', 'ninth', 'tenth', 'last',
            '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th',
            '9th', '10th', '11th', '12th', '13th', '14th', '15th',
            '20th', '30th',
        ]
        return any(o in text.lower() for o in ordinals)

    @staticmethod
    def _filter_coarse_memory(coarse_memory, related_periods):
        """Time period 기반 coarse memory filtering."""
        if not related_periods:
            return coarse_memory

        # Simple overlap-based filtering
        filtered = []
        for mem in coarse_memory:
            mem_st, mem_et = mem["time_period"]
            for rp in related_periods:
                if isinstance(rp, (list, tuple)) and len(rp) >= 2:
                    rp_st, rp_et = float(rp[0]), float(rp[1])
                    # Check overlap
                    if mem_st < rp_et and mem_et > rp_st:
                        filtered.append(mem)
                        break

        return filtered if filtered else coarse_memory

    def _make_result(self, pred, answer, correct, method, **kwargs):
        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": method,
            "confidence": "high" if correct else "low",
            "used_visual": False,
            "traversal_log": [],
            "traversal_summary": method,
            "total_hops": kwargs.get("fine_iterations", 0),
            "fine_periods": kwargs.get("fine_periods", []),
            "coarse_answer": kwargs.get("coarse_answer"),
            "fine_answer": kwargs.get("fine_answer"),
        }
