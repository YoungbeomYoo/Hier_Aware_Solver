"""
HD-EPIC Adapter — HD-EPIC VQA 벤치마크 데이터셋 어댑터

Question format: JSON dict {q_id: {inputs, question, choices, correct_idx}}
Answer format: 0-based index
Memory: streaming_memory_tree
"""

import os
import json
from glob import glob
from adapters.base import BaseAdapter


class HDEpicAdapter(BaseAdapter):

    def load_questions(self) -> dict:
        """HD-EPIC 질문 로딩.

        question_path 디렉토리 내 모든 .json 파일을 로딩.
        {video_id: [normalized_question_dict, ...]} 형태로 반환.
        """
        q_dir = self.config["question_path"]
        q_files = sorted(glob(os.path.join(q_dir, "*.json")))

        questions_by_video = {}
        for qf in q_files:
            with open(qf, "r", encoding="utf-8") as f:
                data = json.load(f)

            for q_id, q_data in data.items():
                inputs = q_data.get("inputs", {})
                # Extract primary video ID
                vid_ids = []
                for key, val in inputs.items():
                    if isinstance(val, dict) and "id" in val:
                        vid_ids.append(val["id"])

                primary_vid = vid_ids[0] if vid_ids else "unknown"

                normalized = {
                    "question_id": q_id,
                    "question": q_data["question"],
                    "options": q_data["choices"],
                    "answer": q_data.get("correct_idx"),
                    "inputs": inputs,
                    "video_ids": vid_ids,
                    "source_file": os.path.basename(qf),
                }

                if primary_vid not in questions_by_video:
                    questions_by_video[primary_vid] = []
                questions_by_video[primary_vid].append(normalized)

        return questions_by_video

    def load_memory(self, video_id: str) -> dict:
        """HD-EPIC 메모리 로딩."""
        mem_dir = self.config["memory_dir"]
        # Try both direct and _synced variants
        for suffix in ["_synced.json", ".json"]:
            path = os.path.join(mem_dir, f"{video_id}{suffix}")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def get_video_path(self, video_id: str) -> str:
        """HD-EPIC 비디오 경로."""
        video_root = self.config.get("video_root", "")
        # HD-EPIC videos are organized by participant ID
        participant = video_id.split("-")[0]  # e.g. P06
        return os.path.join(video_root, participant, f"{video_id}.mp4")

    LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def _to_index(self, val):
        """Convert letter (A/B/C/D) or int-string to 0-based index."""
        if val is None:
            return None
        s = str(val).strip().upper()
        if s in self.LETTER_TO_IDX:
            return self.LETTER_TO_IDX[s]
        try:
            return int(s)
        except ValueError:
            return None

    def check_correct(self, predicted, ground_truth) -> bool:
        """HD-EPIC: index 비교 (letter A/B/C/D도 자동 변환)."""
        pred_idx = self._to_index(predicted)
        gt_idx = self._to_index(ground_truth)
        if pred_idx is None or gt_idx is None:
            return False
        return pred_idx == gt_idx
