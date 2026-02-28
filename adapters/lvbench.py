"""
LVBench Adapter — LVBench 벤치마크 데이터셋 어댑터

Question format: JSONL (video별 qa list)
Answer format: letter (A/B/C/D)
Memory: streaming_memory_tree
"""

import os
import re
import json
from adapters.base import BaseAdapter


class LVBenchAdapter(BaseAdapter):

    def load_questions(self) -> dict:
        """LVBench 질문 로딩.

        question_path: JSONL 파일 (한 줄에 한 비디오).
        question 문자열 안에 (A)...(B)... 형태로 선택지가 포함됨.
        """
        q_path = self.config["question_path"]
        questions_by_video = {}

        with open(q_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                video_key = entry.get("key", "")
                qa_list = entry.get("qa", [])

                normalized_list = []
                for qa in qa_list:
                    question_text, options = self._parse_question(qa)
                    normalized = {
                        "question_id": qa.get("uid", ""),
                        "question": question_text,
                        "options": options,
                        "answer": qa.get("answer", ""),
                        "question_type": qa.get("question_type", []),
                        "time_reference": qa.get("time_reference", ""),
                        "video_type": entry.get("type", ""),
                    }
                    normalized_list.append(normalized)

                if video_key and normalized_list:
                    questions_by_video[video_key] = normalized_list

        return questions_by_video

    @staticmethod
    def _parse_question(qa: dict) -> tuple[str, list[str]]:
        """LVBench question에서 질문 텍스트와 선택지 분리.

        Input: "What year appears?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633"
        Output: ("What year appears?", ["A. 1636", "B. 1366", "C. 1363", "D. 1633"])
        """
        raw_q = qa.get("question", "")
        match = re.search(r"\n\(A\)", raw_q)
        if match:
            question_text = raw_q[: match.start()].strip()
            options_text = raw_q[match.start() :].strip()
            options = re.findall(r"\([A-D]\)\s*[^\n]+", options_text)
            options = [re.sub(r"^\(([A-D])\)\s*", r"\1. ", o) for o in options]
        else:
            question_text = raw_q
            options = []
        return question_text, options

    def load_memory(self, video_id: str) -> dict:
        """LVBench 메모리 로딩."""
        mem_dir = self.config["memory_dir"]
        for suffix in ["_synced.json", ".json"]:
            path = os.path.join(mem_dir, f"{video_id}{suffix}")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def get_video_path(self, video_id: str) -> str:
        """LVBench 비디오 경로."""
        video_root = self.config.get("video_root", "")
        for ext in [".mp4", ".mkv", ".webm"]:
            path = os.path.join(video_root, f"{video_id}{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(video_root, f"{video_id}.mp4")

    def check_correct(self, predicted, ground_truth) -> bool:
        """LVBench: letter 비교."""
        if predicted is None or ground_truth is None:
            return False
        return str(predicted).strip().upper() == str(ground_truth).strip().upper()

    def save_summary(self, results: list[dict]):
        """LVBench: question_type별 세분화된 summary."""
        total = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        accuracy = correct / total if total > 0 else 0.0

        # By question_type
        by_type = {}
        for r in results:
            qtypes = r.get("question_type", [])
            if not qtypes:
                qtypes = ["unknown"]
            for qt in qtypes:
                if qt not in by_type:
                    by_type[qt] = {"total": 0, "correct": 0}
                by_type[qt]["total"] += 1
                if r.get("correct"):
                    by_type[qt]["correct"] += 1

        for qt in by_type:
            t = by_type[qt]["total"]
            by_type[qt]["accuracy"] = round(by_type[qt]["correct"] / t, 4) if t > 0 else 0.0

        summary = {
            "dataset": "LVBench",
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "by_question_type": by_type,
        }

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary
