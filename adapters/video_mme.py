"""
Video-MME Adapter — Video-MME 벤치마크 데이터셋 어댑터

Question format: per-file JSON {videoID, question_id, question, options, answer}
Answer format: letter (A/B/C/D)
Memory: streaming_memory_tree
"""

import os
import json
from glob import glob
from adapters.base import BaseAdapter


class VideoMMEAdapter(BaseAdapter):

    def load_questions(self) -> dict:
        """Video-MME 질문 로딩.

        question_path 디렉토리 내 모든 .json 파일.
        각 파일은 하나의 question (또는 list).
        """
        q_dir = self.config["question_path"]
        q_files = sorted(glob(os.path.join(q_dir, "**", "*.json"), recursive=True))

        questions_by_video = {}
        for qf in q_files:
            try:
                with open(qf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            qlist = data if isinstance(data, list) else [data]

            for qobj in qlist:
                video_id = str(qobj.get("videoID") or qobj.get("video_id") or "")
                question_id = qobj.get("question_id", os.path.basename(qf).replace(".json", ""))

                # Extract subtitles if present
                subtitles = ""
                for k in ["subtitles", "subtitle", "subs", "captions", "caption"]:
                    if k in qobj and qobj[k]:
                        v = qobj[k]
                        subtitles = "\n".join(map(str, v)) if isinstance(v, list) else str(v)
                        break

                normalized = {
                    "question_id": question_id,
                    "question": qobj.get("question", ""),
                    "options": qobj.get("options", []),
                    "answer": qobj.get("answer"),
                    "video_id_raw": video_id,
                    "subtitles": subtitles,
                    "task_type": qobj.get("task_type", ""),
                    "domain": qobj.get("domain", ""),
                    "duration": qobj.get("duration", ""),
                    "source_file": qf,
                }

                if video_id not in questions_by_video:
                    questions_by_video[video_id] = []
                questions_by_video[video_id].append(normalized)

        return questions_by_video

    def load_memory(self, video_id: str) -> dict:
        """Video-MME 메모리 로딩."""
        mem_dir = self.config["memory_dir"]
        for suffix in ["_synced.json", ".json"]:
            path = os.path.join(mem_dir, f"{video_id}{suffix}")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def get_video_path(self, video_id: str) -> str:
        """Video-MME 비디오 경로."""
        video_root = self.config.get("video_root", "")
        for ext in [".mp4", ".mkv", ".webm"]:
            path = os.path.join(video_root, f"{video_id}{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(video_root, f"{video_id}.mp4")

    def check_correct(self, predicted, ground_truth) -> bool:
        """Video-MME: letter 비교."""
        if predicted is None or ground_truth is None:
            return False
        return str(predicted).strip().upper() == str(ground_truth).strip().upper()

    def save_summary(self, results: list[dict]):
        """Video-MME: task_type별 세분화된 summary."""
        total = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        accuracy = correct / total if total > 0 else 0.0

        # By task_type
        by_type = {}
        for r in results:
            tt = r.get("task_type", "unknown")
            if tt not in by_type:
                by_type[tt] = {"total": 0, "correct": 0}
            by_type[tt]["total"] += 1
            if r.get("correct"):
                by_type[tt]["correct"] += 1

        for tt in by_type:
            t = by_type[tt]["total"]
            by_type[tt]["accuracy"] = round(by_type[tt]["correct"] / t, 4) if t > 0 else 0.0

        summary = {
            "dataset": "Video-MME",
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "by_task_type": by_type,
        }

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary
