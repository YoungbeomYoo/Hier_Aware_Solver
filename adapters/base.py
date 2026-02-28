"""
Base Adapter — 데이터셋 어댑터 추상 클래스

모든 벤치마크 어댑터는 이 ABC를 상속하여 구현.
"""

import os
import json
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """데이터셋별 question/memory 로딩 및 결과 저장 추상 인터페이스."""

    def __init__(self, config: dict):
        """
        Args:
            config: paths (memory_dir, question_path, output_dir, video_root 등)
        """
        self.config = config
        self.output_dir = config.get("output_dir", "./output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.by_qid_dir = os.path.join(self.output_dir, "by_qid")
        os.makedirs(self.by_qid_dir, exist_ok=True)

    @abstractmethod
    def load_questions(self) -> dict:
        """질문 데이터 로딩.

        Returns:
            {video_id: [{"question_id": str, "question": str, "options": list,
                         "answer": str, ...}, ...]}
        """
        ...

    @abstractmethod
    def load_memory(self, video_id: str) -> dict:
        """비디오별 메모리 로딩.

        Returns:
            메모리 dict (streaming_memory_tree 포함)
        """
        ...

    @abstractmethod
    def get_video_path(self, video_id: str) -> str:
        """비디오 파일 경로 반환."""
        ...

    @abstractmethod
    def check_correct(self, predicted: str, ground_truth) -> bool:
        """예측과 정답 비교."""
        ...

    def save_result(self, question_id: str, result: dict):
        """Per-question JSON 저장."""
        safe_qid = self._safe_filename(question_id)
        path = os.path.join(self.by_qid_dir, f"{safe_qid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def is_cached(self, question_id: str) -> bool:
        """이미 처리된 question인지 확인 (중단 후 재시작 지원)."""
        safe_qid = self._safe_filename(question_id)
        return os.path.exists(os.path.join(self.by_qid_dir, f"{safe_qid}.json"))

    def save_summary(self, results: list[dict]):
        """전체 결과 summary JSON 저장."""
        total = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        accuracy = correct / total if total > 0 else 0.0

        summary = {
            "dataset": self.__class__.__name__.replace("Adapter", ""),
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
        }

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary

    @staticmethod
    def _safe_filename(s: str) -> str:
        import re
        return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))
