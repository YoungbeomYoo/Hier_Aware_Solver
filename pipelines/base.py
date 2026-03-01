from __future__ import annotations

"""
Base Pipeline — 파이프라인 추상 클래스

모든 solver pipeline은 이 ABC를 상속.
컴포넌트들을 조합하여 solve() 메서드를 구현.
"""

import os
import time
import json
import traceback
from abc import ABC, abstractmethod
from tqdm import tqdm


class BasePipeline(ABC):
    """Solver pipeline 추상 인터페이스.

    Args:
        components: dict of instantiated component objects
        adapter: BaseAdapter instance
        config: pipeline configuration dict
    """

    def __init__(self, components: dict, adapter, config: dict):
        self.components = components
        self.adapter = adapter
        self.config = config
        self.cached = config.get("cached", True)  # Skip already processed

    @abstractmethod
    def solve(self, question_data: dict, memory: dict,
              video_id: str) -> dict:
        """단일 question 풀기.

        Args:
            question_data: normalized question dict
            memory: loaded memory dict
            video_id: video identifier

        Returns:
            result dict with pred, answer, correct, etc.
        """
        ...

    def run_question(self, video_id: str, question_id: str):
        """단일 질문 하나만 처리.

        Args:
            video_id: 비디오 ID
            question_id: 질문 ID

        Returns:
            result dict or None
        """
        questions_by_video = self.adapter.load_questions()

        if video_id not in questions_by_video:
            print(f"  [ERROR] Video {video_id} not found in questions")
            return None

        questions = questions_by_video[video_id]
        q_data = None
        for q in questions:
            if str(q["question_id"]) == str(question_id):
                q_data = q
                break

        if q_data is None:
            # video_id 없이 전체에서 찾기
            for vid, qs in questions_by_video.items():
                for q in qs:
                    if str(q["question_id"]) == str(question_id):
                        q_data = q
                        video_id = vid
                        break
                if q_data:
                    break

        if q_data is None:
            print(f"  [ERROR] Question {question_id} not found")
            return None

        memory = self.adapter.load_memory(video_id)
        if not memory:
            print(f"  [SKIP] No memory for {video_id}")
            return None

        print(f"Processing question {question_id} (video: {video_id})")

        start_t = time.time()
        try:
            result = self.solve(q_data, memory, video_id)
        except Exception as e:
            print(f"  [ERROR] {question_id}: {e}")
            traceback.print_exc()
            result = {
                "question_id": question_id,
                "video_id": video_id,
                "pred": None,
                "answer": q_data.get("answer"),
                "correct": False,
                "error": str(e),
            }

        result["time"] = time.time() - start_t
        result["question_id"] = question_id
        result["video_id"] = video_id

        self.adapter.save_result(question_id, result)
        print(f"  [{question_id}] pred={result.get('pred')} "
              f"gt={result.get('answer')} correct={result.get('correct')}")
        return result

    def run_video(self, video_id: str):
        """단일 비디오의 모든 질문 처리.

        Args:
            video_id: 처리할 비디오 ID

        Returns:
            list of result dicts
        """
        questions_by_video = self.adapter.load_questions()

        if video_id not in questions_by_video:
            print(f"  [ERROR] Video {video_id} not found in questions")
            return []

        questions = questions_by_video[video_id]
        memory = self.adapter.load_memory(video_id)

        if not memory:
            print(f"  [SKIP] No memory for {video_id}")
            return []

        print(f"Processing video {video_id}: {len(questions)} questions")

        results = []
        correct_count = 0
        total_count = 0

        for q_data in tqdm(questions, desc=f"  QA ({video_id})", leave=False):
            q_id = q_data["question_id"]

            if self.cached and self.adapter.is_cached(q_id):
                continue

            start_t = time.time()
            try:
                result = self.solve(q_data, memory, video_id)
            except Exception as e:
                print(f"  [ERROR] {q_id}: {e}")
                traceback.print_exc()
                result = {
                    "question_id": q_id,
                    "video_id": video_id,
                    "pred": None,
                    "answer": q_data.get("answer"),
                    "correct": False,
                    "error": str(e),
                }

            result["time"] = time.time() - start_t
            result["question_id"] = q_id
            result["video_id"] = video_id

            self.adapter.save_result(q_id, result)
            results.append(result)

            if result.get("answer") is not None:
                total_count += 1
                if result.get("correct"):
                    correct_count += 1

            acc = correct_count / total_count if total_count > 0 else 0.0
            print(f"  [{q_id}] pred={result.get('pred')} "
                  f"gt={result.get('answer')} "
                  f"correct={result.get('correct')} "
                  f"| acc={acc:.4f} ({correct_count}/{total_count})")

        print(f"\n  Video {video_id} done: {correct_count}/{total_count}")
        return results

    def run_all(self, video_ids: list[str] | None = None,
                question_ids: set[str] | None = None):
        """전체 벤치마크 실행 (또는 지정된 비디오/문제만).

        Args:
            video_ids: 처리할 비디오 ID 리스트 (None이면 전체)
            question_ids: 처리할 문제 ID set (None이면 전체)

        Returns:
            (all_results, summary)
        """
        questions_by_video = self.adapter.load_questions()

        if video_ids:
            # 지정된 비디오만 필터링
            questions_by_video = {
                vid: qs for vid, qs in questions_by_video.items()
                if vid in video_ids
            }

        if question_ids:
            # 문제 ID 기준 필터링 (해당 문제가 있는 비디오만 남김)
            filtered = {}
            for vid, qs in questions_by_video.items():
                selected = [q for q in qs if str(q.get("question_id", "")) in question_ids]
                if selected:
                    filtered[vid] = selected
            questions_by_video = filtered

        total_questions = sum(len(qs) for qs in questions_by_video.values())
        print(f"Loaded {len(questions_by_video)} videos, {total_questions} questions.")

        all_results = []
        correct_count = 0
        total_count = 0

        for video_id in tqdm(sorted(questions_by_video.keys()), desc="Videos"):
            questions = questions_by_video[video_id]
            memory = self.adapter.load_memory(video_id)

            if not memory:
                print(f"  [SKIP] No memory for {video_id}")
                continue

            for q_data in tqdm(questions, desc=f"  QA ({video_id})", leave=False):
                q_id = q_data["question_id"]

                # Skip if cached
                if self.cached and self.adapter.is_cached(q_id):
                    continue

                start_t = time.time()
                try:
                    result = self.solve(q_data, memory, video_id)
                except Exception as e:
                    print(f"  [ERROR] {q_id}: {e}")
                    traceback.print_exc()
                    result = {
                        "question_id": q_id,
                        "video_id": video_id,
                        "pred": None,
                        "answer": q_data.get("answer"),
                        "correct": False,
                        "error": str(e),
                    }

                result["time"] = time.time() - start_t
                result["question_id"] = q_id
                result["video_id"] = video_id

                # Save per-question
                self.adapter.save_result(q_id, result)
                all_results.append(result)

                # Running accuracy
                if result.get("answer") is not None:
                    total_count += 1
                    if result.get("correct"):
                        correct_count += 1

                acc = correct_count / total_count if total_count > 0 else 0.0
                print(f"  [{q_id}] pred={result.get('pred')} "
                      f"gt={result.get('answer')} "
                      f"correct={result.get('correct')} "
                      f"| running acc={acc:.4f} ({correct_count}/{total_count})")

        summary = self.adapter.save_summary(all_results)
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY: {json.dumps(summary, indent=2)}")

        return all_results, summary
