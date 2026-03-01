"""
Sub-Question Verifier — Vgent-style Structured Reasoning

Verifies sub-questions against leaf segments, filters by positive answers,
and ranks segments by relevance.

Supports text-only (caption-based) and visual (VLM frame-based) modes.
"""

from __future__ import annotations

import json


class SubQuestionVerifier:
    """Verify sub-questions against video segments.

    Args:
        llm_fn: callable(prompt, max_tokens) -> dict
        prompt_template: prompt with {segment_description} and {questions} placeholders
        vision_vlm: VisionVLM instance for visual mode (optional)
        frame_loader: TargetedFrameLoader instance (optional)
        visual_prompt_template: prompt for VLM visual verification (optional)
    """

    def __init__(self, llm_fn, prompt_template: str,
                 vision_vlm=None, frame_loader=None,
                 visual_prompt_template: str | None = None):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template
        self.vision_vlm = vision_vlm
        self.frame_loader = frame_loader
        self.visual_prompt_template = visual_prompt_template

    def verify_text(self, leaves: list[dict], sub_questions: dict) -> dict:
        """Text-only verification: use leaf captions to answer sub-questions.

        Args:
            leaves: list of leaf dicts with 'summary', 'start_time', 'end_time'
            sub_questions: dict like {"Q1": "Is there...?", ...}

        Returns:
            dict {leaf_idx: {"Q1": "yes", "Q2": "no", ...}, ...}
        """
        questions_str = json.dumps(sub_questions, ensure_ascii=False)
        results = {}

        for idx, leaf in enumerate(leaves):
            caption = leaf.get("summary", "") or leaf.get("caption", "")
            if not caption:
                continue

            st = leaf.get("start_time", 0)
            et = leaf.get("end_time", 0)
            segment_desc = f"[{float(st):.0f}s-{float(et):.0f}s] {caption}"

            prompt = self.prompt_template.format(
                segment_description=segment_desc,
                questions=questions_str,
            )

            try:
                result = self.llm_fn(prompt, max_tokens=256)
            except Exception as e:
                print(f"      [Verify] Leaf {idx} error: {e}")
                continue

            if isinstance(result, dict):
                # Filter to valid Q-keyed yes/no answers
                answers = {}
                for k, v in result.items():
                    if isinstance(k, str) and k.startswith("Q"):
                        v_str = str(v).lower().strip()
                        if v_str in ("yes", "no"):
                            answers[k] = v_str
                        else:
                            answers[k] = "no"
                results[idx] = answers

        print(f"    [Verify] Text-verified {len(results)}/{len(leaves)} leaves")
        return results

    def verify_visual(self, leaves: list[dict], sub_questions: dict,
                      video_path: str, max_frames: int = 8) -> dict:
        """Visual verification: use VLM with video frames.

        Args:
            leaves: list of leaf dicts
            sub_questions: dict of sub-questions
            video_path: path to video file
            max_frames: max frames per leaf

        Returns:
            dict {leaf_idx: {"Q1": "yes", ...}, ...}
        """
        if not self.vision_vlm or not self.frame_loader:
            print("    [Verify] No VLM/frame_loader, falling back to text")
            return self.verify_text(leaves, sub_questions)

        questions_str = json.dumps(sub_questions, ensure_ascii=False)
        prompt = self.visual_prompt_template or self.prompt_template
        prompt = prompt.format(
            segment_description="",
            questions=questions_str,
        )

        results = {}

        for idx, leaf in enumerate(leaves):
            st = float(leaf.get("start_time", 0))
            et = float(leaf.get("end_time", 0))

            try:
                frames = self.frame_loader.load_region(
                    video_path, st, et, max_frames=max_frames
                )
            except Exception as e:
                print(f"      [Verify] Frame load error for leaf {idx}: {e}")
                continue

            if not frames:
                continue

            try:
                result = self.vision_vlm.answer_with_frames(
                    frames=frames,
                    question=prompt,
                    options=[],
                    memory_context="",
                )
            except Exception as e:
                print(f"      [Verify] VLM error for leaf {idx}: {e}")
                continue

            if isinstance(result, dict):
                answers = {}
                for k, v in result.items():
                    if isinstance(k, str) and k.startswith("Q"):
                        v_str = str(v).lower().strip()
                        answers[k] = v_str if v_str in ("yes", "no") else "no"
                results[idx] = answers

        print(f"    [Verify] Visual-verified {len(results)}/{len(leaves)} leaves")
        return results

    @staticmethod
    def filter_and_rank(check_results: dict, sub_questions: dict,
                        n_refine: int = 5) -> list[int]:
        """Filter leaves with positive answers and rank by yes-count.

        Args:
            check_results: {leaf_idx: {"Q1": "yes", ...}, ...}
            sub_questions: original sub-questions dict
            n_refine: max leaves to keep

        Returns:
            list of leaf indices, sorted by relevance (most yes-answers first)
        """
        scored = []
        for leaf_idx, answers in check_results.items():
            yes_count = sum(1 for v in answers.values() if v == "yes")
            if yes_count > 0:
                scored.append((leaf_idx, yes_count))

        # Sort by yes_count descending
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [idx for idx, _ in scored[:n_refine]]
        print(f"    [Filter] {len(scored)} positive leaves → kept top {len(selected)}")
        return selected
