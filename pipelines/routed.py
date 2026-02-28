"""
Routed Pipeline — Track A/B routing + Targeted Frame Loading (HD-EPIC style)

Flow:
  TimeRouter.extract →
    Track A: Bottom-up context → Targeted frames
    Track B: QueryDecomposer → HierarchicalNavigator → Targeted frames
  → VisionVLM → AnswerParser
"""

import gc
import json
from PIL import Image
from pipelines.base import BasePipeline
from components.answer_parser import extract_choice_letter, parse_response_index
from components.time_utils import secs_to_time_str


class RoutedPipeline(BasePipeline):
    """Track A/B routed solver pipeline.

    Required components:
        - time_router: TimeRouter
        - formatter: MemoryContextFormatter
        - frame_loader: TargetedFrameLoader
        - vision_vlm: VisionVLM

    Optional components:
        - decomposer: QueryDecomposer (Track B)
        - navigator: HierarchicalNavigator (Track B)
        - uniform_loader: UniformFrameLoader (fallback)
    """

    def solve(self, question_data: dict, memory: dict, video_id: str) -> dict:
        time_router = self.components["time_router"]
        formatter = self.components["formatter"]
        frame_loader = self.components["frame_loader"]
        vision_vlm = self.components["vision_vlm"]
        decomposer = self.components.get("decomposer")
        navigator = self.components.get("navigator")

        question = question_data["question"]
        options = question_data["options"]
        answer = question_data.get("answer")

        tree = memory.get("streaming_memory_tree", {})

        # ============================================================
        # Phase 1: Track A/B Routing
        # ============================================================
        full_text = question + " " + " ".join(options)
        explicit_ranges = time_router.extract_time_ranges(full_text)

        localized_memory = ""
        route_evidence = {}
        target_intervals = []

        if explicit_ranges:
            # Track A: Direct time indexing
            raw_dict = {video_id: memory}
            localized_memory = formatter.format_bottom_up(raw_dict, explicit_ranges)
            hierarchy_paths = time_router.get_hierarchy_path(raw_dict, explicit_ranges)
            route_evidence = {
                "track": "A",
                "matched_time_ranges_abs": [[round(s, 3), round(e, 3)] for s, e in explicit_ranges],
                "hierarchy_path": hierarchy_paths,
            }
            target_intervals = explicit_ranges

            if not localized_memory:
                explicit_ranges = []  # Fall through to Track B

        if not explicit_ranges or not localized_memory:
            # Track B: Semantic search
            if decomposer and navigator:
                parsed_query = decomposer.decompose(question, options)
                cues = parsed_query.get("cues", [])
                raw_dict = {video_id: memory}
                localized_memory, selected_segment, hierarchy_path = navigator.navigate(raw_dict, cues)
                route_evidence = {
                    "track": "B",
                    "cues": cues,
                    "target_action": parsed_query.get("target_action", ""),
                    "selected_segment": selected_segment,
                    "hierarchy_path": hierarchy_path,
                }
                # Parse selected segment time
                parsed_times = time_router.extract_time_ranges(selected_segment)
                target_intervals = parsed_times
            else:
                # No decomposer: use all memory as context
                localized_memory = formatter.format_flat(memory, max_chars=12000)
                route_evidence = {"track": "none"}

        # ============================================================
        # Phase 2: Frame Loading
        # ============================================================
        video_path = self.adapter.get_video_path(video_id)
        frames_np = None
        frame_seconds = []

        if target_intervals:
            frames_np, frame_seconds = frame_loader.load(video_path, target_intervals)

        # Fallback: uniform sampling
        if frames_np is None:
            uniform_loader = self.components.get("uniform_loader")
            if uniform_loader:
                frames_np, frame_seconds = uniform_loader.load(video_path)

        # ============================================================
        # Phase 3: VLM Inference
        # ============================================================
        if frames_np is not None:
            result = vision_vlm.infer(
                frames_np, localized_memory, question, options
            )
            pred_raw = result.get("answer", "")
            pred = extract_choice_letter(pred_raw)
            raw_response = result.get("raw_response", pred_raw)

            del frames_np
            gc.collect()
        else:
            # Text-only fallback
            simple = self.components.get("simple_vlm")
            if simple:
                opt_text = "\n".join(options)
                prompt = (f"{localized_memory}\n\nQuestion: {question}\n"
                          f"Options:\n{opt_text}\n\nThe best answer is:")
                raw_response = simple.infer(prompt)
                pred = extract_choice_letter(raw_response)
            else:
                pred = None
                raw_response = ""

        correct = self.adapter.check_correct(pred, answer)

        return {
            "pred": pred,
            "answer": answer,
            "correct": correct,
            "method": f"routed_{route_evidence.get('track', 'none')}",
            "route_evidence": route_evidence,
            "target_intervals": [[round(s, 3), round(e, 3)] for s, e in target_intervals] if target_intervals else [],
            "n_frames": len(frame_seconds),
            "raw_response": raw_response,
        }
