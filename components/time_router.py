from __future__ import annotations

"""
Time Router — 시간 기반 라우팅 모듈

Track A: 질문/선택지에서 명시적 시간 추출 → Direct Indexing
Track B: Semantic search (QueryDecomposer + HierarchicalNavigator와 조합)
"""

import re
from components.time_utils import time_to_secs, secs_to_time_str


class TimeRouter:
    """질문에서 시간 정보를 추출하고 Track A/B 라우팅을 결정."""

    def extract_time_ranges(self, text: str) -> list[tuple[float, float]]:
        """텍스트에서 타임스탬프를 추출하여 초 단위 (start, end) 리스트로 반환.

        Priority 1: <TIME HH:MM:SS.ms video N> to/and <TIME HH:MM:SS.ms video N>  (range)
        Priority 2: HH:MM:SS - HH:MM:SS 또는 MM:SS ~ MM:SS  (range)
        Priority 3: Single <TIME HH:MM:SS.ms video N>  (point → zero-width range)
        Priority 4: Standalone HH:MM:SS.ms  (point → zero-width range)
        """
        ranges = []

        # Pattern 1: Paired <TIME> tags (range)
        tag_range_pattern = r"<TIME\s+([\d:\.]+)\s+[^>]+>\s*(?:to|and)\s*<TIME\s+([\d:\.]+)\s+[^>]+>"
        for start_str, end_str in re.findall(tag_range_pattern, text):
            ranges.append((time_to_secs(start_str), time_to_secs(end_str)))

        if ranges:
            return ranges

        # Pattern 2: Simple time range formats
        time_pattern = r"(?:(?:[0-9]{1,2}:)?(?:[0-9]{1,2}:[0-9]{1,2}(?:\.[0-9]+)?))"
        range_pattern = rf"({time_pattern})\s*(?:-|~|to)\s*({time_pattern})"
        for start_str, end_str in re.findall(range_pattern, text):
            ranges.append((time_to_secs(start_str), time_to_secs(end_str)))

        if ranges:
            return ranges

        # Pattern 3: Single <TIME> tag (point)
        single_tag_pattern = r"<TIME\s+([\d:\.]+)\s+[^>]*>"
        for time_str in re.findall(single_tag_pattern, text):
            t = time_to_secs(time_str)
            ranges.append((t, t))

        if ranges:
            return ranges

        # Pattern 4: Standalone timestamp (point)
        standalone_pattern = r"(?<!\d)(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)(?!\d)"
        for time_str in re.findall(standalone_pattern, text):
            t = time_to_secs(time_str)
            ranges.append((t, t))

        return ranges

    def parse_time_reference(self, time_ref_str: str) -> tuple[float, float] | None:
        """LVBench time_reference 필드 파싱.

        Args:
            time_ref_str: "00:15-00:19" 형태

        Returns:
            (start_sec, end_sec) or None
        """
        if not time_ref_str:
            return None
        parts = re.split(r"\s*-\s*", time_ref_str.strip())
        if len(parts) != 2:
            return None
        try:
            return (time_to_secs(parts[0]), time_to_secs(parts[1]))
        except Exception:
            return None

    def resolve_target_intervals(
        self, route_evidence: dict,
        temporal_divisor: float = 1.0,
        input_start_secs: float = 0.0
    ) -> list[tuple[float, float]]:
        """라우팅 결과를 절대 비디오 시간으로 변환.

        Track A: matched_time_ranges_abs → 그대로 사용
        Track B: selected_segment → 역변환 (abs = mem * divisor + offset)

        Args:
            route_evidence: {"track": "A"|"B", ...}
            temporal_divisor: memory time → absolute time 변환 비율
            input_start_secs: 비디오 시작 offset

        Returns:
            list of (start_sec, end_sec) absolute time intervals
        """
        intervals = []
        track = route_evidence.get("track", "")

        if track == "A":
            for pair in route_evidence.get("matched_time_ranges_abs", []):
                intervals.append((float(pair[0]), float(pair[1])))
        elif track == "B":
            segment_str = route_evidence.get("selected_segment", "")
            parsed = self.extract_time_ranges(segment_str)
            for mem_start, mem_end in parsed:
                abs_start = mem_start * temporal_divisor + input_start_secs
                abs_end = mem_end * temporal_divisor + input_start_secs
                intervals.append((abs_start, abs_end))

        return intervals

    def get_hierarchy_path(self, raw_memory_dict: dict,
                           target_ranges: list[tuple]) -> list[dict]:
        """Track A에서 매칭된 substep들의 계층 경로 반환."""
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
                        sub_start = time_to_secs(sub["start"])
                        sub_end = time_to_secs(sub["end"])
                        for tr in target_ranges:
                            if max(tr[0], sub_start) < min(tr[1], sub_end):
                                paths.append({
                                    "goal": goal,
                                    "step": step_summary,
                                    "substep": sub.get("substep", ""),
                                    "time": f"{sub['start']} ~ {sub['end']}",
                                    "matched_range": (
                                        f"{secs_to_time_str(tr[0])} ~ {secs_to_time_str(tr[1])}"
                                    ),
                                })
                                break
        return paths
