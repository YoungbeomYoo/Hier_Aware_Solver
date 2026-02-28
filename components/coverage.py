"""
Coverage Analyzer — Time reference coverage 분석

검색한 leaf/frame이 target time reference를 얼마나 커버하는지 계산.
"""

from components.time_router import TimeRouter


class CoverageAnalyzer:
    """시간 구간 기반 coverage 분석.

    Coverage ratio: target 시간 중 커버된 비율
    Precision: 로딩된 구간 중 target과 겹치는 비율
    """

    def __init__(self):
        self._time_router = TimeRouter()

    def compute(self, intervals: list[tuple[float, float]],
                time_ref_str: str) -> dict:
        """Coverage 계산.

        Args:
            intervals: [(start_sec, end_sec), ...] 검색/로딩된 구간들
            time_ref_str: "00:15-00:19" 형태의 target time reference

        Returns:
            {
                "time_ref_seconds": [start, end] or None,
                "coverage_ratio": float (0-1),
                "precision": float (0-1),
                "hit": bool,
                "active_segments": list
            }
        """
        time_ref = self._time_router.parse_time_reference(time_ref_str)
        if time_ref is None:
            return {
                "time_ref_seconds": None,
                "coverage_ratio": None,
                "precision": None,
                "hit": None,
            }

        ref_start, ref_end = time_ref
        if ref_end < ref_start:
            ref_end = ref_start

        if not intervals:
            return {
                "time_ref_seconds": [ref_start, ref_end],
                "active_segments": [],
                "coverage_ratio": 0.0,
                "precision": 0.0,
                "hit": False,
            }

        ref_duration = max(ref_end - ref_start, 1.0)
        overlap_duration = 0.0
        hit = False

        for seg_start, seg_end in intervals:
            overlap_start = max(ref_start, seg_start)
            overlap_end = min(ref_end, seg_end)
            if overlap_start < overlap_end:
                overlap_duration += overlap_end - overlap_start
                hit = True
            elif ref_start == ref_end and seg_start <= ref_start <= seg_end:
                hit = True
                overlap_duration = 1.0

        coverage_ratio = min(overlap_duration / ref_duration, 1.0)

        total_active_duration = sum(max(0, e - s) for s, e in intervals)
        precision = min(overlap_duration / total_active_duration, 1.0) if total_active_duration > 0 else 0.0

        return {
            "time_ref_seconds": [ref_start, ref_end],
            "active_segments": intervals[:20],
            "coverage_ratio": round(coverage_ratio, 4),
            "precision": round(precision, 4),
            "hit": hit,
        }

    def compute_from_entries(self, leaf_entries: list[dict],
                             time_ref_str: str) -> dict:
        """Leaf entries에서 intervals를 추출하여 coverage 계산."""
        intervals = []
        for entry in leaf_entries:
            leaf = entry["leaf"]
            if "start_time" in leaf:
                intervals.append((float(leaf["start_time"]), float(leaf["end_time"])))
        intervals.sort()
        return self.compute(intervals, time_ref_str)
