"""
Frame Loader — 비디오 프레임 로딩 모듈

- TargetedFrameLoader: 지정 시간 구간에서 프레임 추출
- UniformFrameLoader: 균등 간격으로 프레임 추출
"""

import numpy as np
from decord import VideoReader, cpu


class TargetedFrameLoader:
    """지정된 시간 구간들에서만 프레임을 로드.

    구간 길이에 비례하여 max_frames를 배분.
    인접 구간은 자동 병합 (gap < merge_gap_sec).
    """

    def __init__(self, max_frames: int = 32, merge_gap_sec: float = 1.0):
        self.max_frames = max_frames
        self.merge_gap_sec = merge_gap_sec

    def load(self, video_path: str, intervals: list[tuple],
             max_frames: int | None = None) -> tuple[np.ndarray | None, list[float]]:
        """시간 구간 기반 프레임 로딩.

        Args:
            video_path: 비디오 파일 경로
            intervals: [(start_sec, end_sec), ...] 리스트
            max_frames: override max frames (None이면 self.max_frames 사용)

        Returns:
            (frames_ndarray, frame_seconds_list) 또는 (None, [])
        """
        if not intervals:
            return None, []

        max_frames = max_frames or self.max_frames

        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        vid_frames = len(vr)

        # Merge overlapping/adjacent intervals
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + self.merge_gap_sec:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        # Distribute frames proportionally
        durations = [max(0.1, e - s) for s, e in merged]
        total_dur = sum(durations)
        n_per_interval = [max(1, int(round(max_frames * (d / total_dur)))) for d in durations]

        while sum(n_per_interval) > max_frames:
            idx = n_per_interval.index(max(n_per_interval))
            n_per_interval[idx] -= 1
        while sum(n_per_interval) < max_frames:
            idx = durations.index(max(durations))
            n_per_interval[idx] += 1

        # Generate frame indices per interval
        all_idxs = []
        for (s, e), n in zip(merged, n_per_interval):
            if n <= 0:
                continue
            start_f = max(0, int(s * fps))
            end_f = min(vid_frames - 1, int(e * fps))
            if end_f <= start_f:
                end_f = start_f + 1
            idxs = np.linspace(start_f, end_f, n).astype(int)
            idxs = np.clip(idxs, 0, vid_frames - 1)
            all_idxs.append(idxs)

        if not all_idxs:
            return None, []

        frame_idxs = np.concatenate(all_idxs)
        # Deduplicate preserving order
        _, unique_mask = np.unique(frame_idxs, return_index=True)
        frame_idxs = frame_idxs[np.sort(unique_mask)]

        frames = vr.get_batch(frame_idxs).asnumpy()
        frame_seconds = (frame_idxs / fps).tolist()

        return frames, frame_seconds


    def load_per_interval(
        self, video_path: str, intervals: list[tuple],
        frames_per_interval: int = 3,
    ) -> list[dict]:
        """구간별 분리된 프레임 로딩 (Scout용).

        TargetedFrameLoader.load()와 달리 구간을 merge하지 않고
        각 구간별로 독립적으로 프레임을 추출하여 분리 반환.

        Args:
            video_path: 비디오 파일 경로
            intervals: [(start_sec, end_sec), ...] 리스트
            frames_per_interval: 구간당 프레임 수

        Returns:
            [
                {
                    "interval": (start_sec, end_sec),
                    "frames": np.ndarray (N, H, W, 3),
                    "frame_seconds": [float, ...],
                },
                ...
            ]
        """
        if not intervals:
            return []

        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        vid_frames = len(vr)

        results = []
        for s, e in intervals:
            start_f = max(0, int(s * fps))
            end_f = min(vid_frames - 1, int(e * fps))
            if end_f <= start_f:
                end_f = start_f + 1

            n = min(frames_per_interval, end_f - start_f + 1)
            idxs = np.linspace(start_f, end_f, n).astype(int)
            idxs = np.clip(idxs, 0, vid_frames - 1)

            frames = vr.get_batch(idxs).asnumpy()
            frame_secs = (idxs / fps).tolist()

            results.append({
                "interval": (s, e),
                "frames": frames,
                "frame_seconds": frame_secs,
            })

        return results


class UniformFrameLoader:
    """비디오 전체 또는 구간에서 균등 간격으로 프레임 추출."""

    def __init__(self, n_frames: int = 32):
        self.n_frames = n_frames

    def load(self, video_path: str, start_sec: float = 0.0,
             end_sec: float | None = None,
             n_frames: int | None = None) -> tuple[np.ndarray | None, list[float]]:
        """균등 간격 프레임 로딩.

        Args:
            video_path: 비디오 파일 경로
            start_sec: 시작 시간 (초)
            end_sec: 끝 시간 (초). None이면 비디오 끝까지.
            n_frames: override (None이면 self.n_frames 사용)

        Returns:
            (frames_ndarray, frame_seconds_list) 또는 (None, [])
        """
        n_frames = n_frames or self.n_frames

        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        vid_frames = len(vr)

        start_f = max(0, int(start_sec * fps))
        end_f = min(vid_frames - 1, int(end_sec * fps)) if end_sec else vid_frames - 1

        if end_f <= start_f:
            return None, []

        frame_idxs = np.linspace(start_f, end_f, n_frames).astype(int)
        frame_idxs = np.clip(frame_idxs, 0, vid_frames - 1)

        frames = vr.get_batch(frame_idxs).asnumpy()
        frame_seconds = (frame_idxs / fps).tolist()

        return frames, frame_seconds
