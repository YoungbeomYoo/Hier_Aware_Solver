"""
Time Utilities — 시간 문자열 ↔ 초 변환
"""


def time_to_secs(t) -> float:
    """MM:SS, HH:MM:SS, 또는 숫자 문자열을 float 초로 변환."""
    try:
        parts = str(t).split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(t)
    except Exception:
        return 0.0


def secs_to_time_str(s: float) -> str:
    """float 초를 HH:MM:SS 또는 MM:SS 문자열로 변환."""
    s = float(s)
    h = int(s) // 3600
    m = (int(s) % 3600) // 60
    sec = s - h * 3600 - m * 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:05.2f}"
    return f"{m:02d}:{sec:04.1f}"
