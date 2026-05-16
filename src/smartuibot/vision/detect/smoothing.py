# src/smartuibot/vision/detect/smoothing.py
from __future__ import annotations

from smartuibot.core.types import Detection


class SmoothingFilter:
    """Keeps a detection visible for `persist_frames` extra frames after it
    stops being reported, reducing flicker."""

    def __init__(self, persist_frames: int = 3) -> None:
        self._persist = persist_frames
        self._ages: dict[str, int] = {}
        self._last: dict[str, Detection] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        present = {d.label: d for d in detections}
        for label, det in present.items():
            self._ages[label] = 0
            self._last[label] = det

        out: list[Detection] = []
        for label in list(self._ages.keys()):
            if label in present:
                out.append(present[label])
                continue
            self._ages[label] += 1
            if self._ages[label] > self._persist:
                del self._ages[label]
                del self._last[label]
            else:
                out.append(self._last[label])
        return out
