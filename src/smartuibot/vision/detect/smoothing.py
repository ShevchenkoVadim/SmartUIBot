# src/smartuibot/vision/detect/smoothing.py
from __future__ import annotations

from smartuibot.core.types import Detection


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


class SmoothingFilter:
    """Each detection persists for `persist_frames` extra frames after it stops
    being reported, reducing flicker. Detections are matched across frames per
    instance (same label + IoU >= ``iou_threshold``), so multiple boxes sharing
    a label are tracked independently."""

    def __init__(self, persist_frames: int = 3, iou_threshold: float = 0.3) -> None:
        self._persist = persist_frames
        self._iou_threshold = iou_threshold
        # (detection, frames_since_seen)
        self._tracks: list[tuple[Detection, int]] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        matched: set[int] = set()
        next_tracks: list[tuple[Detection, int]] = []

        for det in detections:
            best_i = -1
            best_score = self._iou_threshold
            for i, (prev, _age) in enumerate(self._tracks):
                if i in matched or prev.label != det.label:
                    continue
                s = _iou(det, prev)
                if s >= best_score:
                    best_score = s
                    best_i = i
            if best_i >= 0:
                matched.add(best_i)
            next_tracks.append((det, 0))

        for i, (prev, age) in enumerate(self._tracks):
            if i in matched:
                continue
            new_age = age + 1
            if new_age <= self._persist:
                next_tracks.append((prev, new_age))

        self._tracks = next_tracks
        return [d for d, _ in self._tracks]