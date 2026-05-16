# tests/fakes/capture.py
from __future__ import annotations

import numpy as np

from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import Monitor


class FakeCaptureBackend:
    """Deterministic synthetic frames; lets the pipeline run headless."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._w = width
        self._h = height
        self._seq = 0

    def list_monitors(self) -> list[Monitor]:
        return [Monitor(index=1, x=0, y=0, width=self._w, height=self._h)]

    def grab(self, roi: ROI) -> np.ndarray:
        self._seq += 1
        img = np.full((roi.height, roi.width, 3), self._seq % 256, dtype=np.uint8)
        return img
