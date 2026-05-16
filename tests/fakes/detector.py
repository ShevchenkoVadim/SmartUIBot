# tests/fakes/detector.py
from __future__ import annotations

import numpy as np

from smartuibot.core.types import Detection

_Script = list[tuple[str, float, int, int, int, int]]


class FakeDetector:
    """Returns scripted detections per call, then []. Deterministic."""

    def __init__(self, scripted: list[_Script]) -> None:
        self._scripted = scripted
        self._i = 0

    def infer(self, image: np.ndarray) -> list[Detection]:
        if self._i >= len(self._scripted):
            return []
        spec = self._scripted[self._i]
        self._i += 1
        return [
            Detection(label=lbl, confidence=conf, class_id=0,
                      x1=x1, y1=y1, x2=x2, y2=y2)
            for (lbl, conf, x1, y1, x2, y2) in spec
        ]

    def reload(self, model_path: str) -> None:
        self._i = 0
