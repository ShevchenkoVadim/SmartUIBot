# src/smartuibot/vision/detect/detector.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from smartuibot.core.types import Detection


@runtime_checkable
class Detector(Protocol):
    def infer(self, image: np.ndarray) -> list[Detection]: ...

    def reload(self, model_path: str) -> None: ...
