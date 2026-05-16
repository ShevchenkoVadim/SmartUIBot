# src/smartuibot/vision/detect/detector.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from smartuibot.core.types import Detection, Image


@runtime_checkable
class Detector(Protocol):
    def infer(self, image: Image) -> list[Detection]: ...

    def reload(self, model_path: str) -> None: ...
