# src/smartuibot/vision/ocr/engine.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from smartuibot.core.types import Image


@runtime_checkable
class OcrEngine(Protocol):
    def recognize(self, image: Image) -> tuple[str, float]:
        """Return (text, confidence in [0,1]) for one cropped box image.
        ('', 0.0) when nothing is read."""
