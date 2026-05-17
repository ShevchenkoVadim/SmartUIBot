# tests/fakes/ocr.py
from __future__ import annotations

from smartuibot.core.types import Image


class FakeOcrEngine:
    """Deterministic OCR for headless tests: returns a fixed (text, conf)."""

    def __init__(self, text: str = "", confidence: float = 0.0) -> None:
        self._text = text
        self._conf = confidence

    def recognize(self, image: Image) -> tuple[str, float]:
        return self._text, self._conf
