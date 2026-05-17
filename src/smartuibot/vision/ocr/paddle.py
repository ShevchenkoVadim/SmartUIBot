# src/smartuibot/vision/ocr/paddle.py
from __future__ import annotations

from typing import Any

from smartuibot.core.types import Image


class PaddleOcrEngine:
    """Real OCR via PaddleOCR. `paddleocr` is imported lazily in __init__ so
    the heavy dependency never loads for headless tests / disabled OCR."""

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR  # lazy: heavy optional dependency

        self._ocr: Any = PaddleOCR(lang=lang, show_log=False, use_gpu=False)

    def recognize(self, image: Image) -> tuple[str, float]:
        result = self._ocr.ocr(image, cls=False)
        if not result or not result[0]:
            return "", 0.0
        texts: list[str] = []
        confs: list[float] = []
        for line in result[0]:
            txt, conf = line[1]
            texts.append(str(txt))
            confs.append(float(conf))
        if not texts:
            return "", 0.0
        return " ".join(texts), min(confs)
