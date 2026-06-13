# src/smartuibot/vision/ocr/paddle.py
from __future__ import annotations

from typing import Any

from smartuibot.core.types import Image


class PaddleOcrEngine:
    """Real OCR via PaddleOCR. `paddleocr` is imported lazily in __init__ so
    the heavy dependency never loads for headless tests / disabled OCR."""

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR  # lazy: heavy optional dependency

        # enable_mkldnn=False avoids a paddlepaddle 3.x oneDNN CPU bug
        # (ConvertPirAttribute2RuntimeAttribute) that crashes inference.
        self._ocr: Any = PaddleOCR(lang=lang, enable_mkldnn=False)

    def recognize(self, image: Image) -> tuple[str, float]:
        # PaddleOCR 3.x: predict() returns a list of page-dict results with
        # `rec_texts` / `rec_scores`. PaddleOCR <=2.x had a different shape;
        # we no longer support it because pyproject pins paddleocr>=2.7 and
        # the 3.x line is what installs today.
        result = self._ocr.predict(image)
        if not result:
            return "", 0.0
        texts: list[str] = []
        confs: list[float] = []
        for page in result:
            rec_texts = page.get("rec_texts") or []
            rec_scores = page.get("rec_scores") or []
            for t, c in zip(rec_texts, rec_scores, strict=False):
                texts.append(str(t))
                confs.append(float(c))
        if not texts:
            return "", 0.0
        return " ".join(texts), min(confs)
