import numpy as np
import pytest


def test_paddle_engine_module_imports_without_paddle() -> None:
    # Module import must NOT import paddleocr (lazy in __init__).
    from smartuibot.vision.ocr.paddle import PaddleOcrEngine

    assert PaddleOcrEngine is not None


@pytest.mark.ocr
def test_paddle_engine_reads_text() -> None:
    from smartuibot.vision.ocr.paddle import PaddleOcrEngine

    eng = PaddleOcrEngine(lang="en")
    img = np.full((40, 160, 3), 255, dtype=np.uint8)  # blank -> '' is fine
    text, conf = eng.recognize(img)
    assert isinstance(text, str) and 0.0 <= conf <= 1.0
