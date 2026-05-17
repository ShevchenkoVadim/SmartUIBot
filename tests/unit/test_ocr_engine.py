import numpy as np

from smartuibot.vision.ocr.engine import OcrEngine
from tests.fakes.ocr import FakeOcrEngine


def test_fake_ocr_engine_satisfies_protocol_and_returns_scripted() -> None:
    eng = FakeOcrEngine(text="Close", confidence=0.92)
    assert isinstance(eng, OcrEngine)
    text, conf = eng.recognize(np.zeros((4, 4, 3), dtype=np.uint8))
    assert text == "Close" and conf == 0.92


def test_fake_ocr_engine_default_is_empty() -> None:
    text, conf = FakeOcrEngine().recognize(np.zeros((2, 2, 3), dtype=np.uint8))
    assert text == "" and conf == 0.0
