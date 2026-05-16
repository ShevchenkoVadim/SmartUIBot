import numpy as np

from smartuibot.vision.detect.detector import Detector
from tests.fakes.detector import FakeDetector


def test_fake_detector_satisfies_protocol_and_returns_scripted() -> None:
    det: Detector = FakeDetector(scripted=[
        [("enemy", 0.9, 0, 0, 10, 10)],
        [],
    ])
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    first = det.infer(img)
    assert first[0].label == "enemy" and first[0].confidence == 0.9
    assert det.infer(img) == []
    det.reload("ignored.pt")  # must not raise
