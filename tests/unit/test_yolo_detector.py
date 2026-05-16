import numpy as np
import pytest

from smartuibot.core.types import Detection
from smartuibot.vision.detect.yolo import Yolo11Detector, _results_to_detections


def test_results_to_detections_pure_mapping() -> None:
    class _Box:
        def __init__(self) -> None:
            self.xyxy: list[np.ndarray] = [np.array([1.0, 2.0, 11.0, 12.0])]
            self.conf: list[np.ndarray] = [np.array(0.81)]
            self.cls: list[np.ndarray] = [np.array(0.0)]

    class _Res:
        names: dict[int, str] = {0: "person"}
        boxes: list[_Box] = [_Box()]

    dets = _results_to_detections([_Res()], conf_threshold=0.5)
    assert len(dets) == 1
    assert dets[0].label == "person"
    assert dets[0].confidence == pytest.approx(0.81)
    assert dets[0].class_id == 0
    assert dets[0].x1 == 1
    assert dets[0].y1 == 2
    assert dets[0].x2 == 11
    assert dets[0].y2 == 12


def test_results_to_detections_filters_low_confidence() -> None:
    class _Box:
        def __init__(self) -> None:
            self.xyxy: list[np.ndarray] = [np.array([0.0, 0.0, 1.0, 1.0])]
            self.conf: list[np.ndarray] = [np.array(0.10)]
            self.cls: list[np.ndarray] = [np.array(0.0)]

    class _Res:
        names: dict[int, str] = {0: "person"}
        boxes: list[_Box] = [_Box()]

    assert _results_to_detections([_Res()], conf_threshold=0.5) == []


@pytest.mark.model
def test_real_yolo_infers_schema() -> None:
    det = Yolo11Detector(model_path="yolo11n.pt", device="cpu", confidence=0.25)
    out = det.infer(np.zeros((320, 320, 3), dtype=np.uint8))
    assert isinstance(out, list)
    for d in out:
        assert isinstance(d, Detection)
