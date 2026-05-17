# tests/unit/test_types.py
import numpy as np
import pytest

from smartuibot.core.types import ROI, Detection, Frame


def test_roi_roundtrip_dict() -> None:
    roi = ROI(monitor=1, x=10, y=20, width=300, height=200)
    assert ROI.from_dict(roi.as_dict()) == roi


def test_roi_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        ROI(monitor=1, x=0, y=0, width=0, height=100)


def test_frame_carries_image_and_metadata() -> None:
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    roi = ROI(monitor=1, x=0, y=0, width=5, height=4)
    f = Frame(image=img, timestamp=1.0, seq=7, roi=roi)
    assert f.seq == 7 and f.image.shape == (4, 5, 3)


def test_detection_box_area_and_validation() -> None:
    d = Detection(label="x", confidence=0.9, class_id=0, x1=0, y1=0, x2=10, y2=4)
    assert d.area == 40
    with pytest.raises(ValueError):
        Detection(label="x", confidence=1.5, class_id=0, x1=0, y1=0, x2=1, y2=1)


def test_detection_defaults_have_no_text() -> None:
    from smartuibot.core.types import Detection

    d = Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2, y2=2)
    assert d.text is None
    assert d.text_confidence == 0.0


def test_detection_with_text_and_validation() -> None:
    import pytest

    from smartuibot.core.types import Detection

    d = Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2,
                  y2=2, text="Close", text_confidence=0.9)
    assert d.text == "Close" and d.text_confidence == 0.9
    with pytest.raises(ValueError):
        Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2,
                  y2=2, text="x", text_confidence=1.5)
