from smartuibot.core.types import Detection
from smartuibot.vision.detect.smoothing import SmoothingFilter


def _d(label: str, x: int = 0, y: int = 0, size: int = 5) -> Detection:
    return Detection(
        label=label, confidence=0.9, class_id=0,
        x1=x, y1=y, x2=x + size, y2=y + size,
    )


def test_detection_persists_for_n_frames_after_disappearing() -> None:
    f = SmoothingFilter(persist_frames=2)
    assert [d.label for d in f.update([_d("a")])] == ["a"]
    assert [d.label for d in f.update([])] == ["a"]      # frame 1 missing -> kept
    assert [d.label for d in f.update([])] == ["a"]      # frame 2 missing -> kept
    assert [d.label for d in f.update([])] == []         # frame 3 missing -> dropped


def test_reappearing_detection_resets_persistence() -> None:
    f = SmoothingFilter(persist_frames=1)
    f.update([_d("a")])
    f.update([])
    assert [d.label for d in f.update([_d("a")])] == ["a"]


def test_multiple_detections_with_same_label_are_all_kept() -> None:
    f = SmoothingFilter(persist_frames=1)
    dets = [_d("button", x=0), _d("button", x=100), _d("button", x=200)]
    out = f.update(dets)
    assert len(out) == 3
    assert {(d.x1, d.y1) for d in out} == {(0, 0), (100, 0), (200, 0)}


def test_per_instance_persistence_for_same_label() -> None:
    f = SmoothingFilter(persist_frames=1)
    a = _d("button", x=0)
    b = _d("button", x=100)
    f.update([a, b])
    # Only `a` reappears; `b` should ghost for 1 frame then drop.
    out1 = f.update([a])
    assert {(d.x1, d.y1) for d in out1} == {(0, 0), (100, 0)}
    out2 = f.update([a])
    assert {(d.x1, d.y1) for d in out2} == {(0, 0)}
