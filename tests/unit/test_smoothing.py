from smartuibot.core.types import Detection
from smartuibot.vision.detect.smoothing import SmoothingFilter


def _d(label: str) -> Detection:
    return Detection(label=label, confidence=0.9, class_id=0, x1=0, y1=0, x2=5, y2=5)


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
