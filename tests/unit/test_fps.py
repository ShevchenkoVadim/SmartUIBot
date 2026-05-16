# tests/unit/test_fps.py
from smartuibot.core.fps import FpsMeter


def test_fps_zero_before_two_ticks() -> None:
    m = FpsMeter(window=10)
    assert m.fps == 0.0
    m.tick(now=100.0)
    assert m.fps == 0.0


def test_fps_computed_over_window() -> None:
    m = FpsMeter(window=5)
    for i in range(5):
        m.tick(now=float(i) * 0.5)  # 2 fps spacing
    assert round(m.fps, 1) == 2.0


def test_window_drops_old_samples() -> None:
    m = FpsMeter(window=3)
    for i in range(10):
        m.tick(now=float(i))  # 1 fps spacing
    assert round(m.fps, 1) == 1.0
