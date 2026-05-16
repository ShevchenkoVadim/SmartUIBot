import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint  # noqa: E402

from smartuibot.core.types import ROI  # noqa: E402
from smartuibot.ui.roi_selector import selection_to_roi  # noqa: E402

# scale = capture-monitor pixels / Qt-screen logical points.
_MON = (2048, 1280)


def test_selection_to_roi_scale_one_maps_one_to_one() -> None:
    # macOS Retina here: mss space == Qt logical space -> scale 1.0
    roi = selection_to_roi(QPoint(20, 10), QPoint(100, 90), 1.0, 1.0, _MON, 1)
    assert roi == ROI(monitor=1, x=20, y=10, width=80, height=80)


def test_selection_to_roi_scale_two_maps_to_capture_pixels() -> None:
    # e.g. Windows HiDPI: mss reports physical, Qt logical is half -> scale 2.0
    roi = selection_to_roi(QPoint(20, 10), QPoint(100, 90), 2.0, 2.0,
                           (4096, 2560), 1)
    assert roi == ROI(monitor=1, x=40, y=20, width=160, height=160)


def test_selection_to_roi_normalizes_drag_direction() -> None:
    a = selection_to_roi(QPoint(100, 90), QPoint(20, 10), 1.0, 1.0, _MON, 2)
    b = selection_to_roi(QPoint(20, 10), QPoint(100, 90), 1.0, 1.0, _MON, 2)
    assert a == b == ROI(monitor=2, x=20, y=10, width=80, height=80)


def test_selection_to_roi_below_minimum_returns_none() -> None:
    assert selection_to_roi(
        QPoint(5, 5), QPoint(9, 9), 1.0, 1.0, _MON, 1) is None


def test_selection_to_roi_minimum_uses_capture_pixels() -> None:
    # 5x5 logical * scale 2 = 10x10 capture px >= 8 -> valid
    roi = selection_to_roi(QPoint(0, 0), QPoint(5, 5), 2.0, 2.0,
                           (4096, 2560), 1)
    assert roi == ROI(monitor=1, x=0, y=0, width=10, height=10)


def test_selection_to_roi_clamps_to_monitor_bounds() -> None:
    # an over-drag past the monitor is clamped, never exceeds capture space
    roi = selection_to_roi(QPoint(-50, -50), QPoint(5000, 5000),
                           1.0, 1.0, _MON, 1)
    assert roi == ROI(monitor=1, x=0, y=0, width=2048, height=1280)
