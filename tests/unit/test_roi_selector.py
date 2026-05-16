import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint  # noqa: E402

from smartuibot.core.types import ROI  # noqa: E402
from smartuibot.ui.roi_selector import rect_to_roi  # noqa: E402


def test_rect_to_roi_normalizes_drag_direction() -> None:
    # drag from bottom-right to top-left still yields a positive ROI
    roi = rect_to_roi(QPoint(100, 90), QPoint(20, 10), monitor=1)
    assert roi == ROI(monitor=1, x=20, y=10, width=80, height=80)


def test_rect_to_roi_minimum_size_enforced() -> None:
    roi = rect_to_roi(QPoint(5, 5), QPoint(6, 6), monitor=2)
    assert roi.width >= 1 and roi.height >= 1 and roi.monitor == 2
