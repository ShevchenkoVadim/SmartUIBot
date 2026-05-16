# src/smartuibot/ui/roi_selector.py
from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QWidget

from smartuibot.core.types import ROI


def rect_to_roi(p1: QPoint, p2: QPoint, monitor: int) -> ROI:
    x = min(p1.x(), p2.x())
    y = min(p1.y(), p2.y())
    w = max(1, abs(p1.x() - p2.x()))
    h = max(1, abs(p1.y() - p2.y()))
    return ROI(monitor=monitor, x=x, y=y, width=w, height=h)


class ROISelectorOverlay(QWidget):
    """Fullscreen translucent overlay; drag a rectangle, release to confirm."""

    def __init__(self, monitor: int, on_selected: Callable[[ROI], None]) -> None:
        super().__init__()
        self._monitor = monitor
        self._on_selected = on_selected
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.35)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        self._origin = event.position().toPoint()
        self._current = event.position().toPoint()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        self._current = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self._origin is not None:
            roi = rect_to_roi(self._origin, event.position().toPoint(), self._monitor)
            self._on_selected(roi)
        self.close()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if event is None:
            return
        if self._origin is None or self._current is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 200, 0), 2))
        painter.drawRect(QRect(self._origin, self._current))
