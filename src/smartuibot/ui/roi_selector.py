# src/smartuibot/ui/roi_selector.py
from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import (
    QColor,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QScreen,
    QShowEvent,
)
from PyQt6.QtWidgets import QWidget

from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import Monitor

_MIN_SELECTION_PX = 8


def selection_to_roi(
    origin: QPoint,
    current: QPoint,
    scale_x: float,
    scale_y: float,
    monitor_size: tuple[int, int],
    monitor: int,
) -> ROI | None:
    """Convert a drag (two Qt logical-point corners) into a capture-pixel ROI.

    `scale_*` is the ratio between the capture backend's monitor size and the
    Qt screen's logical size (NOT `devicePixelRatio` — mss on macOS already
    works in that pixel space, so DPR would double-count). The result is
    clamped to the monitor so an over-drag can never request an off-screen
    region. Returns None for a sub-minimum drag (treated as a cancel)."""
    left = min(origin.x(), current.x())
    top = min(origin.y(), current.y())
    width = abs(origin.x() - current.x())
    height = abs(origin.y() - current.y())
    mon_w, mon_h = monitor_size
    px = round(left * scale_x)
    py = round(top * scale_y)
    x = max(0, min(px, mon_w))
    y = max(0, min(py, mon_h))
    right = max(0, min(px + round(width * scale_x), mon_w))
    bottom = max(0, min(py + round(height * scale_y), mon_h))
    cw = right - x
    ch = bottom - y
    if cw < _MIN_SELECTION_PX or ch < _MIN_SELECTION_PX:
        return None
    return ROI(monitor=monitor, x=x, y=y, width=cw, height=ch)


def _resolve_screen(mon: Monitor) -> QScreen | None:
    """Pick the QScreen for the configured monitor. Match by logical size or
    by physical size (logical * DPR) — mss may report either depending on the
    platform — then fall back to index, then primary."""
    screens = QGuiApplication.screens()
    if not screens:
        return None
    for s in screens:
        g = s.geometry()
        dpr = s.devicePixelRatio()
        if (g.width() == mon.width and g.height() == mon.height) or (
            round(g.width() * dpr) == mon.width
            and round(g.height() * dpr) == mon.height
        ):
            return s
    idx = mon.index - 1
    if 0 <= idx < len(screens):
        return screens[idx]
    return QGuiApplication.primaryScreen()


class ROISelectorOverlay(QWidget):
    """Fullscreen translucent overlay on the configured monitor: drag a
    rectangle (macOS-screenshot style), release to confirm, Esc to cancel."""

    def __init__(self, monitor: Monitor,
                 on_selected: Callable[[ROI], None]) -> None:
        super().__init__()
        self._mon = monitor
        self._on_selected = on_selected
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.35)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        screen = _resolve_screen(monitor)
        if screen is not None:
            self.setScreen(screen)
            self.setGeometry(screen.geometry())

    def showEvent(self, event: QShowEvent | None) -> None:
        # Shown as a normal (non-fullscreen) frameless window so macOS keeps it
        # translucent over the live desktop; force it to the front and take
        # keyboard focus so Esc cancels and the drag is captured.
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def _scale(self) -> tuple[float, float]:
        """capture-monitor pixels per Qt logical point, measured (not DPR)."""
        s = self.screen()
        if s is None:
            return 1.0, 1.0
        g = s.geometry()
        sx = self._mon.width / g.width() if g.width() else 1.0
        sy = self._mon.height / g.height() if g.height() else 1.0
        return sx, sy

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
            sx, sy = self._scale()
            roi = selection_to_roi(
                self._origin, event.position().toPoint(), sx, sy,
                (self._mon.width, self._mon.height), self._mon.index)
            if roi is not None:
                self._on_selected(roi)
        self.close()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if event is None:
            return
        if self._origin is None or self._current is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 200, 0), 2))
        painter.drawRect(QRect(self._origin, self._current))
