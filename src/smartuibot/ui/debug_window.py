# src/smartuibot/ui/debug_window.py
from __future__ import annotations

import queue

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import (
    ActionAborted,
    ActionCompleted,
    ActionStarted,
    DetectionsReady,
    FpsTick,
    LogRecord,
    ModeChanged,
)
from smartuibot.core.types import Detection


def draw_boxes(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image.copy()
    for d in detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{d.label} {d.confidence:.2f}",
            (d.x1, max(0, d.y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return out


class DebugWindow(QWidget):
    """Read-only debug view. Events arrive on worker threads; they are queued
    and drained on the Qt thread via a timer (Qt rule: GUI on main thread)."""

    def __init__(self, bus: EventBus, preview_max_width: int = 960) -> None:
        super().__init__()
        self.setWindowTitle("SmartUIBot — Debug")
        self._max_w = preview_max_width
        self._events: queue.Queue[object] = queue.Queue()
        self._fps: dict[str, float] = {"capture": 0.0, "detection": 0.0}
        self._det_count = 0
        self._mode = "disarmed"
        self._actions: list[str] = []

        self._preview = QLabel("waiting for frames…")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mode_label = QLabel("mode: disarmed")
        self._action_list = QListWidget()
        self._fps_label = QLabel(self.fps_text())
        self._table = QListWidget()
        self._logs = QPlainTextEdit()
        self._logs.setReadOnly(True)

        right = QVBoxLayout()
        right.addWidget(self._mode_label)
        right.addWidget(self._fps_label)
        right.addWidget(QLabel("Detections:"))
        right.addWidget(self._table)
        right.addWidget(QLabel("Logs:"))
        right.addWidget(self._logs)
        right.addWidget(QLabel("Actions:"))
        right.addWidget(self._action_list)
        root = QHBoxLayout(self)
        root.addWidget(self._preview, stretch=3)
        right_box = QWidget()
        right_box.setLayout(right)
        root.addWidget(right_box, stretch=2)

        bus.subscribe(DetectionsReady, self._events.put)
        bus.subscribe(FpsTick, self._events.put)
        bus.subscribe(LogRecord, self._events.put)
        bus.subscribe(ModeChanged, self._events.put)
        bus.subscribe(ActionStarted, self._events.put)
        bus.subscribe(ActionCompleted, self._events.put)
        bus.subscribe(ActionAborted, self._events.put)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain)
        self._timer.start(33)  # ~30 Hz UI refresh

    def detection_count(self) -> int:
        return self._det_count

    def mode_text(self) -> str:
        return self._mode

    def action_count(self) -> int:
        return len(self._actions)

    def fps_text(self) -> str:
        return (
            f"capture {self._fps['capture']:.1f} fps"
            f" | detection {self._fps['detection']:.1f} fps"
        )

    def _drain(self) -> None:
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            if isinstance(ev, DetectionsReady):
                self._on_detections(ev)
            elif isinstance(ev, FpsTick):
                self._fps[ev.name] = ev.fps
                self._fps_label.setText(self.fps_text())
            elif isinstance(ev, LogRecord):
                self._logs.appendPlainText(
                    f"[{ev.level}] {ev.logger}: {ev.message}"
                )
            elif isinstance(ev, ModeChanged):
                self._mode = ev.mode
                self._mode_label.setText(f"mode: {ev.mode}")
            elif isinstance(ev, ActionStarted):
                self._actions.append(f"▶ {ev.behavior_name}")
                self._action_list.addItem(self._actions[-1])
            elif isinstance(ev, ActionCompleted):
                self._actions.append(f"✓ {ev.behavior_name}")
                self._action_list.addItem(self._actions[-1])
            elif isinstance(ev, ActionAborted):
                self._actions.append(f"✗ {ev.behavior_name} ({ev.reason})")
                self._action_list.addItem(self._actions[-1])

    def _on_detections(self, ev: DetectionsReady) -> None:
        self._det_count = len(ev.detections)
        self._table.clear()
        for d in ev.detections:
            self._table.addItem(
                f"{d.label}  {d.confidence:.2f}  [{d.x1},{d.y1},{d.x2},{d.y2}]"
            )
        rendered = draw_boxes(ev.frame.image, list(ev.detections))
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        h = rgb.shape[0]
        w = rgb.shape[1]
        qimg = QImage(
            bytes(rgb.data), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)
        if w > self._max_w:
            pix = pix.scaledToWidth(
                self._max_w, Qt.TransformationMode.SmoothTransformation
            )
        self._preview.setPixmap(pix)
