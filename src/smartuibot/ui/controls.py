# src/smartuibot/ui/controls.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

from smartuibot.core.types import ROI

RoiSelectorFactory = Callable[[Callable[[ROI], None]], Any]


class UiController:
    """Pure glue between UI actions and the container. No Qt imports used in
    its logic, so every action is unit-testable with a fake container."""

    def __init__(
        self,
        container: Any,
        state_path: Path,
        save_roi: Callable[[Path, ROI], None],
    ) -> None:
        self.container = container
        self._state_path = state_path
        self._save_roi = save_roi
        self._paused = False
        self._roi_factory: RoiSelectorFactory | None = None
        self._overlay: Any | None = None

    def set_roi_selector_factory(self, factory: RoiSelectorFactory) -> None:
        self._roi_factory = factory

    def start(self) -> None:
        self.container.start()

    def stop(self) -> None:
        self.container.stop()

    def toggle_pause(self) -> str:
        self._paused = not self._paused
        for worker in (self.container.capture, self.container.detection):
            worker.pause() if self._paused else worker.resume()
        return "paused" if self._paused else "running"

    def set_confidence(self, value: float) -> None:
        self.container.detection.set_confidence(value)

    def reload_model(self, model_path: str) -> None:
        self.container.detection.reload_model(model_path)

    def apply_roi(self, roi: ROI) -> None:
        self.container.capture.set_roi(roi)
        self._save_roi(self._state_path, roi)

    def request_roi_selection(self) -> None:
        if self._roi_factory is None:
            return
        self._overlay = self._roi_factory(self.apply_roi)
        self._overlay.show()

    def is_armed(self) -> bool:
        return bool(self.container.mode.is_armed())

    def toggle_arm(self) -> str:
        if self.container.mode.is_armed():
            self.container.mode.disarm()
            return "disarmed"
        self.container.mode.arm()
        return "armed"


class ControlBar(QWidget):
    """Buttons + confidence slider wired to a UiController."""

    def __init__(self, controller: UiController, model_path: str) -> None:
        super().__init__()
        self._c = controller
        self._model_path = model_path

        start_btn = QPushButton("Start")
        stop_btn = QPushButton("Stop")
        self.pause_btn = QPushButton("Pause")
        roi_btn = QPushButton("Select ROI")
        reload_btn = QPushButton("Reload model")

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(35)

        start_btn.clicked.connect(self._c.start)
        stop_btn.clicked.connect(self._c.stop)
        self.pause_btn.clicked.connect(self._on_pause)
        roi_btn.clicked.connect(self._c.request_roi_selection)
        reload_btn.clicked.connect(lambda: self._c.reload_model(self._model_path))
        self.arm_btn = QPushButton("Arm")
        self.arm_btn.clicked.connect(self._on_arm)
        self.confidence_slider.valueChanged.connect(
            lambda v: self._c.set_confidence(v / 100.0))

        layout = QHBoxLayout(self)
        for w in (start_btn, stop_btn, self.pause_btn, self.arm_btn, roi_btn, reload_btn,
                  QLabel("conf"), self.confidence_slider):
            layout.addWidget(w)

    def _on_pause(self) -> None:
        state = self._c.toggle_pause()
        self.pause_btn.setText("Resume" if state == "paused" else "Pause")

    def _on_arm(self) -> None:
        state = self._c.toggle_arm()
        self.arm_btn.setText("Disarm" if state == "armed" else "Arm")
