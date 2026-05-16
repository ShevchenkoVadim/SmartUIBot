# src/smartuibot/vision/capture/service.py
from __future__ import annotations

import threading
import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import FpsTick, FrameCaptured
from smartuibot.core.fps import FpsMeter
from smartuibot.core.service import Service
from smartuibot.core.types import ROI, Frame
from smartuibot.vision.capture.backend import CaptureBackend


class CaptureService(Service):
    def __init__(
        self,
        backend: CaptureBackend,
        bus: EventBus,
        roi: ROI,
        target_fps: int = 60,
    ) -> None:
        super().__init__(name="capture", bus=bus)
        self._backend = backend
        self._roi = roi
        self._roi_lock = threading.Lock()
        self._period = 1.0 / float(target_fps)
        self._seq = 0
        self._fps = FpsMeter(window=30)

    def set_roi(self, roi: ROI) -> None:
        with self._roi_lock:
            self._roi = roi

    def run_once(self) -> None:
        start = time.monotonic()
        with self._roi_lock:
            roi = self._roi
        image = self._backend.grab(roi)
        self._seq += 1
        frame = Frame(image=image, timestamp=time.monotonic(), seq=self._seq, roi=roi)
        self._bus.publish(FrameCaptured(frame=frame))
        self._fps.tick()
        self._bus.publish(FpsTick(name="capture", fps=self._fps.fps))
        elapsed = time.monotonic() - start
        if elapsed < self._period:
            time.sleep(self._period - elapsed)
