# src/smartuibot/vision/detect/service.py
from __future__ import annotations

import threading

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsReady, FpsTick, FrameCaptured
from smartuibot.core.fps import FpsMeter
from smartuibot.core.latest_queue import LatestQueue
from smartuibot.core.service import Service
from smartuibot.core.types import Frame
from smartuibot.vision.detect.detector import Detector
from smartuibot.vision.detect.smoothing import SmoothingFilter


class DetectionService(Service):
    def __init__(
        self,
        detector: Detector,
        bus: EventBus,
        smoothing_frames: int = 3,
        confidence: float = 0.0,
    ) -> None:
        super().__init__(name="detection", bus=bus)
        self._detector = detector
        self._queue: LatestQueue[Frame] = LatestQueue()
        self._smoothing = SmoothingFilter(persist_frames=smoothing_frames)
        self._fps = FpsMeter(window=30)
        self._lock = threading.Lock()
        self._confidence = confidence
        bus.subscribe(FrameCaptured, self._on_frame)

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._confidence

    def set_confidence(self, value: float) -> None:
        """Runtime-adjustable post-inference threshold (called from UI thread)."""
        with self._lock:
            self._confidence = max(0.0, min(1.0, value))

    def reload_model(self, model_path: str) -> None:
        """Hot-reload the detector model (called from UI thread)."""
        self._detector.reload(model_path)

    def _on_frame(self, event: FrameCaptured) -> None:
        self._queue.put(event.frame)

    def run_once(self) -> None:
        frame = self._queue.get(timeout=0.1)
        if frame is None:
            return
        threshold = self.confidence
        raw = [d for d in self._detector.infer(frame.image) if d.confidence >= threshold]
        smoothed = self._smoothing.update(raw)
        self._bus.publish(DetectionsReady(frame=frame, detections=tuple(smoothed)))
        self._fps.tick()
        self._bus.publish(FpsTick(name="detection", fps=self._fps.fps))
