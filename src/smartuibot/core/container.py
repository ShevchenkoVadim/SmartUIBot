# src/smartuibot/core/container.py
from __future__ import annotations

from pathlib import Path

from smartuibot.core.config import AppConfig
from smartuibot.core.event_bus import EventBus
from smartuibot.core.logging_setup import setup_logging
from smartuibot.core.types import ROI
from smartuibot.core.watchdog import Watchdog
from smartuibot.vision.capture.backend import CaptureBackend
from smartuibot.vision.capture.service import CaptureService
from smartuibot.vision.detect.detector import Detector
from smartuibot.vision.detect.service import DetectionService


class AppContainer:
    """Owns and wires every singleton. Platform adapters are injected so the
    whole pipeline can run headless with fakes."""

    def __init__(
        self,
        config: AppConfig,
        roi: ROI,
        capture_backend: CaptureBackend,
        detector: Detector,
    ) -> None:
        self.config = config
        self.bus = EventBus()
        setup_logging(level=config.logging.level,
                      log_dir=Path(config.logging.dir), bus=self.bus)
        self.capture = CaptureService(
            backend=capture_backend, bus=self.bus, roi=roi,
            target_fps=config.capture.target_fps)
        self.detection = DetectionService(
            detector=detector, bus=self.bus,
            smoothing_frames=config.detection.smoothing_frames,
            confidence=config.detection.confidence)
        self.watchdog = Watchdog([self.capture, self.detection], bus=self.bus)

    def start(self) -> None:
        self.detection.start()
        self.capture.start()
        self.watchdog.start()

    def stop(self) -> None:
        self.watchdog.stop()
        self.capture.stop()
        self.detection.stop()
