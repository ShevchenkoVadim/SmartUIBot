# src/smartuibot/core/container.py
from __future__ import annotations

import random
from pathlib import Path

from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.registry import load_behaviors
from smartuibot.ai.service import DecisionService
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.core.config import AppConfig
from smartuibot.core.event_bus import EventBus
from smartuibot.core.logging_setup import setup_logging
from smartuibot.core.types import ROI
from smartuibot.core.watchdog import Watchdog
from smartuibot.input.backend import InputBackend
from smartuibot.input.motion import MotionParams
from smartuibot.input.service import ActionService
from smartuibot.vision.capture.backend import CaptureBackend
from smartuibot.vision.capture.service import CaptureService
from smartuibot.vision.detect.detector import Detector
from smartuibot.vision.detect.service import DetectionService
from smartuibot.vision.ocr.service import OcrService


class AppContainer:
    """Owns and wires every singleton. Platform adapters are injected so the
    whole pipeline can run headless with fakes."""

    def __init__(
        self,
        config: AppConfig,
        roi: ROI,
        capture_backend: CaptureBackend,
        detector: Detector,
        input_backend: InputBackend | None = None,
    ) -> None:
        if input_backend is None:
            from smartuibot.input.backend import NoOpInputBackend
            input_backend = NoOpInputBackend()
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
        self.ocr = OcrService(engine=None, bus=self.bus, labels=frozenset(),
                              min_confidence=0.0, enabled=False)
        self.mode = ModeFSM()
        if config.input.start_armed:
            self.mode.arm()
        ic = config.input
        dc = config.decision
        rng = random.Random(dc.rng_seed)
        behaviors = load_behaviors(Path(config.behaviors_path))
        policy = UtilityPolicy(
            behaviors, tick_hz=dc.tick_hz,
            anti_loop_window=dc.anti_loop_window,
            anti_loop_max_repeats=dc.anti_loop_max_repeats,
            hesitation_prob=dc.hesitation_prob, rng=rng)
        self.decision = DecisionService(
            bus=self.bus, policy=policy, tracker=WorldStateTracker(),
            mode=self.mode, tick_hz=dc.tick_hz)
        self.action = ActionService(
            bus=self.bus, backend=input_backend, mode=self.mode,
            motion=MotionParams(
                move_steps=ic.move_steps, jitter_px=ic.jitter_px,
                reaction_min_s=ic.reaction_min_s, reaction_max_s=ic.reaction_max_s,
                keystroke_min_s=ic.keystroke_min_s, keystroke_max_s=ic.keystroke_max_s,
                overshoot_prob=ic.overshoot_prob),
            max_actions_per_second=ic.max_actions_per_second,
            roi_confine=ic.roi_confine, rng=random.Random(dc.rng_seed + 1))
        self.watchdog = Watchdog(
            [self.capture, self.detection, self.ocr, self.decision, self.action],
            bus=self.bus)

    def start(self) -> None:
        self.action.start()
        self.decision.start()
        self.ocr.start()
        self.detection.start()
        self.capture.start()
        self.watchdog.start()

    def stop(self) -> None:
        self.watchdog.stop()
        self.mode.disarm()
        self.capture.stop()
        self.detection.stop()
        self.ocr.stop()
        self.decision.stop()
        self.action.stop()
