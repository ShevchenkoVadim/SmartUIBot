# tests/integration/test_closed_loop.py
import random
import time

from smartuibot.ai.behavior import Behavior, BehaviorStep, Condition
from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.service import DecisionService
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ActionStarted
from smartuibot.core.types import ROI
from smartuibot.input.motion import MotionParams
from smartuibot.input.service import ActionService
from smartuibot.vision.capture.service import CaptureService
from smartuibot.vision.detect.service import DetectionService
from smartuibot.vision.ocr.service import OcrService
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector
from tests.fakes.input import RecordingInputBackend

_ROI = ROI(monitor=1, x=0, y=0, width=16, height=16)


def _wire(
    mode: ModeFSM, backend: RecordingInputBackend
) -> tuple[EventBus, CaptureService, DetectionService, OcrService,
           DecisionService, ActionService]:
    bus = EventBus()
    behaviors = (Behavior(name="attack",
                          condition=Condition(labels=frozenset({"enemy"}),
                                               min_confidence=0.1),
                          base_utility=5.0, scale_by_confidence=False,
                          steps=(BehaviorStep(kind="click", target="detection"),)),)
    policy = UtilityPolicy(behaviors, tick_hz=50.0, anti_loop_window=5,
                           anti_loop_max_repeats=99, hesitation_prob=0.0,
                           rng=random.Random(1))
    decision = DecisionService(bus=bus, policy=policy,
                               tracker=WorldStateTracker(), mode=mode, tick_hz=50.0)
    action = ActionService(bus=bus, backend=backend, mode=mode,
                           motion=MotionParams(2, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
                           max_actions_per_second=1000.0, roi_confine=True,
                           rng=random.Random(2))
    ocr = OcrService(engine=None, bus=bus, labels=frozenset(),
                     min_confidence=0.0, enabled=False)
    detection = DetectionService(detector=FakeDetector(
        scripted=[[("enemy", 0.9, 4, 4, 12, 12)]] * 500), bus=bus,
        smoothing_frames=1, confidence=0.1)
    capture = CaptureService(backend=FakeCaptureBackend(), bus=bus, roi=_ROI,
                             target_fps=120)
    return bus, capture, detection, ocr, decision, action


def test_perceive_decide_act_closed_loop_headless() -> None:
    mode = ModeFSM()
    mode.arm()
    backend = RecordingInputBackend()
    bus, capture, detection, ocr, decision, action = _wire(mode, backend)
    started: list[ActionStarted] = []
    bus.subscribe(ActionStarted, started.append)
    for s in (action, decision, ocr, detection, capture):
        s.start()
    time.sleep(0.8)
    for s in (capture, detection, ocr, decision, action):
        s.stop()
    assert started, "closed loop did not produce an action"
    assert ("click", ("left",)) in backend.calls
    assert ("move_to", (8, 8)) in backend.calls


def test_disarm_halts_injection() -> None:
    mode = ModeFSM()
    mode.arm()
    backend = RecordingInputBackend()
    bus, capture, detection, ocr, decision, action = _wire(mode, backend)
    for s in (action, decision, ocr, detection, capture):
        s.start()
    time.sleep(0.2)
    mode.disarm()
    n_after_disarm = len(backend.calls)
    time.sleep(0.3)
    for s in (capture, detection, ocr, decision, action):
        s.stop()
    assert len(backend.calls) - n_after_disarm <= 3


def test_text_gated_behavior_fires_only_with_matching_ocr_text() -> None:
    import random as _random

    from smartuibot.ai.utility import UtilityPolicy
    from tests.fakes.ocr import FakeOcrEngine

    mode = ModeFSM()
    mode.arm()
    backend = RecordingInputBackend()
    bus = EventBus()
    behaviors = (Behavior(name="close",
                          condition=Condition(labels=frozenset({"button"}),
                                               min_confidence=0.1,
                                               text_any=frozenset({"close"})),
                          base_utility=5.0, scale_by_confidence=False,
                          steps=(BehaviorStep(kind="click",
                                              target="detection"),)),)
    policy = UtilityPolicy(behaviors, tick_hz=50.0, anti_loop_window=5,
                           anti_loop_max_repeats=99, hesitation_prob=0.0,
                           rng=_random.Random(1))
    decision = DecisionService(bus=bus, policy=policy,
                               tracker=WorldStateTracker(), mode=mode,
                               tick_hz=50.0)
    action = ActionService(bus=bus, backend=backend, mode=mode,
                           motion=MotionParams(2, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
                           max_actions_per_second=1000.0, roi_confine=True,
                           rng=_random.Random(2))
    ocr = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    detection = DetectionService(detector=FakeDetector(
        scripted=[[("button", 0.9, 4, 4, 12, 12)]] * 500), bus=bus,
        smoothing_frames=1, confidence=0.1)
    capture = CaptureService(backend=FakeCaptureBackend(), bus=bus, roi=_ROI,
                             target_fps=120)
    started: list[ActionStarted] = []
    bus.subscribe(ActionStarted, started.append)
    for s in (action, decision, ocr, detection, capture):
        s.start()
    time.sleep(0.8)
    for s in (capture, detection, ocr, decision, action):
        s.stop()
    assert started, "text-gated behavior did not fire with matching OCR text"
