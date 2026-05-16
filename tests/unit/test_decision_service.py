# tests/unit/test_decision_service.py
import random
import time

import numpy as np

from smartuibot.ai.behavior import Behavior, BehaviorStep, Condition
from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.service import DecisionService
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ActionRequested, DetectionsReady
from smartuibot.core.types import ROI, Detection, Frame

_ROI = ROI(monitor=1, x=0, y=0, width=8, height=8)


def _frame() -> Frame:
    return Frame(image=np.zeros((8, 8, 3), dtype=np.uint8),
                 timestamp=time.monotonic(), seq=1, roi=_ROI)


def _enemy() -> Detection:
    return Detection(label="enemy", confidence=0.9, class_id=0, x1=2, y1=2, x2=6, y2=6)


def _svc(bus: EventBus, mode: ModeFSM) -> DecisionService:
    behaviors = (Behavior(name="attack",
                          condition=Condition(labels=frozenset({"enemy"})),
                          base_utility=5.0,
                          steps=(BehaviorStep(kind="click", target="detection"),),
                          scale_by_confidence=False),)
    policy = UtilityPolicy(behaviors, tick_hz=50.0, anti_loop_window=5,
                           anti_loop_max_repeats=99, hesitation_prob=0.0,
                           rng=random.Random(1))
    return DecisionService(bus=bus, policy=policy, tracker=WorldStateTracker(),
                           mode=mode, tick_hz=50.0)


def test_no_actions_when_disarmed() -> None:
    bus = EventBus()
    out: list[ActionRequested] = []
    bus.subscribe(ActionRequested, out.append)
    mode = ModeFSM()  # DISARMED
    svc = _svc(bus, mode)
    svc.start()
    bus.publish(DetectionsReady(frame=_frame(), detections=(_enemy(),)))
    time.sleep(0.15)
    svc.stop()
    assert out == []


def test_emits_action_when_armed() -> None:
    bus = EventBus()
    out: list[ActionRequested] = []
    bus.subscribe(ActionRequested, out.append)
    mode = ModeFSM()
    mode.arm()
    svc = _svc(bus, mode)
    svc.start()
    bus.publish(DetectionsReady(frame=_frame(), detections=(_enemy(),)))
    time.sleep(0.2)
    svc.stop()
    assert out
    assert out[0].behavior_name == "attack"
    assert out[0].steps[0].kind == "click"
    assert (out[0].steps[0].x, out[0].steps[0].y) == (4, 4)  # centroid of (2,2,6,6)
