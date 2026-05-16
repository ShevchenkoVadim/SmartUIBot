# tests/unit/test_action_service.py
import random
import time

from smartuibot.ai.mode import ModeFSM
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import (
    ActionAborted,
    ActionCompleted,
    ActionRequested,
    ActionStarted,
)
from smartuibot.core.types import ROI, ActionStep
from smartuibot.input.motion import MotionParams
from smartuibot.input.service import ActionService
from tests.fakes.input import RecordingInputBackend

_ROI = ROI(monitor=1, x=100, y=50, width=40, height=30)


def _params() -> MotionParams:
    return MotionParams(move_steps=4, jitter_px=0, reaction_min_s=0.0,
                        reaction_max_s=0.0, keystroke_min_s=0.0,
                        keystroke_max_s=0.0, overshoot_prob=0.0)


def _svc(bus: EventBus, mode: ModeFSM, backend: RecordingInputBackend) -> ActionService:
    return ActionService(bus=bus, backend=backend, mode=mode, motion=_params(),
                          max_actions_per_second=1000.0, roi_confine=True,
                          rng=random.Random(0))


def test_executes_click_at_roi_offset_screen_coords() -> None:
    bus = EventBus()
    backend = RecordingInputBackend()
    mode = ModeFSM()
    mode.arm()
    started: list[ActionStarted] = []
    completed: list[ActionCompleted] = []
    bus.subscribe(ActionStarted, started.append)
    bus.subscribe(ActionCompleted, completed.append)
    svc = _svc(bus, mode, backend)
    svc.start()
    bus.publish(ActionRequested(
        behavior_name="attack",
        steps=(ActionStep(kind="move", x=10, y=10),
               ActionStep(kind="click", x=10, y=10, button="left")),
        roi=_ROI, priority=5.0))
    time.sleep(0.3)
    svc.stop()
    assert started and completed
    assert ("click", ("left",)) in backend.calls
    assert backend.calls[-2:][0][0] == "move_to"
    last_move = [c for c in backend.calls if c[0] == "move_to"][-1]
    assert last_move == ("move_to", (110, 60))


def test_roi_confine_clamps_out_of_bounds_target() -> None:
    bus = EventBus()
    backend = RecordingInputBackend()
    mode = ModeFSM()
    mode.arm()
    svc = _svc(bus, mode, backend)
    svc.start()
    bus.publish(ActionRequested(
        behavior_name="x",
        steps=(ActionStep(kind="move", x=999, y=-5),),
        roi=_ROI, priority=1.0))
    time.sleep(0.2)
    svc.stop()
    last_move = [c for c in backend.calls if c[0] == "move_to"][-1]
    assert last_move == ("move_to", (139, 50))


def test_disarm_mid_action_aborts_and_stops_injecting() -> None:
    bus = EventBus()
    backend = RecordingInputBackend()
    mode = ModeFSM()
    mode.arm()
    aborted: list[ActionAborted] = []
    bus.subscribe(ActionAborted, aborted.append)
    steps = tuple(ActionStep(kind="wait", duration_s=0.05) for _ in range(20))
    svc = _svc(bus, mode, backend)
    svc.start()
    bus.publish(ActionRequested(behavior_name="long", steps=steps,
                                roi=_ROI, priority=1.0))
    time.sleep(0.1)
    mode.disarm()
    time.sleep(0.2)
    svc.stop()
    assert aborted and aborted[0].reason == "disarmed"
