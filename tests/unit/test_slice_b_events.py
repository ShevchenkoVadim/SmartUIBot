# tests/unit/test_slice_b_events.py
from smartuibot.core.events import (
    ActionAborted,
    ActionCompleted,
    ActionRequested,
    ActionStarted,
    Event,
    ModeChanged,
)
from smartuibot.core.types import ROI, ActionStep


def test_action_requested_carries_steps_roi_priority() -> None:
    roi = ROI(monitor=1, x=0, y=0, width=10, height=10)
    e = ActionRequested(behavior_name="attack",
                        steps=(ActionStep(kind="click", x=1, y=2),),
                        roi=roi, priority=3.0)
    assert isinstance(e, Event)
    assert e.behavior_name == "attack" and e.priority == 3.0
    assert e.steps[0].kind == "click" and e.roi == roi


def test_other_events_subclass_event() -> None:
    for ev in (ActionStarted("a"), ActionCompleted("a"),
               ActionAborted("a", "disarmed"), ModeChanged("armed")):
        assert isinstance(ev, Event)
