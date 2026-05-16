# tests/unit/test_action_step.py
import pytest

from smartuibot.core.types import ActionStep


def test_action_step_defaults() -> None:
    s = ActionStep(kind="wait", duration_s=0.5)
    assert s.kind == "wait" and s.duration_s == 0.5
    assert s.x == 0 and s.y == 0 and s.button == "left" and s.key == ""


def test_action_step_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ActionStep(kind="teleport")
