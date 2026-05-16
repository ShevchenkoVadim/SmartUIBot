# tests/unit/test_mode.py
from smartuibot.ai.mode import Mode, ModeFSM


def test_starts_disarmed_and_arms() -> None:
    m = ModeFSM()
    assert m.mode == Mode.DISARMED
    assert m.is_armed() is False
    assert m.arm() is True
    assert m.mode == Mode.ARMED and m.is_armed() is True
    assert m.arm() is False  # already armed → no transition


def test_pause_resume_only_from_valid_states() -> None:
    m = ModeFSM()
    assert m.pause() is False  # cannot pause while disarmed
    m.arm()
    assert m.pause() is True and m.mode == Mode.PAUSED and not m.is_armed()
    assert m.resume() is True and m.mode == Mode.ARMED


def test_disarm_from_any_state() -> None:
    m = ModeFSM()
    m.arm()
    assert m.disarm() is True and m.mode == Mode.DISARMED
    assert m.disarm() is False  # already disarmed
