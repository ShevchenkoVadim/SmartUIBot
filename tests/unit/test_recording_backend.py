# tests/unit/test_recording_backend.py
from smartuibot.input.backend import InputBackend, NoOpInputBackend
from tests.fakes.input import RecordingInputBackend


def test_recording_backend_satisfies_protocol_and_records() -> None:
    be: InputBackend = RecordingInputBackend()
    be.move_to(10, 20)
    be.click("left")
    be.key_down("a")
    be.key_up("a")
    be.type_text("hi")
    assert be.calls == [
        ("move_to", (10, 20)),
        ("click", ("left",)),
        ("key_down", ("a",)),
        ("key_up", ("a",)),
        ("type_text", ("hi",)),
    ]


def test_noop_backend_satisfies_protocol() -> None:
    be: InputBackend = NoOpInputBackend()
    assert isinstance(be, InputBackend)
    be.move_to(1, 2)
    be.click("left")  # must not raise
