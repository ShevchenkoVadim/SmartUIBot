# tests/unit/test_events.py
from smartuibot.core.events import (
    DetectionsReady,
    Event,
    FpsTick,
    FrameCaptured,
    LogRecord,
    ServiceError,
    StateChanged,
)


def test_events_are_subclasses_of_event() -> None:
    for cls in (FrameCaptured, DetectionsReady, FpsTick, ServiceError, LogRecord, StateChanged):
        assert issubclass(cls, Event)


def test_fps_tick_fields() -> None:
    e = FpsTick(name="capture", fps=42.5)
    assert e.name == "capture" and e.fps == 42.5


def test_service_error_defaults_nonfatal() -> None:
    e = ServiceError(service="capture", error="boom")
    assert e.fatal is False
