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


def test_detections_enriched_event_shape() -> None:
    import numpy as np

    from smartuibot.core.events import DetectionsEnriched
    from smartuibot.core.types import ROI, Detection, Frame

    roi = ROI(monitor=1, x=0, y=0, width=4, height=4)
    frame = Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), timestamp=0.0,
                  seq=1, roi=roi)
    det = Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2,
                    y2=2, text="Close", text_confidence=0.9)
    ev = DetectionsEnriched(frame=frame, detections=(det,))
    assert ev.detections[0].text == "Close"
