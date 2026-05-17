import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from smartuibot.core.event_bus import EventBus  # noqa: E402
from smartuibot.core.events import DetectionsEnriched, FpsTick, LogRecord  # noqa: E402
from smartuibot.core.types import ROI, Detection, Frame  # noqa: E402
from smartuibot.ui.debug_window import DebugWindow, draw_boxes  # noqa: E402


def test_draw_boxes_does_not_mutate_input_and_keeps_shape() -> None:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    dets = [Detection(label="e", confidence=0.9, class_id=0, x1=2, y1=2, x2=20, y2=20)]
    out = draw_boxes(img, dets)
    assert out.shape == img.shape
    assert img.sum() == 0  # original untouched (copy made)
    assert out.sum() > 0   # something drawn


def test_debug_window_consumes_bus_events_without_error() -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    bus = EventBus()
    win = DebugWindow(bus=bus, preview_max_width=320)
    frame = Frame(image=np.zeros((30, 40, 3), dtype=np.uint8),
                  timestamp=0.0, seq=1, roi=ROI(monitor=1, x=0, y=0, width=40, height=30))
    bus.publish(DetectionsEnriched(frame=frame, detections=(
        Detection(label="button", confidence=0.8, class_id=0, x1=1, y1=1,
                  x2=5, y2=5, text="Close", text_confidence=0.9),)))
    bus.publish(FpsTick(name="capture", fps=55.0))
    bus.publish(FpsTick(name="detection", fps=7.0))
    bus.publish(LogRecord(level="INFO", logger="t", message="hi", ts=0.0))
    win._drain()  # process queued events synchronously
    assert win.detection_count() == 1
    assert win.last_detection_texts() == ["Close"]
    assert "55.0" in win.fps_text()
    win.close()
