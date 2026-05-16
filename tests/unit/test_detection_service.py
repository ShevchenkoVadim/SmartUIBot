import time

import numpy as np

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsReady, FrameCaptured
from smartuibot.core.types import ROI, Frame
from smartuibot.vision.detect.service import DetectionService
from tests.fakes.detector import FakeDetector


def _frame(seq: int) -> Frame:
    return Frame(image=np.zeros((8, 8, 3), dtype=np.uint8),
                 timestamp=time.monotonic(), seq=seq,
                 roi=ROI(monitor=1, x=0, y=0, width=8, height=8))


def test_detection_service_consumes_frames_and_publishes() -> None:
    bus = EventBus()
    out: list[DetectionsReady] = []
    bus.subscribe(DetectionsReady, out.append)
    det = FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 4, 4)]] * 3)
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1)
    svc.start()
    bus.publish(FrameCaptured(frame=_frame(1)))
    time.sleep(0.2)
    svc.stop()
    assert out
    assert out[0].detections[0].label == "enemy"


def test_only_latest_frame_is_processed_under_backlog() -> None:
    bus = EventBus()
    out: list[DetectionsReady] = []
    bus.subscribe(DetectionsReady, out.append)
    det = FakeDetector(scripted=[[("x", 0.9, 0, 0, 1, 1)]] * 50)
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1)
    for s in range(10):
        bus.publish(FrameCaptured(frame=_frame(s)))  # before start: only last kept
    svc.start()
    time.sleep(0.15)
    svc.stop()
    # Latest-wins queue means we did NOT emit 10 results for the backlog.
    assert 1 <= len(out) <= 3
    assert out[-1].frame.seq == 9


def test_runtime_confidence_filters_low_confidence_detections() -> None:
    bus = EventBus()
    out: list[DetectionsReady] = []
    bus.subscribe(DetectionsReady, out.append)
    det = FakeDetector(scripted=[[("hi", 0.9, 0, 0, 1, 1), ("lo", 0.2, 0, 0, 1, 1)]] * 5)
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1, confidence=0.5)
    svc.start()
    bus.publish(FrameCaptured(frame=_frame(1)))
    time.sleep(0.15)
    svc.stop()
    labels = {d.label for d in out[0].detections}
    assert labels == {"hi"}  # 0.2 < 0.5 threshold filtered out


def test_set_confidence_and_reload_model_are_thread_safe_noops_on_fake() -> None:
    bus = EventBus()
    det = FakeDetector(scripted=[[("x", 0.9, 0, 0, 1, 1)]])
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1, confidence=0.3)
    svc.set_confidence(0.95)        # must not raise
    svc.reload_model("ignored.pt")  # delegates to detector.reload
    assert svc.confidence == 0.95
