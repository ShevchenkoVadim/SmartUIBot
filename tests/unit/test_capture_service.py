import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import FpsTick, FrameCaptured
from smartuibot.core.types import ROI
from smartuibot.vision.capture.service import CaptureService
from tests.fakes.capture import FakeCaptureBackend


def test_capture_service_publishes_frames_and_fps() -> None:
    bus = EventBus()
    frames: list[FrameCaptured] = []
    fps: list[FpsTick] = []
    bus.subscribe(FrameCaptured, frames.append)
    bus.subscribe(FpsTick, lambda e: fps.append(e) if e.name == "capture" else None)
    svc = CaptureService(
        backend=FakeCaptureBackend(),
        bus=bus,
        roi=ROI(monitor=1, x=0, y=0, width=32, height=24),
        target_fps=120,
    )
    svc.start()
    time.sleep(0.2)
    svc.stop()
    assert len(frames) > 2
    assert frames[0].frame.seq < frames[-1].frame.seq
    assert any(f.fps > 0 for f in fps)


def test_set_roi_changes_frame_size_without_restart() -> None:
    bus = EventBus()
    frames: list[FrameCaptured] = []
    bus.subscribe(FrameCaptured, frames.append)
    svc = CaptureService(FakeCaptureBackend(), bus,
                          ROI(monitor=1, x=0, y=0, width=10, height=10), target_fps=120)
    svc.start()
    time.sleep(0.05)
    svc.set_roi(ROI(monitor=1, x=0, y=0, width=20, height=15))
    time.sleep(0.1)
    svc.stop()
    assert frames[-1].frame.image.shape == (15, 20, 3)
