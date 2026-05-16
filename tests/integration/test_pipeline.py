import time
from pathlib import Path

import numpy as np

from smartuibot.core.config import AppConfig, load_config
from smartuibot.core.container import AppContainer
from smartuibot.core.events import DetectionsReady, ServiceError
from smartuibot.core.types import ROI, Detection
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector


def _cfg(tmp_path: Path) -> AppConfig:
    p = tmp_path / "d.yaml"
    p.write_text(
        "capture: {backend: auto, target_fps: 90, monitor: 1}\n"
        "detection: {model: yolo11n.pt, confidence: 0.3, device: cpu, "
        "tracking: false, smoothing_frames: 2}\n"
        "ui: {preview_max_width: 960}\n"
        "logging: {level: INFO, dir: " + str(tmp_path / "logs") + "}\n"
        "hotkeys: {emergency_stop: \"<ctrl>+<alt>+q\"}\n"
    )
    return load_config(p)


def test_frames_flow_and_stale_frames_are_dropped(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    class _SlowDetector(FakeDetector):
        def infer(self, image: np.ndarray) -> list[Detection]:
            time.sleep(0.05)  # detector slower than capture
            return super().infer(image)

    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=24, height=24),
        capture_backend=FakeCaptureBackend(),
        detector=_SlowDetector(scripted=[[("e", 0.9, 0, 0, 2, 2)]] * 200),
    )
    errors: list[ServiceError] = []
    results: list[DetectionsReady] = []
    container.bus.subscribe(ServiceError, errors.append)
    container.bus.subscribe(DetectionsReady, results.append)

    container.start()
    time.sleep(1.0)
    container.stop()

    assert not errors, f"unexpected service errors: {errors}"
    assert results, "no detections produced"
    # Detector (~20 fps) far slower than capture (~90 fps): far fewer results
    # than frames, proving stale frames were dropped, not buffered.
    assert len(results) < 60
    seqs = [r.frame.seq for r in results]
    assert seqs == sorted(seqs), "results must be in monotonically increasing seq order"
