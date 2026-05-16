import time
from pathlib import Path

from smartuibot.core.config import load_config
from smartuibot.core.container import AppContainer
from smartuibot.core.events import DetectionsReady
from smartuibot.core.types import ROI
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector


def test_container_wires_pipeline_with_injected_fakes(tmp_path: Path) -> None:
    default = tmp_path / "d.yaml"
    default.write_text(
        "capture: {backend: auto, target_fps: 120, monitor: 1}\n"
        "detection: {model: yolo11n.pt, confidence: 0.3, device: cpu, "
        "tracking: false, smoothing_frames: 1}\n"
        "ui: {preview_max_width: 960}\n"
        "logging: {level: INFO, dir: " + str(tmp_path / 'logs') + "}\n"
        "hotkeys: {emergency_stop: \"<ctrl>+<alt>+q\"}\n"
    )
    cfg = load_config(default)
    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=16, height=16),
        capture_backend=FakeCaptureBackend(),
        detector=FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 4, 4)]] * 40),
    )
    out: list[DetectionsReady] = []
    container.bus.subscribe(DetectionsReady, out.append)
    container.start()
    time.sleep(0.3)
    container.stop()
    assert out, "expected detections to flow capture->detect->bus"
    assert out[-1].detections[0].label == "enemy"
