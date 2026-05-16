# tests/unit/test_container_slice_b.py
import time
from pathlib import Path

from smartuibot.core.config import load_config
from smartuibot.core.container import AppContainer
from smartuibot.core.events import ActionStarted
from smartuibot.core.types import ROI
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector
from tests.fakes.input import RecordingInputBackend

_CFG = (
    "capture: {backend: auto, target_fps: 120, monitor: 1}\n"
    "detection: {model: yolo11n.pt, confidence: 0.1, device: cpu,"
    " tracking: false, smoothing_frames: 1}\n"
    "ui: {preview_max_width: 960}\n"
    "logging: {level: INFO, dir: %LOGS%}\n"
    'hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}\n'
    "decision: {tick_hz: 50.0, anti_loop_window: 5, anti_loop_max_repeats: 99,"
    " hesitation_prob: 0.0, rng_seed: 1}\n"
    "input: {backend: auto, max_actions_per_second: 1000.0, roi_confine: true,"
    " start_armed: true, move_steps: 3, jitter_px: 0, reaction_min_s: 0.0,"
    " reaction_max_s: 0.0, keystroke_min_s: 0.0, keystroke_max_s: 0.0,"
    " overshoot_prob: 0.0}\n"
    "behaviors_path: %BEH%\n"
)

_BEH = """
behaviors:
  - name: attack
    base_utility: 5.0
    condition: {labels: [enemy], min_confidence: 0.1}
    scale_by_confidence: false
    steps:
      - {kind: click, target: detection, button: left}
"""


def test_container_runs_full_closed_loop_with_fakes(tmp_path: Path) -> None:
    beh = tmp_path / "behaviors.yaml"
    beh.write_text(_BEH)
    cfg_path = tmp_path / "d.yaml"
    cfg_path.write_text(_CFG.replace("%LOGS%", str(tmp_path / "logs"))
                            .replace("%BEH%", str(beh)))
    cfg = load_config(cfg_path)
    backend = RecordingInputBackend()
    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=16, height=16),
        capture_backend=FakeCaptureBackend(),
        detector=FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 8, 8)]] * 400),
        input_backend=backend,
    )
    started: list[ActionStarted] = []
    container.bus.subscribe(ActionStarted, started.append)
    container.mode.arm()
    container.start()
    time.sleep(0.6)
    container.stop()
    assert started, "expected the closed loop to trigger an action"
    assert ("click", ("left",)) in backend.calls
