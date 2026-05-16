# tests/unit/test_config_slice_b.py
import textwrap
from pathlib import Path

from smartuibot.core.config import load_config

_CFG = """
capture: {backend: auto, target_fps: 60, monitor: 1}
detection: {model: yolo11n.pt, confidence: 0.35, device: auto, tracking: false, smoothing_frames: 3}
ui: {preview_max_width: 960}
logging: {level: INFO, dir: logs}
hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
decision:
  tick_hz: 10.0
  anti_loop_window: 30
  anti_loop_max_repeats: 3
  hesitation_prob: 0.02
  rng_seed: 7
input:
  backend: auto
  max_actions_per_second: 8.0
  roi_confine: true
  start_armed: false
  move_steps: 24
  jitter_px: 2
  reaction_min_s: 0.08
  reaction_max_s: 0.22
  keystroke_min_s: 0.03
  keystroke_max_s: 0.09
  overshoot_prob: 0.15
behaviors_path: configs/behaviors.yaml
"""


def test_loads_decision_and_input_config(tmp_path: Path) -> None:
    p = tmp_path / "d.yaml"
    p.write_text(textwrap.dedent(_CFG))
    cfg = load_config(p)
    assert cfg.decision.tick_hz == 10.0
    assert cfg.decision.rng_seed == 7
    assert cfg.input.backend == "auto"
    assert cfg.input.roi_confine is True
    assert cfg.input.start_armed is False
    assert cfg.behaviors_path == "configs/behaviors.yaml"
