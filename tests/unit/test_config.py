import textwrap
from pathlib import Path

import pytest

from smartuibot.core.config import AppConfig, load_config


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_loads_defaults(tmp_path: Path) -> None:
    default = _write(tmp_path, "d.yaml", """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {
            model: yolo11n.pt, confidence: 0.35, device: auto,
            tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """)
    cfg = load_config(default)
    assert isinstance(cfg, AppConfig)
    assert cfg.detection.confidence == 0.35
    assert cfg.capture.target_fps == 60


def test_user_overrides_merge_over_defaults(tmp_path: Path) -> None:
    default = _write(tmp_path, "d.yaml", """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {
            model: yolo11n.pt, confidence: 0.35, device: auto,
            tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """)
    user = _write(tmp_path, "u.yaml", "detection: {confidence: 0.7}\n")
    cfg = load_config(default, user)
    assert cfg.detection.confidence == 0.7
    assert cfg.detection.model == "yolo11n.pt"  # untouched


def test_invalid_confidence_rejected(tmp_path: Path) -> None:
    default = _write(tmp_path, "d.yaml", """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {
            model: yolo11n.pt, confidence: 9.0, device: auto,
            tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """)
    with pytest.raises(ValueError):
        load_config(default)
