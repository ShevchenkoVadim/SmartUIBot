# src/smartuibot/core/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    backend: str
    target_fps: int
    monitor: int


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    model: str
    confidence: float
    device: str
    tracking: bool
    smoothing_frames: int


@dataclass(frozen=True, slots=True)
class UIConfig:
    preview_max_width: int


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str
    dir: str


@dataclass(frozen=True, slots=True)
class HotkeyConfig:
    emergency_stop: str


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    tick_hz: float
    anti_loop_window: int
    anti_loop_max_repeats: int
    hesitation_prob: float
    rng_seed: int


@dataclass(frozen=True, slots=True)
class InputConfig:
    backend: str
    max_actions_per_second: float
    roi_confine: bool
    start_armed: bool
    move_steps: int
    jitter_px: int
    reaction_min_s: float
    reaction_max_s: float
    keystroke_min_s: float
    keystroke_max_s: float
    overshoot_prob: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    capture: CaptureConfig
    detection: DetectionConfig
    ui: UIConfig
    logging: LoggingConfig
    hotkeys: HotkeyConfig
    decision: DecisionConfig = field(
        default_factory=lambda: DecisionConfig(10.0, 30, 3, 0.02, 7))
    input: InputConfig = field(
        default_factory=lambda: InputConfig("auto", 8.0, True, False, 24, 2,
                                            0.08, 0.22, 0.03, 0.09, 0.15))
    behaviors_path: str = "configs/behaviors.yaml"

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection.confidence <= 1.0:
            raise ValueError("detection.confidence must be in [0, 1]")
        if self.capture.target_fps <= 0:
            raise ValueError("capture.target_fps must be positive")
        if self.detection.smoothing_frames < 1:
            raise ValueError("detection.smoothing_frames must be >= 1")
        if self.decision.tick_hz <= 0:
            raise ValueError("decision.tick_hz must be positive")
        if not 0.0 <= self.input.overshoot_prob <= 1.0:
            raise ValueError("input.overshoot_prob must be in [0, 1]")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(default_path: Path, user_path: Path | None = None) -> AppConfig:
    data: dict[str, Any] = yaml.safe_load(Path(default_path).read_text()) or {}
    if user_path is not None and Path(user_path).exists():
        user = yaml.safe_load(Path(user_path).read_text()) or {}
        data = _deep_merge(data, user)
    return AppConfig(
        capture=CaptureConfig(**data["capture"]),
        detection=DetectionConfig(**data["detection"]),
        ui=UIConfig(**data["ui"]),
        logging=LoggingConfig(**data["logging"]),
        hotkeys=HotkeyConfig(**data["hotkeys"]),
        decision=DecisionConfig(**data["decision"]) if "decision" in data
            else DecisionConfig(10.0, 30, 3, 0.02, 7),
        input=InputConfig(**data["input"]) if "input" in data
            else InputConfig("auto", 8.0, True, False, 24, 2, 0.08, 0.22,
                             0.03, 0.09, 0.15),
        behaviors_path=str(data.get("behaviors_path", "configs/behaviors.yaml")),
    )
