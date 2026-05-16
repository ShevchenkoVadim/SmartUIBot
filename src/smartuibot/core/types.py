# src/smartuibot/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ROI:
    monitor: int
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if self.monitor < 0:
            raise ValueError("ROI monitor index must be >= 0")

    def as_dict(self) -> dict[str, int]:
        return {
            "monitor": self.monitor,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ROI:
        return cls(
            monitor=int(d["monitor"]),
            x=int(d["x"]),
            y=int(d["y"]),
            width=int(d["width"]),
            height=int(d["height"]),
        )


@dataclass(slots=True)
class Frame:
    image: np.ndarray  # HxWx3 BGR uint8
    timestamp: float
    seq: int
    roi: ROI


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
