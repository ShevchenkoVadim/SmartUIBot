# src/smartuibot/vision/capture/backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from smartuibot.core.types import ROI, Image


@dataclass(frozen=True, slots=True)
class Monitor:
    index: int
    x: int
    y: int
    width: int
    height: int


@runtime_checkable
class CaptureBackend(Protocol):
    def list_monitors(self) -> list[Monitor]: ...

    def grab(self, roi: ROI) -> Image:
        """Return an HxWx3 BGR uint8 array for the ROI."""
