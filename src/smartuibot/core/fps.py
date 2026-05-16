# src/smartuibot/core/fps.py
from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Rolling FPS over the last `window` tick timestamps."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> None:
        self._times.append(time.monotonic() if now is None else now)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span
