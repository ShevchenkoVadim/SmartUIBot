# src/smartuibot/vision/capture/mss_backend.py
from __future__ import annotations

import numpy as np

from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import Monitor


class MssBackend:
    """Screen capture via `mss`. Works on macOS (Intel/ARM), Windows, Linux."""

    def __init__(self) -> None:
        import mss

        self._sct = mss.mss()

    def list_monitors(self) -> list[Monitor]:
        # mss.monitors[0] is the virtual "all monitors" rect; real ones start at 1.
        out: list[Monitor] = []
        for i, m in enumerate(self._sct.monitors[1:], start=1):
            out.append(Monitor(index=i, x=m["left"], y=m["top"],
                                width=m["width"], height=m["height"]))
        return out

    @staticmethod
    def _to_bgr(bgra: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(bgra[:, :, :3])

    def grab(self, roi: ROI) -> np.ndarray:
        mon = self._sct.monitors[roi.monitor]
        region = {
            "left": mon["left"] + roi.x,
            "top": mon["top"] + roi.y,
            "width": roi.width,
            "height": roi.height,
        }
        shot = self._sct.grab(region)
        bgra = np.asarray(shot, dtype=np.uint8)  # mss returns BGRA
        return self._to_bgr(bgra)
