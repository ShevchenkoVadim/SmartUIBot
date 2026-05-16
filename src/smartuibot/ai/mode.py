# src/smartuibot/ai/mode.py
from __future__ import annotations

import threading


class Mode:
    DISARMED = "disarmed"
    ARMED = "armed"
    PAUSED = "paused"


class ModeFSM:
    """Thread-safe coarse mode gate. Injection is allowed only in ARMED."""

    def __init__(self) -> None:
        self._mode = Mode.DISARMED
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def is_armed(self) -> bool:
        with self._lock:
            return self._mode == Mode.ARMED

    def arm(self) -> bool:
        with self._lock:
            if self._mode in (Mode.DISARMED, Mode.PAUSED):
                self._mode = Mode.ARMED
                return True
            return False

    def disarm(self) -> bool:
        with self._lock:
            if self._mode != Mode.DISARMED:
                self._mode = Mode.DISARMED
                return True
            return False

    def pause(self) -> bool:
        with self._lock:
            if self._mode == Mode.ARMED:
                self._mode = Mode.PAUSED
                return True
            return False

    def resume(self) -> bool:
        with self._lock:
            if self._mode == Mode.PAUSED:
                self._mode = Mode.ARMED
                return True
            return False
