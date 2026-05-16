# src/smartuibot/core/service.py
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ServiceError, StateChanged


class Service(ABC):
    """Base worker: owns a thread, updates a heartbeat each loop, and converts
    an unhandled exception into a fatal ServiceError then stops cleanly."""

    def __init__(self, name: str, bus: EventBus) -> None:
        self.name = name
        self._bus = bus
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.last_heartbeat: float = 0.0

    @abstractmethod
    def run_once(self) -> None:
        """One unit of work. Called repeatedly until stop()."""

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()
        self._bus.publish(StateChanged(service=self.name, state="running"))

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._bus.publish(StateChanged(service=self.name, state="stopped"))

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.02)
                continue
            try:
                self.run_once()
                self.last_heartbeat = time.monotonic()
            except Exception as exc:  # noqa: BLE001 - boundary
                self._bus.publish(ServiceError(service=self.name, error=repr(exc), fatal=True))
                return
