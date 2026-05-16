# src/smartuibot/core/watchdog.py
from __future__ import annotations

import logging
import threading
import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ServiceError
from smartuibot.core.service import Service

_log = logging.getLogger("smartuibot.watchdog")


class Watchdog:
    """Restarts a Service whose thread has died, with exponential backoff."""

    def __init__(
        self,
        services: list[Service],
        bus: EventBus,
        check_interval: float = 1.0,
        base_backoff: float = 0.5,
        max_retries: int = 5,
    ) -> None:
        self._services = services
        self._bus = bus
        self._interval = check_interval
        self._base = base_backoff
        self._max_retries = max_retries
        self._retries: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            for svc in self._services:
                if svc.is_alive:
                    self._retries[svc.name] = 0
                    continue
                n = self._retries.get(svc.name, 0)
                if n >= self._max_retries:
                    self._bus.publish(ServiceError(
                        service=svc.name, error="max restarts exceeded", fatal=True))
                    continue
                self._retries[svc.name] = n + 1
                backoff = self._base * (2 ** n)
                _log.warning("restarting %s (attempt %d) after %.2fs", svc.name, n + 1, backoff)
                time.sleep(backoff)
                svc.start()
