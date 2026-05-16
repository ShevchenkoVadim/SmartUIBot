# src/smartuibot/core/event_bus.py
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

from smartuibot.core.events import Event

E = TypeVar("E", bound=Event)
_log = logging.getLogger("smartuibot.event_bus")


class EventBus:
    """Thread-safe synchronous pub/sub. A subscriber exception is logged and
    swallowed so it can never break the publisher or other subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[type[Event], list[Callable[..., None]]] = {}

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        with self._lock:
            self._subs.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._subs.get(type(event), ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - isolation is the whole point
                _log.exception("subscriber for %s failed", type(event).__name__)
