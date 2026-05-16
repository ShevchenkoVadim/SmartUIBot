# src/smartuibot/core/latest_queue.py
from __future__ import annotations

import threading


class LatestQueue[T]:
    """Holds at most one item. put() overwrites any pending item so consumers
    always see the freshest value (drop-old backpressure)."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._item: T | None = None
        self._has_item = False

    def put(self, item: T) -> None:
        with self._cv:
            self._item = item
            self._has_item = True
            self._cv.notify()

    def get(self, timeout: float) -> T | None:
        with self._cv:
            if not self._has_item:
                self._cv.wait(timeout)
            if not self._has_item:
                return None
            item = self._item
            self._item = None
            self._has_item = False
            return item

    def clear(self) -> None:
        with self._cv:
            self._item = None
            self._has_item = False
