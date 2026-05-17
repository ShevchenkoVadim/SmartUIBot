# src/smartuibot/ai/service.py
from __future__ import annotations

import threading
import time

from smartuibot.ai.behavior import resolve_steps
from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ActionRequested, DetectionsEnriched
from smartuibot.core.service import Service
from smartuibot.core.types import ROI, Detection


class DecisionService(Service):
    def __init__(
        self,
        bus: EventBus,
        policy: UtilityPolicy,
        tracker: WorldStateTracker,
        mode: ModeFSM,
        tick_hz: float,
    ) -> None:
        super().__init__(name="decision", bus=bus)
        self._policy = policy
        self._tracker = tracker
        self._mode = mode
        self._period = 1.0 / tick_hz
        self._lock = threading.Lock()
        self._latest: tuple[tuple[Detection, ...], ROI] | None = None
        bus.subscribe(DetectionsEnriched, self._on_detections)

    def _on_detections(self, event: DetectionsEnriched) -> None:
        with self._lock:
            self._latest = (event.detections, event.frame.roi)

    def run_once(self) -> None:
        start = time.monotonic()
        if not self._mode.is_armed():
            time.sleep(self._period)
            return
        with self._lock:
            latest = self._latest
        if latest is None:
            time.sleep(self._period)
            return
        detections, roi = latest
        ws = self._tracker.snapshot(detections, roi)
        result = self._policy.choose(ws)
        if result is not None:
            behavior, chosen, score = result
            steps = resolve_steps(behavior.steps, chosen, roi)
            self._tracker.record(behavior.name)
            self._bus.publish(ActionRequested(
                behavior_name=behavior.name, steps=steps, roi=roi, priority=score))
        elapsed = time.monotonic() - start
        if elapsed < self._period:
            time.sleep(self._period - elapsed)
