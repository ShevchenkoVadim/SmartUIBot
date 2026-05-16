# src/smartuibot/input/service.py
from __future__ import annotations

import random
import time

from smartuibot.ai.mode import ModeFSM
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import (
    ActionAborted,
    ActionCompleted,
    ActionRequested,
    ActionStarted,
)
from smartuibot.core.latest_queue import LatestQueue
from smartuibot.core.service import Service
from smartuibot.core.types import ROI, ActionStep
from smartuibot.input.backend import InputBackend
from smartuibot.input.motion import MotionParams, bezier_path, reaction_delay


def _confine(x: int, y: int, roi: ROI) -> tuple[int, int]:
    cx = min(max(x, 0), roi.width - 1)
    cy = min(max(y, 0), roi.height - 1)
    return cx, cy


class ActionService(Service):
    """Executes ActionRequested via the InputBackend with humanized motion.
    One action at a time; aborts immediately when disarmed."""

    def __init__(
        self,
        bus: EventBus,
        backend: InputBackend,
        mode: ModeFSM,
        *,
        motion: MotionParams,
        max_actions_per_second: float,
        roi_confine: bool,
        rng: random.Random,
    ) -> None:
        super().__init__(name="action", bus=bus)
        self._backend = backend
        self._mode = mode
        self._motion = motion
        self._min_interval = 1.0 / max_actions_per_second if max_actions_per_second > 0 else 0.0
        self._roi_confine = roi_confine
        self._rng = rng
        self._queue: LatestQueue[ActionRequested] = LatestQueue()
        self._cursor = (0, 0)
        bus.subscribe(ActionRequested, self._on_request)

    def _on_request(self, event: ActionRequested) -> None:
        self._queue.put(event)

    def _screen(self, x: int, y: int, roi: ROI) -> tuple[int, int]:
        if self._roi_confine:
            x, y = _confine(x, y, roi)
        return roi.x + x, roi.y + y

    def _aborted(self) -> bool:
        return self._stop.is_set() or not self._mode.is_armed()

    def run_once(self) -> None:
        req = self._queue.get(timeout=0.1)
        if req is None:
            return
        if not self._mode.is_armed():
            return
        self._bus.publish(ActionStarted(behavior_name=req.behavior_name))
        for step in req.steps:
            if self._aborted():
                self._bus.publish(ActionAborted(
                    behavior_name=req.behavior_name, reason="disarmed"))
                return
            self._exec_step(step, req.roi)
            if self._min_interval:
                time.sleep(self._min_interval)
        self._bus.publish(ActionCompleted(behavior_name=req.behavior_name))

    def _exec_step(self, step: ActionStep, roi: ROI) -> None:
        if step.kind == "move":
            target = self._screen(step.x, step.y, roi)
            for px, py in bezier_path(self._cursor, target,
                                      params=self._motion, rng=self._rng):
                if self._aborted():
                    return
                self._backend.move_to(px, py)
            self._cursor = target
        elif step.kind == "click":
            target = self._screen(step.x, step.y, roi)
            self._backend.move_to(*target)
            self._cursor = target
            time.sleep(reaction_delay(self._motion, self._rng))
            self._backend.click(step.button)
        elif step.kind == "key":
            self._backend.key_down(step.key)
            self._backend.key_up(step.key)
        else:  # wait
            time.sleep(step.duration_s)
