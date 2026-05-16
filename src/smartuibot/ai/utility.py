# src/smartuibot/ai/utility.py
from __future__ import annotations

import random

from smartuibot.ai.behavior import Behavior
from smartuibot.ai.world_state import WorldState
from smartuibot.core.types import Detection

_ANTI_LOOP_PENALTY = 0.1


class UtilityPolicy:
    """Scores condition-satisfying behaviors and picks the argmax, with
    cooldown exclusion, anti-loop penalty, and seeded human randomness."""

    def __init__(
        self,
        behaviors: tuple[Behavior, ...],
        *,
        tick_hz: float,
        anti_loop_window: int,
        anti_loop_max_repeats: int,
        hesitation_prob: float,
        rng: random.Random,
    ) -> None:
        self._behaviors = behaviors
        self._tick_hz = tick_hz
        self._anti_loop_window = anti_loop_window
        self._anti_loop_max_repeats = anti_loop_max_repeats
        self._hesitation_prob = hesitation_prob
        self._rng = rng

    def choose(
        self, ws: WorldState
    ) -> tuple[Behavior, Detection | None, float] | None:
        if self._rng.random() < self._hesitation_prob:
            return None
        best: tuple[Behavior, Detection | None, float] | None = None
        for b in self._behaviors:
            chosen = b.condition.match(ws)
            if chosen is None:
                continue
            since = ws.ticks_since(b.name)
            cooldown_ticks = round(b.cooldown_s * self._tick_hz)
            if since is not None and since < cooldown_ticks:
                continue
            score = b.base_utility
            if b.scale_by_confidence and chosen is not None:
                score *= chosen.confidence
            if ws.recent_count(b.name, self._anti_loop_window) > self._anti_loop_max_repeats:
                score *= _ANTI_LOOP_PENALTY
            score *= 1.0 + self._rng.uniform(-0.05, 0.05)
            if best is None or score > best[2]:
                best = (b, chosen, score)
        return best
