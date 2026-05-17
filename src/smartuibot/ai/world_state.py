# src/smartuibot/ai/world_state.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from smartuibot.core.types import ROI, Detection


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


@dataclass(frozen=True, slots=True)
class WorldState:
    detections: tuple[Detection, ...]
    roi: ROI
    tick: int
    recent: tuple[tuple[str, int], ...]  # (behavior_name, tick), oldest first

    def best_match(
        self,
        labels: frozenset[str],
        min_confidence: float,
        min_count: int,
        text_any: frozenset[str] = frozenset(),
    ) -> Detection | None:
        matches = sorted(
            (d for d in self.detections
             if d.label in labels and d.confidence >= min_confidence
             and (not text_any or (
                 d.text is not None
                 and any(n in _normalize(d.text) for n in text_any)))),
            key=lambda d: d.confidence,
            reverse=True,
        )
        if not matches or len(matches) < min_count:
            return None
        return matches[0]

    def ticks_since(self, name: str) -> int | None:
        for n, t in reversed(self.recent):
            if n == name:
                return self.tick - t
        return None

    def recent_count(self, name: str, window: int) -> int:
        return sum(
            1 for n, t in self.recent
            if n == name and 0 <= self.tick - t < window
        )


class WorldStateTracker:
    def __init__(self, recent_size: int = 64) -> None:
        self._tick = 0
        self._recent: deque[tuple[str, int]] = deque(maxlen=recent_size)

    def snapshot(self, detections: tuple[Detection, ...], roi: ROI) -> WorldState:
        self._tick += 1
        return WorldState(
            detections=detections, roi=roi, tick=self._tick,
            recent=tuple(self._recent),
        )

    def record(self, name: str) -> None:
        self._recent.append((name, self._tick))

