# src/smartuibot/ai/behavior.py
from __future__ import annotations

from dataclasses import dataclass, field

from smartuibot.ai.world_state import WorldState
from smartuibot.core.types import ROI, ActionStep, Detection


@dataclass(frozen=True, slots=True)
class Condition:
    labels: frozenset[str]
    min_confidence: float = 0.0
    min_count: int = 1

    def match(self, ws: WorldState) -> Detection | None:
        return ws.best_match(self.labels, self.min_confidence, self.min_count)


@dataclass(frozen=True, slots=True)
class BehaviorStep:
    kind: str  # move | click | key | wait
    target: str = "detection"  # detection | roi_center | fixed
    x: int = 0
    y: int = 0
    button: str = "left"
    key: str = ""
    duration_s: float = 0.0


@dataclass(frozen=True, slots=True)
class Behavior:
    name: str
    condition: Condition
    base_utility: float
    cooldown_s: float = 0.0
    steps: tuple[BehaviorStep, ...] = field(default_factory=tuple)
    scale_by_confidence: bool = True


def resolve_steps(
    steps: tuple[BehaviorStep, ...], chosen: Detection | None, roi: ROI
) -> tuple[ActionStep, ...]:
    out: list[ActionStep] = []
    for s in steps:
        if s.kind in ("move", "click"):
            if s.target == "detection" and chosen is not None:
                x = (chosen.x1 + chosen.x2) // 2
                y = (chosen.y1 + chosen.y2) // 2
            elif s.target == "roi_center":
                x = roi.width // 2
                y = roi.height // 2
            else:  # fixed (or detection with no chosen → use fixed coords)
                x = s.x
                y = s.y
            out.append(ActionStep(kind=s.kind, x=x, y=y, button=s.button))
        elif s.kind == "key":
            out.append(ActionStep(kind="key", key=s.key))
        else:  # wait
            out.append(ActionStep(kind="wait", duration_s=s.duration_s))
    return tuple(out)
