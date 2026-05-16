# src/smartuibot/ai/registry.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from smartuibot.ai.behavior import Behavior, BehaviorStep, Condition

_ALLOWED_KINDS = frozenset({"move", "click", "key", "wait"})


def _build_step(raw: dict[str, Any]) -> BehaviorStep:
    kind = str(raw["kind"])
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"behavior step kind {kind!r} not in {sorted(_ALLOWED_KINDS)}")
    return BehaviorStep(
        kind=kind,
        target=str(raw.get("target", "detection")),
        x=int(raw.get("x", 0)),
        y=int(raw.get("y", 0)),
        button=str(raw.get("button", "left")),
        key=str(raw.get("key", "")),
        duration_s=float(raw.get("duration_s", 0.0)),
    )


def load_behaviors(path: Path) -> tuple[Behavior, ...]:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    out: list[Behavior] = []
    for raw in data.get("behaviors", []):
        cond_raw = raw["condition"]
        cond = Condition(
            labels=frozenset(str(x) for x in cond_raw["labels"]),
            min_confidence=float(cond_raw.get("min_confidence", 0.0)),
            min_count=int(cond_raw.get("min_count", 1)),
        )
        steps = tuple(_build_step(s) for s in raw.get("steps", []))
        base_utility = float(raw["base_utility"])
        cooldown_s = float(raw.get("cooldown_s", 0.0))
        if base_utility <= 0:
            raise ValueError(f"behavior {raw.get('name')!r}: base_utility must be > 0")
        if cooldown_s < 0:
            raise ValueError(f"behavior {raw.get('name')!r}: cooldown_s must be >= 0")
        out.append(Behavior(
            name=str(raw["name"]), condition=cond, base_utility=base_utility,
            cooldown_s=cooldown_s, steps=steps,
            scale_by_confidence=bool(raw.get("scale_by_confidence", True)),
        ))
    return tuple(out)

