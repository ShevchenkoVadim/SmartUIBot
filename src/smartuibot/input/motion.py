# src/smartuibot/input/motion.py
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotionParams:
    move_steps: int
    jitter_px: int
    reaction_min_s: float
    reaction_max_s: float
    keystroke_min_s: float
    keystroke_max_s: float
    overshoot_prob: float


def bezier_path(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    params: MotionParams,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Quadratic bezier with a random control point and per-point jitter.
    First point is exactly `start`, last is exactly `end`."""
    n = max(2, params.move_steps)
    x0, y0 = start
    x1, y1 = end
    cx = (x0 + x1) / 2 + rng.uniform(-abs(x1 - x0 or 1), abs(x1 - x0 or 1)) * 0.3
    cy = (y0 + y1) / 2 + rng.uniform(-abs(y1 - y0 or 1), abs(y1 - y0 or 1)) * 0.3
    pts: list[tuple[int, int]] = []
    for i in range(n):
        t = i / (n - 1)
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
        by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
        if 0 < i < n - 1 and params.jitter_px > 0:
            bx += rng.uniform(-params.jitter_px, params.jitter_px)
            by += rng.uniform(-params.jitter_px, params.jitter_px)
        pts.append((round(bx), round(by)))
    pts[0] = start
    pts[-1] = end
    return pts


def reaction_delay(params: MotionParams, rng: random.Random) -> float:
    return rng.uniform(params.reaction_min_s, params.reaction_max_s)


def keystroke_delays(n: int, params: MotionParams, rng: random.Random) -> list[float]:
    return [rng.uniform(params.keystroke_min_s, params.keystroke_max_s)
            for _ in range(n)]


def maybe_overshoot(
    end: tuple[int, int], params: MotionParams, rng: random.Random
) -> tuple[int, int] | None:
    if rng.random() >= params.overshoot_prob:
        return None
    ox = end[0] + rng.randint(-6, 6)
    oy = end[1] + rng.randint(-6, 6)
    return (ox, oy)
