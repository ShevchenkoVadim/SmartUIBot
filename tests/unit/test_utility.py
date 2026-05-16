# tests/unit/test_utility.py
import random

from smartuibot.ai.behavior import Behavior, BehaviorStep, Condition
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldState, WorldStateTracker
from smartuibot.core.types import ROI, Detection

_ROI = ROI(monitor=1, x=0, y=0, width=10, height=10)


def _det(label: str, conf: float) -> Detection:
    return Detection(label=label, confidence=conf, class_id=0, x1=0, y1=0, x2=4, y2=4)


def _b(name: str, label: str, util: float, cooldown: float = 0.0) -> Behavior:
    return Behavior(name=name, condition=Condition(labels=frozenset({label})),
                    base_utility=util, cooldown_s=cooldown,
                    steps=(BehaviorStep(kind="click", target="detection"),),
                    scale_by_confidence=False)


def _policy(behaviors: tuple[Behavior, ...], **kw: object) -> UtilityPolicy:
    return UtilityPolicy(
        behaviors,
        tick_hz=float(kw.get("tick_hz", 10.0)),
        anti_loop_window=int(kw.get("anti_loop_window", 5)),
        anti_loop_max_repeats=int(kw.get("anti_loop_max_repeats", 2)),
        hesitation_prob=float(kw.get("hesitation_prob", 0.0)),
        rng=random.Random(1234),
    )


def test_argmax_behavior_chosen() -> None:
    p = _policy((_b("low", "enemy", 1.0), _b("high", "enemy", 5.0)))
    ws = WorldState(detections=(_det("enemy", 0.9),), roi=_ROI, tick=1, recent=())
    result = p.choose(ws)
    assert result is not None
    behavior, chosen, score = result
    assert behavior.name == "high" and chosen is not None and score > 4.0


def test_cooldown_excludes_recent_behavior() -> None:
    t = WorldStateTracker()
    p = _policy((_b("attack", "enemy", 5.0, cooldown=1.0),), tick_hz=10.0)
    ws1 = t.snapshot((_det("enemy", 0.9),), _ROI)
    assert p.choose(ws1) is not None
    t.record("attack")
    ws2 = t.snapshot((_det("enemy", 0.9),), _ROI)  # tick 2, cooldown=10 ticks
    assert p.choose(ws2) is None  # still cooling down


def test_hesitation_returns_none() -> None:
    p = _policy((_b("attack", "enemy", 5.0),), hesitation_prob=1.0)
    ws = WorldState(detections=(_det("enemy", 0.9),), roi=_ROI, tick=1, recent=())
    assert p.choose(ws) is None


def test_anti_loop_penalizes_repeated_behavior() -> None:
    recent = tuple(("spam", t) for t in range(1, 6))
    ws = WorldState(detections=(_det("enemy", 0.9), _det("ally", 0.9)),
                    roi=_ROI, tick=6, recent=recent)
    p = _policy((_b("spam", "enemy", 5.0), _b("rare", "ally", 1.0)),
                anti_loop_window=10, anti_loop_max_repeats=2)
    result = p.choose(ws)
    assert result is not None and result[0].name == "rare"


def test_choose_is_deterministic_with_same_seed() -> None:
    ws = WorldState(detections=(_det("enemy", 0.9),), roi=_ROI, tick=1, recent=())
    r1 = _policy((_b("a", "enemy", 3.0), _b("b", "enemy", 3.0))).choose(ws)
    r2 = _policy((_b("a", "enemy", 3.0), _b("b", "enemy", 3.0))).choose(ws)
    assert r1 is not None and r2 is not None
    assert r1[0].name == r2[0].name        # same behavior chosen
    assert r1[2] == r2[2]                   # identical score (seeded jitter)
