# tests/unit/test_motion.py
import random

from smartuibot.input.motion import (
    MotionParams,
    bezier_path,
    keystroke_delays,
    maybe_overshoot,
    reaction_delay,
)


def _params() -> MotionParams:
    return MotionParams(move_steps=20, jitter_px=2, reaction_min_s=0.05,
                        reaction_max_s=0.15, keystroke_min_s=0.02,
                        keystroke_max_s=0.08, overshoot_prob=0.5)


def test_bezier_path_starts_and_ends_at_endpoints_and_is_deterministic() -> None:
    p = _params()
    a = bezier_path((0, 0), (100, 50), params=p, rng=random.Random(7))
    b = bezier_path((0, 0), (100, 50), params=p, rng=random.Random(7))
    assert a == b  # deterministic with same seed
    assert len(a) == p.move_steps
    assert a[0] == (0, 0)
    assert a[-1] == (100, 50)  # exact final landing


def test_reaction_delay_in_range_and_seeded() -> None:
    p = _params()
    d = reaction_delay(p, random.Random(3))
    assert p.reaction_min_s <= d <= p.reaction_max_s
    assert reaction_delay(p, random.Random(3)) == d


def test_keystroke_delays_length_and_bounds() -> None:
    p = _params()
    ds = keystroke_delays(4, p, random.Random(9))
    assert len(ds) == 4
    assert all(p.keystroke_min_s <= d <= p.keystroke_max_s for d in ds)


def test_maybe_overshoot_returns_point_or_none_seeded() -> None:
    p = _params()
    r1 = maybe_overshoot((100, 100), p, random.Random(0))
    r2 = maybe_overshoot((100, 100), p, random.Random(0))
    assert r1 == r2  # deterministic
    assert r1 is None or (isinstance(r1, tuple) and len(r1) == 2)
