# tests/unit/test_world_state.py
from smartuibot.ai.world_state import WorldState, WorldStateTracker
from smartuibot.core.types import ROI, Detection

_ROI = ROI(monitor=1, x=0, y=0, width=100, height=100)


def _det(label: str, conf: float) -> Detection:
    return Detection(label=label, confidence=conf, class_id=0, x1=0, y1=0, x2=10, y2=10)


def test_best_match_picks_highest_confidence_over_threshold() -> None:
    ws = WorldState(detections=(_det("enemy", 0.4), _det("enemy", 0.9)),
                    roi=_ROI, tick=1, recent=())
    m = ws.best_match(frozenset({"enemy"}), min_confidence=0.5, min_count=1)
    assert m is not None and m.confidence == 0.9
    assert ws.best_match(frozenset({"enemy"}), 0.95, 1) is None
    assert ws.best_match(frozenset({"ally"}), 0.0, 1) is None


def test_tracker_increments_tick_and_tracks_recent() -> None:
    t = WorldStateTracker(recent_size=8)
    ws1 = t.snapshot((_det("enemy", 0.8),), _ROI)
    assert ws1.tick == 1 and ws1.ticks_since("attack") is None
    t.record("attack")
    ws2 = t.snapshot((), _ROI)
    assert ws2.tick == 2
    assert ws2.ticks_since("attack") == 1
    assert ws2.recent_count("attack", window=10) == 1
    assert ws2.recent_count("attack", window=0) == 0

