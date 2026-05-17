# tests/unit/test_behavior.py
from smartuibot.ai.behavior import Behavior, BehaviorStep, Condition, resolve_steps
from smartuibot.ai.world_state import WorldState
from smartuibot.core.types import ROI, Detection

_ROI = ROI(monitor=1, x=5, y=7, width=100, height=80)


def _det(label: str, conf: float, box: tuple[int, int, int, int]) -> Detection:
    return Detection(label=label, confidence=conf, class_id=0,
                     x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def test_condition_match_returns_chosen_detection() -> None:
    ws = WorldState(detections=(_det("enemy", 0.8, (0, 0, 20, 40)),),
                    roi=_ROI, tick=1, recent=())
    c = Condition(labels=frozenset({"enemy"}), min_confidence=0.5)
    chosen = c.match(ws)
    assert chosen is not None and chosen.label == "enemy"
    assert Condition(labels=frozenset({"boss"})).match(ws) is None


def test_resolve_steps_detection_centroid_and_roi_center_and_fixed() -> None:
    chosen = _det("enemy", 0.8, (10, 20, 30, 60))  # centroid (20, 40) in frame px
    steps = (
        BehaviorStep(kind="move", target="detection"),
        BehaviorStep(kind="click", target="detection", button="left"),
        BehaviorStep(kind="move", target="roi_center"),
        BehaviorStep(kind="key", key="space"),
        BehaviorStep(kind="wait", duration_s=0.3),
        BehaviorStep(kind="click", target="fixed", x=4, y=9),
    )
    out = resolve_steps(steps, chosen, _ROI)
    assert (out[0].kind, out[0].x, out[0].y) == ("move", 20, 40)
    assert (out[1].kind, out[1].x, out[1].y, out[1].button) == ("click", 20, 40, "left")
    assert (out[2].x, out[2].y) == (50, 40)  # roi center = (w//2, h//2)
    assert out[3].kind == "key" and out[3].key == "space"
    assert out[4].kind == "wait" and out[4].duration_s == 0.3
    assert (out[5].x, out[5].y) == (4, 9)


def test_behavior_is_frozen_value() -> None:
    b = Behavior(name="attack", condition=Condition(labels=frozenset({"enemy"})),
                 base_utility=2.0, cooldown_s=1.0,
                 steps=(BehaviorStep(kind="click", target="detection"),))
    assert b.name == "attack" and b.base_utility == 2.0 and b.scale_by_confidence is True


def test_condition_text_any_filters_match() -> None:
    d = _det("button", 0.9, (0, 0, 10, 10))
    d = Detection(label=d.label, confidence=d.confidence, class_id=0,
                  x1=0, y1=0, x2=10, y2=10, text="CLOSE")
    ws = WorldState(detections=(d,), roi=_ROI, tick=1, recent=())
    assert Condition(labels=frozenset({"button"}),
                     text_any=frozenset({"close"})).match(ws) is not None
    assert Condition(labels=frozenset({"button"}),
                     text_any=frozenset({"buy"})).match(ws) is None
    # default (no text_any) unchanged
    assert Condition(labels=frozenset({"button"})).match(ws) is not None
