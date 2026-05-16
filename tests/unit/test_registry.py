# tests/unit/test_registry.py
from pathlib import Path

import pytest

from smartuibot.ai.registry import load_behaviors

_GOOD = """
behaviors:
  - name: attack
    base_utility: 3.0
    cooldown_s: 0.5
    condition: {labels: [enemy], min_confidence: 0.4}
    steps:
      - {kind: move, target: detection}
      - {kind: click, target: detection, button: left}
  - name: idle
    base_utility: 0.1
    condition: {labels: [__any__], min_confidence: 0.0, min_count: 0}
    steps:
      - {kind: wait, duration_s: 0.5}
"""


def test_load_valid_behaviors(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text(_GOOD)
    bs = load_behaviors(p)
    assert [b.name for b in bs] == ["attack", "idle"]
    assert bs[0].condition.labels == frozenset({"enemy"})
    assert bs[0].steps[1].kind == "click"


def test_rejects_bad_step_kind(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text(
        "behaviors:\n - name: x\n   base_utility: 1.0\n"
        "   condition: {labels: [a]}\n   steps:\n    - {kind: explode}\n"
    )
    with pytest.raises(ValueError):
        load_behaviors(p)


def test_rejects_nonpositive_utility(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text(
        "behaviors:\n - name: x\n   base_utility: 0\n"
        "   condition: {labels: [a]}\n   steps: []\n"
    )
    with pytest.raises(ValueError):
        load_behaviors(p)
