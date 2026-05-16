# SmartUIBot Slice B Implementation Plan (Decision Engine + Input Automation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop on Slice A — consume `DetectionsReady`, decide via a utility-AI policy over a config-driven behavior registry, and perform real, humanized mouse/keyboard input, with hard safety controls and full headless testability.

**Architecture:** Decoupled services over the existing thread-safe `EventBus`. `DecisionService` (a `Service`) builds an immutable `WorldState`, runs `UtilityPolicy` over `BehaviorRegistry`, and emits `ActionRequested`. `ActionService` (a `Service`, own thread) executes resolved steps through an `InputBackend` Protocol using pure `HumanizedMotion` plans. A shared `ModeFSM` gates injection (DISARMED by default). Only `ActionService` + backends touch the OS; everything else is unit/integration-testable headless with a `RecordingInputBackend`.

**Tech Stack:** Python 3.12, builds on Slice A (`smartuibot.core.*`, `Service`, `EventBus`), `pynput` (macOS/Linux), `pydirectinput` (Windows, deferred run-test), `PyYAML`, pytest/ruff/mypy --strict.

---

## Conventions (apply to EVERY task)

- Work in the project venv: `.venv/bin/python|pytest|ruff|mypy`. Branch is created at execution time (see Execution Handoff). Do not switch branches mid-plan.
- TDD: write failing test → see it fail → minimal impl → see it pass → gate → commit. One task = one commit.
- **PEP 695 generics** (`class X[T]:`), never `Generic[T]` (ruff UP046). Function-level `TypeVar` is fine.
- Add `-> None` to every test function (mypy --strict). Annotate `tmp_path: Path`, `monkeypatch: pytest.MonkeyPatch` where used.
- Per task run `.venv/bin/ruff check <files>` (clean) and `.venv/bin/mypy` ("Success: no issues found"). For untyped libs, the `[[tool.mypy.overrides]]` block already covers `mss/ultralytics/cv2/pynput`; **add `pydirectinput`** in Task 12.
- Commit message ends with a blank line then: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (use `git -c user.email="lacost.st@gmail.com" -c user.name="Vadim Shevchenko" commit ...` if git complains about identity).
- Do NOT modify Slice-A read-only components' behavior. Slice-A files are only *extended* where this plan says "Modify".
- All randomness via an injected `random.Random` (seedable) — never module-level `random`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/smartuibot/core/types.py` | **Modify**: add `ActionStep` value type |
| `src/smartuibot/core/events.py` | **Modify**: add `ActionRequested/ActionStarted/ActionCompleted/ActionAborted/ModeChanged` |
| `src/smartuibot/core/config.py` | **Modify**: add `DecisionConfig`, `InputConfig`, `behaviors_path` |
| `src/smartuibot/core/container.py` | **Modify**: wire ModeFSM + decision + action services + input backend |
| `src/smartuibot/ai/world_state.py` | `WorldState` snapshot + `WorldStateTracker` (tick + recent ring) |
| `src/smartuibot/ai/behavior.py` | `Condition`, `BehaviorStep`, `Behavior`, `resolve_steps` |
| `src/smartuibot/ai/registry.py` | `load_behaviors` (YAML → validated `Behavior`s, no code-eval) |
| `src/smartuibot/ai/mode.py` | `Mode` constants + thread-safe `ModeFSM` |
| `src/smartuibot/ai/utility.py` | `UtilityPolicy` (score/cooldown/anti-loop/hesitation, seeded) |
| `src/smartuibot/ai/service.py` | `DecisionService(Service)` |
| `src/smartuibot/input/backend.py` | `InputBackend` Protocol |
| `src/smartuibot/input/motion.py` | `MotionParams` + pure `HumanizedMotion` plan functions |
| `src/smartuibot/input/pynput_backend.py` | `PynputBackend` (real, thin) |
| `src/smartuibot/input/pydirectinput_backend.py` | `PyDirectInputBackend` (Windows, deferred run-test) |
| `src/smartuibot/input/service.py` | `ActionService(Service)` |
| `src/smartuibot/platform_support/detect.py` | **Modify**: add `resolve_input_backend_name` |
| `src/smartuibot/ui/controls.py` | **Modify**: ARM/DISARM in `UiController` + `ControlBar` |
| `src/smartuibot/ui/debug_window.py` | **Modify**: show mode + action timeline |
| `src/smartuibot/app.py` | **Modify**: input backend factory, ModeFSM wiring, e-stop → disarm/abort |
| `configs/default.yaml` | **Modify**: `decision:` + `input:` + `behaviors_path` |
| `configs/behaviors.yaml` | **Create**: example game-agnostic behaviors |
| `tests/fakes/input.py` | `RecordingInputBackend` |

---

### Task 1: `ActionStep` value type

**Files:**
- Modify: `src/smartuibot/core/types.py`
- Test: `tests/unit/test_action_step.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_action_step.py
import pytest

from smartuibot.core.types import ActionStep


def test_action_step_defaults() -> None:
    s = ActionStep(kind="wait", duration_s=0.5)
    assert s.kind == "wait" and s.duration_s == 0.5
    assert s.x == 0 and s.y == 0 and s.button == "left" and s.key == ""


def test_action_step_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ActionStep(kind="teleport")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_action_step.py -q`
Expected: FAIL — `ImportError: cannot import name 'ActionStep'`

- [ ] **Step 3: Append to `src/smartuibot/core/types.py`** (add at end of file, keep existing content):

```python
_ACTION_KINDS = frozenset({"move", "click", "key", "wait"})


@dataclass(frozen=True, slots=True)
class ActionStep:
    kind: str
    x: int = 0
    y: int = 0
    button: str = "left"
    key: str = ""
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _ACTION_KINDS:
            raise ValueError(f"ActionStep.kind must be one of {sorted(_ACTION_KINDS)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_action_step.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/core/types.py tests/unit/test_action_step.py
.venv/bin/mypy
git add src/smartuibot/core/types.py tests/unit/test_action_step.py
git commit -m "feat(core): ActionStep value type"
```
Expected: ruff clean; mypy Success.

---

### Task 2: Slice-B events

**Files:**
- Modify: `src/smartuibot/core/events.py`
- Test: `tests/unit/test_slice_b_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_slice_b_events.py
from smartuibot.core.events import (
    ActionAborted, ActionCompleted, ActionRequested, ActionStarted, Event, ModeChanged,
)
from smartuibot.core.types import ROI, ActionStep


def test_action_requested_carries_steps_roi_priority() -> None:
    roi = ROI(monitor=1, x=0, y=0, width=10, height=10)
    e = ActionRequested(behavior_name="attack",
                        steps=(ActionStep(kind="click", x=1, y=2),),
                        roi=roi, priority=3.0)
    assert isinstance(e, Event)
    assert e.behavior_name == "attack" and e.priority == 3.0
    assert e.steps[0].kind == "click" and e.roi == roi


def test_other_events_subclass_event() -> None:
    for ev in (ActionStarted("a"), ActionCompleted("a"),
               ActionAborted("a", "disarmed"), ModeChanged("armed")):
        assert isinstance(ev, Event)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_slice_b_events.py -q`
Expected: FAIL — `ImportError: cannot import name 'ActionRequested'`

- [ ] **Step 3: Modify `src/smartuibot/core/events.py`**

Change the import line `from smartuibot.core.types import Detection, Frame` to:
```python
from smartuibot.core.types import ROI, ActionStep, Detection, Frame
```
Then append at end of file:
```python
@dataclass(frozen=True, slots=True)
class ActionRequested(Event):
    behavior_name: str
    steps: tuple[ActionStep, ...]
    roi: ROI
    priority: float


@dataclass(frozen=True, slots=True)
class ActionStarted(Event):
    behavior_name: str


@dataclass(frozen=True, slots=True)
class ActionCompleted(Event):
    behavior_name: str


@dataclass(frozen=True, slots=True)
class ActionAborted(Event):
    behavior_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ModeChanged(Event):
    mode: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_slice_b_events.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/core/events.py tests/unit/test_slice_b_events.py
.venv/bin/mypy
git add src/smartuibot/core/events.py tests/unit/test_slice_b_events.py
git commit -m "feat(core): Slice-B action + mode events"
```

---

### Task 3: `WorldState` + `WorldStateTracker`

**Files:**
- Create: `src/smartuibot/ai/world_state.py`
- Test: `tests/unit/test_world_state.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_world_state.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ai.world_state`

- [ ] **Step 3: Create `src/smartuibot/ai/world_state.py`**

```python
# src/smartuibot/ai/world_state.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from smartuibot.core.types import ROI, Detection


@dataclass(frozen=True, slots=True)
class WorldState:
    detections: tuple[Detection, ...]
    roi: ROI
    tick: int
    recent: tuple[tuple[str, int], ...]  # (behavior_name, tick), oldest first

    def best_match(
        self, labels: frozenset[str], min_confidence: float, min_count: int
    ) -> Detection | None:
        matches = sorted(
            (d for d in self.detections
             if d.label in labels and d.confidence >= min_confidence),
            key=lambda d: d.confidence,
            reverse=True,
        )
        if len(matches) < min_count:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_world_state.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ai/world_state.py tests/unit/test_world_state.py
.venv/bin/mypy
git add src/smartuibot/ai/world_state.py tests/unit/test_world_state.py
git commit -m "feat(ai): WorldState snapshot + tracker"
```

---

### Task 4: `Condition`, `BehaviorStep`, `Behavior`, `resolve_steps`

**Files:**
- Create: `src/smartuibot/ai/behavior.py`
- Test: `tests/unit/test_behavior.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_behavior.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ai.behavior`

- [ ] **Step 3: Create `src/smartuibot/ai/behavior.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_behavior.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ai/behavior.py tests/unit/test_behavior.py
.venv/bin/mypy
git add src/smartuibot/ai/behavior.py tests/unit/test_behavior.py
git commit -m "feat(ai): Condition, Behavior, resolve_steps"
```

---

### Task 5: `BehaviorRegistry` (YAML loader, validated, no code-eval)

**Files:**
- Create: `src/smartuibot/ai/registry.py`
- Test: `tests/unit/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ai.registry`

- [ ] **Step 3: Create `src/smartuibot/ai/registry.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_registry.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ai/registry.py tests/unit/test_registry.py
.venv/bin/mypy
git add src/smartuibot/ai/registry.py tests/unit/test_registry.py
git commit -m "feat(ai): YAML behavior registry (validated, no code-eval)"
```

---

### Task 6: `ModeFSM`

**Files:**
- Create: `src/smartuibot/ai/mode.py`
- Test: `tests/unit/test_mode.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_mode.py
from smartuibot.ai.mode import Mode, ModeFSM


def test_starts_disarmed_and_arms() -> None:
    m = ModeFSM()
    assert m.mode == Mode.DISARMED
    assert m.is_armed() is False
    assert m.arm() is True
    assert m.mode == Mode.ARMED and m.is_armed() is True
    assert m.arm() is False  # already armed → no transition


def test_pause_resume_only_from_valid_states() -> None:
    m = ModeFSM()
    assert m.pause() is False  # cannot pause while disarmed
    m.arm()
    assert m.pause() is True and m.mode == Mode.PAUSED and not m.is_armed()
    assert m.resume() is True and m.mode == Mode.ARMED


def test_disarm_from_any_state() -> None:
    m = ModeFSM()
    m.arm()
    assert m.disarm() is True and m.mode == Mode.DISARMED
    assert m.disarm() is False  # already disarmed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_mode.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ai.mode`

- [ ] **Step 3: Create `src/smartuibot/ai/mode.py`**

```python
# src/smartuibot/ai/mode.py
from __future__ import annotations

import threading


class Mode:
    DISARMED = "disarmed"
    ARMED = "armed"
    PAUSED = "paused"


class ModeFSM:
    """Thread-safe coarse mode gate. Injection is allowed only in ARMED."""

    def __init__(self) -> None:
        self._mode = Mode.DISARMED
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def is_armed(self) -> bool:
        with self._lock:
            return self._mode == Mode.ARMED

    def arm(self) -> bool:
        with self._lock:
            if self._mode in (Mode.DISARMED, Mode.PAUSED):
                self._mode = Mode.ARMED
                return True
            return False

    def disarm(self) -> bool:
        with self._lock:
            if self._mode != Mode.DISARMED:
                self._mode = Mode.DISARMED
                return True
            return False

    def pause(self) -> bool:
        with self._lock:
            if self._mode == Mode.ARMED:
                self._mode = Mode.PAUSED
                return True
            return False

    def resume(self) -> bool:
        with self._lock:
            if self._mode == Mode.PAUSED:
                self._mode = Mode.ARMED
                return True
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_mode.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ai/mode.py tests/unit/test_mode.py
.venv/bin/mypy
git add src/smartuibot/ai/mode.py tests/unit/test_mode.py
git commit -m "feat(ai): thread-safe ModeFSM gate"
```

---

### Task 7: `UtilityPolicy`

**Files:**
- Create: `src/smartuibot/ai/utility.py`
- Test: `tests/unit/test_utility.py`

- [ ] **Step 1: Write the failing test**

```python
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
    # "spam" repeated > max in window gets penalized below "rare"
    recent = tuple(("spam", t) for t in range(1, 6))
    ws = WorldState(detections=(_det("enemy", 0.9), _det("ally", 0.9)),
                    roi=_ROI, tick=6, recent=recent)
    p = _policy((_b("spam", "enemy", 5.0), _b("rare", "ally", 1.0)),
                anti_loop_window=10, anti_loop_max_repeats=2)
    result = p.choose(ws)
    assert result is not None and result[0].name == "rare"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_utility.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ai.utility`

- [ ] **Step 3: Create `src/smartuibot/ai/utility.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_utility.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ai/utility.py tests/unit/test_utility.py
.venv/bin/mypy
git add src/smartuibot/ai/utility.py tests/unit/test_utility.py
git commit -m "feat(ai): UtilityPolicy (score/cooldown/anti-loop/hesitation)"
```

---

### Task 8: `DecisionService`

**Files:**
- Create: `src/smartuibot/ai/service.py`
- Test: `tests/unit/test_decision_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_decision_service.py
import random
import time

import numpy as np

from smartuibot.ai.behavior import Behavior, BehaviorStep, Condition
from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.service import DecisionService
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ActionRequested, DetectionsReady
from smartuibot.core.types import ROI, Detection, Frame

_ROI = ROI(monitor=1, x=0, y=0, width=8, height=8)


def _frame() -> Frame:
    return Frame(image=np.zeros((8, 8, 3), dtype=np.uint8),
                 timestamp=time.monotonic(), seq=1, roi=_ROI)


def _enemy() -> Detection:
    return Detection(label="enemy", confidence=0.9, class_id=0, x1=2, y1=2, x2=6, y2=6)


def _svc(bus: EventBus, mode: ModeFSM) -> DecisionService:
    behaviors = (Behavior(name="attack",
                          condition=Condition(labels=frozenset({"enemy"})),
                          base_utility=5.0,
                          steps=(BehaviorStep(kind="click", target="detection"),),
                          scale_by_confidence=False),)
    policy = UtilityPolicy(behaviors, tick_hz=50.0, anti_loop_window=5,
                           anti_loop_max_repeats=99, hesitation_prob=0.0,
                           rng=random.Random(1))
    return DecisionService(bus=bus, policy=policy, tracker=WorldStateTracker(),
                           mode=mode, tick_hz=50.0)


def test_no_actions_when_disarmed() -> None:
    bus = EventBus()
    out: list[ActionRequested] = []
    bus.subscribe(ActionRequested, out.append)
    mode = ModeFSM()  # DISARMED
    svc = _svc(bus, mode)
    svc.start()
    bus.publish(DetectionsReady(frame=_frame(), detections=(_enemy(),)))
    time.sleep(0.15)
    svc.stop()
    assert out == []


def test_emits_action_when_armed() -> None:
    bus = EventBus()
    out: list[ActionRequested] = []
    bus.subscribe(ActionRequested, out.append)
    mode = ModeFSM()
    mode.arm()
    svc = _svc(bus, mode)
    svc.start()
    bus.publish(DetectionsReady(frame=_frame(), detections=(_enemy(),)))
    time.sleep(0.2)
    svc.stop()
    assert out
    assert out[0].behavior_name == "attack"
    assert out[0].steps[0].kind == "click"
    assert (out[0].steps[0].x, out[0].steps[0].y) == (4, 4)  # centroid of (2,2,6,6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_decision_service.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ai.service`

- [ ] **Step 3: Create `src/smartuibot/ai/service.py`**

```python
# src/smartuibot/ai/service.py
from __future__ import annotations

import threading
import time

from smartuibot.ai.behavior import resolve_steps
from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ActionRequested, DetectionsReady
from smartuibot.core.service import Service
from smartuibot.core.types import ROI, Detection


class DecisionService(Service):
    def __init__(
        self,
        bus: EventBus,
        policy: UtilityPolicy,
        tracker: WorldStateTracker,
        mode: ModeFSM,
        tick_hz: float,
    ) -> None:
        super().__init__(name="decision", bus=bus)
        self._policy = policy
        self._tracker = tracker
        self._mode = mode
        self._period = 1.0 / tick_hz
        self._lock = threading.Lock()
        self._latest: tuple[tuple[Detection, ...], ROI] | None = None
        bus.subscribe(DetectionsReady, self._on_detections)

    def _on_detections(self, event: DetectionsReady) -> None:
        with self._lock:
            self._latest = (event.detections, event.frame.roi)

    def run_once(self) -> None:
        start = time.monotonic()
        if not self._mode.is_armed():
            time.sleep(self._period)
            return
        with self._lock:
            latest = self._latest
        if latest is None:
            time.sleep(self._period)
            return
        detections, roi = latest
        ws = self._tracker.snapshot(detections, roi)
        result = self._policy.choose(ws)
        if result is not None:
            behavior, chosen, score = result
            steps = resolve_steps(behavior.steps, chosen, roi)
            self._tracker.record(behavior.name)
            self._bus.publish(ActionRequested(
                behavior_name=behavior.name, steps=steps, roi=roi, priority=score))
        elapsed = time.monotonic() - start
        if elapsed < self._period:
            time.sleep(self._period - elapsed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_decision_service.py -q`
Expected: PASS (2 passed). Run twice for stability (timing test).

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ai/service.py tests/unit/test_decision_service.py
.venv/bin/mypy
git add src/smartuibot/ai/service.py tests/unit/test_decision_service.py
git commit -m "feat(ai): DecisionService (armed-gated, emits ActionRequested)"
```

---

### Task 9: `InputBackend` Protocol + `RecordingInputBackend`

**Files:**
- Create: `src/smartuibot/input/backend.py`
- Create: `tests/fakes/input.py`
- Test: `tests/unit/test_recording_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_recording_backend.py
from smartuibot.input.backend import InputBackend
from tests.fakes.input import RecordingInputBackend


def test_recording_backend_satisfies_protocol_and_records() -> None:
    be: InputBackend = RecordingInputBackend()
    be.move_to(10, 20)
    be.click("left")
    be.key_down("a")
    be.key_up("a")
    be.type_text("hi")
    assert be.calls == [
        ("move_to", (10, 20)),
        ("click", ("left",)),
        ("key_down", ("a",)),
        ("key_up", ("a",)),
        ("type_text", ("hi",)),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_recording_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.input.backend`

- [ ] **Step 3: Create files**

```python
# src/smartuibot/input/backend.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class InputBackend(Protocol):
    def move_to(self, x: int, y: int) -> None: ...
    def mouse_down(self, button: str) -> None: ...
    def mouse_up(self, button: str) -> None: ...
    def click(self, button: str) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def type_text(self, text: str) -> None: ...
```

```python
# tests/fakes/input.py
from __future__ import annotations

from typing import Any


class RecordingInputBackend:
    """Records calls instead of touching the OS; lets the loop run headless."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def move_to(self, x: int, y: int) -> None:
        self.calls.append(("move_to", (x, y)))

    def mouse_down(self, button: str) -> None:
        self.calls.append(("mouse_down", (button,)))

    def mouse_up(self, button: str) -> None:
        self.calls.append(("mouse_up", (button,)))

    def click(self, button: str) -> None:
        self.calls.append(("click", (button,)))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", (key,)))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", (key,)))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", (text,)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_recording_backend.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/input/backend.py tests/fakes/input.py tests/unit/test_recording_backend.py
.venv/bin/mypy
git add src/smartuibot/input/backend.py tests/fakes/input.py tests/unit/test_recording_backend.py
git commit -m "feat(input): InputBackend protocol + recording fake"
```

---

### Task 10: `HumanizedMotion` (pure, seeded)

**Files:**
- Create: `src/smartuibot/input/motion.py`
- Test: `tests/unit/test_motion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_motion.py
import random

from smartuibot.input.motion import (
    MotionParams, bezier_path, keystroke_delays, maybe_overshoot, reaction_delay,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_motion.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.input.motion`

- [ ] **Step 3: Create `src/smartuibot/input/motion.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_motion.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/input/motion.py tests/unit/test_motion.py
.venv/bin/mypy
git add src/smartuibot/input/motion.py tests/unit/test_motion.py
git commit -m "feat(input): pure seeded HumanizedMotion plans"
```

---

### Task 11: `resolve_input_backend_name`

**Files:**
- Modify: `src/smartuibot/platform_support/detect.py`
- Test: `tests/unit/test_resolve_input_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_resolve_input_backend.py
from smartuibot.platform_support.detect import resolve_input_backend_name


def test_auto_resolves_by_os() -> None:
    assert resolve_input_backend_name("auto", os_name="windows") == "pydirectinput"
    assert resolve_input_backend_name("auto", os_name="macos") == "pynput"
    assert resolve_input_backend_name("auto", os_name="linux") == "pynput"


def test_explicit_respected() -> None:
    assert resolve_input_backend_name("pynput", os_name="windows") == "pynput"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_resolve_input_backend.py -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_input_backend_name'`

- [ ] **Step 3: Append to `src/smartuibot/platform_support/detect.py`** (keep existing `current_os` / `resolve_backend_name`):

```python
def resolve_input_backend_name(configured: str, os_name: str | None = None) -> str:
    os_name = os_name or current_os()
    if configured != "auto":
        return configured
    return "pydirectinput" if os_name == "windows" else "pynput"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_resolve_input_backend.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/platform_support/detect.py tests/unit/test_resolve_input_backend.py
.venv/bin/mypy
git add src/smartuibot/platform_support/detect.py tests/unit/test_resolve_input_backend.py
git commit -m "feat(platform): input backend selection"
```

---

### Task 12: `PyDirectInputBackend` (Windows; deferred run-test) + dep/mypy

**Files:**
- Modify: `pyproject.toml`
- Create: `src/smartuibot/input/pydirectinput_backend.py`
- Test: `tests/unit/test_pydirectinput_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pydirectinput_backend.py
import sys

import pytest

from smartuibot.input.backend import InputBackend


@pytest.mark.skipif(sys.platform != "win32",
                    reason="pydirectinput is Windows-only; deferred run-test")
def test_pydirectinput_backend_satisfies_protocol() -> None:
    from smartuibot.input.pydirectinput_backend import PyDirectInputBackend

    be: InputBackend = PyDirectInputBackend()
    assert isinstance(be, InputBackend)


def test_module_imports_without_pydirectinput_installed() -> None:
    # The module must import on any OS; the dependency is imported lazily.
    import importlib

    mod = importlib.import_module("smartuibot.input.pydirectinput_backend")
    assert hasattr(mod, "PyDirectInputBackend")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_pydirectinput_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.input.pydirectinput_backend`

- [ ] **Step 3: Modify `pyproject.toml`**

In `[project] dependencies`, change the `pynput` line block to add pydirectinput as a Windows-only marker dependency — replace:
```toml
    "pynput>=1.7",
]
```
with:
```toml
    "pynput>=1.7",
    "pydirectinput>=1.0; sys_platform == 'win32'",
]
```
In the existing `[[tool.mypy.overrides]]` `module = [...]` list, add `"pydirectinput.*"` so it reads (keep the other entries):
```toml
module = ["mss.*", "ultralytics.*", "cv2.*", "pynput.*", "pydirectinput.*"]
```

- [ ] **Step 4: Create `src/smartuibot/input/pydirectinput_backend.py`**

```python
# src/smartuibot/input/pydirectinput_backend.py
from __future__ import annotations


class PyDirectInputBackend:
    """Windows input via pydirectinput (game-compatible scancodes).
    pydirectinput is imported lazily so this module imports on any OS;
    constructing it off-Windows raises a clear error."""

    def __init__(self) -> None:
        import pydirectinput

        pydirectinput.FAILSAFE = True
        self._pdi = pydirectinput

    def move_to(self, x: int, y: int) -> None:
        self._pdi.moveTo(x, y)

    def mouse_down(self, button: str) -> None:
        self._pdi.mouseDown(button=button)

    def mouse_up(self, button: str) -> None:
        self._pdi.mouseUp(button=button)

    def click(self, button: str) -> None:
        self._pdi.click(button=button)

    def key_down(self, key: str) -> None:
        self._pdi.keyDown(key)

    def key_up(self, key: str) -> None:
        self._pdi.keyUp(key)

    def type_text(self, text: str) -> None:
        self._pdi.write(text)
```

- [ ] **Step 5: Reinstall (picks up dep marker), run tests, gate, commit**

```bash
.venv/bin/pip install -q -e ".[dev]"
.venv/bin/pytest tests/unit/test_pydirectinput_backend.py -q
.venv/bin/ruff check src/smartuibot/input/pydirectinput_backend.py tests/unit/test_pydirectinput_backend.py pyproject.toml
.venv/bin/mypy
git add pyproject.toml src/smartuibot/input/pydirectinput_backend.py tests/unit/test_pydirectinput_backend.py
git commit -m "feat(input): Windows PyDirectInputBackend (deferred run-test)"
```
Expected: on macOS the Windows-only test is skipped, the import test passes; ruff/mypy clean.

---

### Task 13: `PynputBackend` (real, thin; env-gated run-test)

**Files:**
- Create: `src/smartuibot/input/pynput_backend.py`
- Test: `tests/unit/test_pynput_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pynput_backend.py
import importlib
import os

import pytest

from smartuibot.input.backend import InputBackend


def test_module_imports() -> None:
    mod = importlib.import_module("smartuibot.input.pynput_backend")
    assert hasattr(mod, "PynputBackend")


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="real input device not available in CI",
)
def test_pynput_backend_constructs_and_satisfies_protocol() -> None:
    from smartuibot.input.pynput_backend import PynputBackend

    be: InputBackend = PynputBackend()
    assert isinstance(be, InputBackend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_pynput_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.input.pynput_backend`

- [ ] **Step 3: Create `src/smartuibot/input/pynput_backend.py`**

```python
# src/smartuibot/input/pynput_backend.py
from __future__ import annotations

from typing import Any


class PynputBackend:
    """Real mouse/keyboard via pynput (macOS/Linux + universal).
    pynput controllers are created lazily so the module imports headless."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller as KeyboardController
        from pynput.mouse import Button, Controller as MouseController

        self._mouse = MouseController()
        self._kbd = KeyboardController()
        self._buttons: dict[str, Any] = {
            "left": Button.left, "right": Button.right, "middle": Button.middle,
        }

    def _btn(self, button: str) -> Any:
        return self._buttons.get(button, self._buttons["left"])

    def move_to(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def mouse_down(self, button: str) -> None:
        self._mouse.press(self._btn(button))

    def mouse_up(self, button: str) -> None:
        self._mouse.release(self._btn(button))

    def click(self, button: str) -> None:
        self._mouse.click(self._btn(button), 1)

    def key_down(self, key: str) -> None:
        self._kbd.press(key)

    def key_up(self, key: str) -> None:
        self._kbd.release(key)

    def type_text(self, text: str) -> None:
        self._kbd.type(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_pynput_backend.py -q`
Expected: PASS (2 passed locally; the second test may be skipped under `CI=true`).

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/input/pynput_backend.py tests/unit/test_pynput_backend.py
.venv/bin/mypy
git add src/smartuibot/input/pynput_backend.py tests/unit/test_pynput_backend.py
git commit -m "feat(input): real PynputBackend (env-gated run-test)"
```

---

### Task 14: `ActionService` (humanized exec, ROI-confine, abort, preemption)

**Files:**
- Create: `src/smartuibot/input/service.py`
- Test: `tests/unit/test_action_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_action_service.py
import random
import time

from smartuibot.ai.mode import ModeFSM
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import (
    ActionAborted, ActionCompleted, ActionRequested, ActionStarted,
)
from smartuibot.core.types import ROI, ActionStep
from smartuibot.input.motion import MotionParams
from smartuibot.input.service import ActionService
from tests.fakes.input import RecordingInputBackend

_ROI = ROI(monitor=1, x=100, y=50, width=40, height=30)


def _params() -> MotionParams:
    return MotionParams(move_steps=4, jitter_px=0, reaction_min_s=0.0,
                        reaction_max_s=0.0, keystroke_min_s=0.0,
                        keystroke_max_s=0.0, overshoot_prob=0.0)


def _svc(bus: EventBus, mode: ModeFSM, backend: RecordingInputBackend) -> ActionService:
    return ActionService(bus=bus, backend=backend, mode=mode, motion=_params(),
                          max_actions_per_second=1000.0, roi_confine=True,
                          rng=random.Random(0))


def test_executes_click_at_roi_offset_screen_coords() -> None:
    bus = EventBus()
    backend = RecordingInputBackend()
    mode = ModeFSM(); mode.arm()
    started: list[ActionStarted] = []
    completed: list[ActionCompleted] = []
    bus.subscribe(ActionStarted, started.append)
    bus.subscribe(ActionCompleted, completed.append)
    svc = _svc(bus, mode, backend)
    svc.start()
    bus.publish(ActionRequested(
        behavior_name="attack",
        steps=(ActionStep(kind="move", x=10, y=10),
               ActionStep(kind="click", x=10, y=10, button="left")),
        roi=_ROI, priority=5.0))
    time.sleep(0.3)
    svc.stop()
    assert started and completed
    # frame (10,10) -> screen (roi.x+10, roi.y+10) = (110, 60)
    assert ("click", ("left",)) in backend.calls
    assert backend.calls[-2:][0][0] == "move_to"
    last_move = [c for c in backend.calls if c[0] == "move_to"][-1]
    assert last_move == ("move_to", (110, 60))


def test_roi_confine_clamps_out_of_bounds_target() -> None:
    bus = EventBus()
    backend = RecordingInputBackend()
    mode = ModeFSM(); mode.arm()
    svc = _svc(bus, mode, backend)
    svc.start()
    bus.publish(ActionRequested(
        behavior_name="x",
        steps=(ActionStep(kind="move", x=999, y=-5),),  # outside ROI 40x30
        roi=_ROI, priority=1.0))
    time.sleep(0.2)
    svc.stop()
    last_move = [c for c in backend.calls if c[0] == "move_to"][-1]
    # clamped to (roi.width-1, 0) then offset → (100+39, 50+0)
    assert last_move == ("move_to", (139, 50))


def test_disarm_mid_action_aborts_and_stops_injecting() -> None:
    bus = EventBus()
    backend = RecordingInputBackend()
    mode = ModeFSM(); mode.arm()
    aborted: list[ActionAborted] = []
    bus.subscribe(ActionAborted, aborted.append)
    # long action: many waits
    steps = tuple(ActionStep(kind="wait", duration_s=0.05) for _ in range(20))
    svc = _svc(bus, mode, backend)
    svc.start()
    bus.publish(ActionRequested(behavior_name="long", steps=steps,
                                roi=_ROI, priority=1.0))
    time.sleep(0.1)
    mode.disarm()
    time.sleep(0.2)
    svc.stop()
    assert aborted and aborted[0].reason == "disarmed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_action_service.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.input.service`

- [ ] **Step 3: Create `src/smartuibot/input/service.py`**

```python
# src/smartuibot/input/service.py
from __future__ import annotations

import random
import time

from smartuibot.ai.mode import ModeFSM
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import (
    ActionAborted, ActionCompleted, ActionRequested, ActionStarted,
)
from smartuibot.core.latest_queue import LatestQueue
from smartuibot.core.service import Service
from smartuibot.core.types import ROI, ActionStep
from smartuibot.input.backend import InputBackend
from smartuibot.input.motion import MotionParams, bezier_path, reaction_delay


def _confine(x: int, y: int, roi: ROI) -> tuple[int, int]:
    cx = min(max(x, 0), roi.width - 1)
    cy = min(max(y, 0), roi.height - 1)
    return cx, cy


class ActionService(Service):
    """Executes ActionRequested via the InputBackend with humanized motion.
    One action at a time; aborts immediately when disarmed."""

    def __init__(
        self,
        bus: EventBus,
        backend: InputBackend,
        mode: ModeFSM,
        *,
        motion: MotionParams,
        max_actions_per_second: float,
        roi_confine: bool,
        rng: random.Random,
    ) -> None:
        super().__init__(name="action", bus=bus)
        self._backend = backend
        self._mode = mode
        self._motion = motion
        self._min_interval = 1.0 / max_actions_per_second if max_actions_per_second > 0 else 0.0
        self._roi_confine = roi_confine
        self._rng = rng
        self._queue: LatestQueue[ActionRequested] = LatestQueue()
        self._cursor = (0, 0)
        bus.subscribe(ActionRequested, self._on_request)

    def _on_request(self, event: ActionRequested) -> None:
        self._queue.put(event)

    def _screen(self, x: int, y: int, roi: ROI) -> tuple[int, int]:
        if self._roi_confine:
            x, y = _confine(x, y, roi)
        return roi.x + x, roi.y + y

    def _aborted(self) -> bool:
        return self._stop.is_set() or not self._mode.is_armed()

    def run_once(self) -> None:
        req = self._queue.get(timeout=0.1)
        if req is None:
            return
        if not self._mode.is_armed():
            return
        self._bus.publish(ActionStarted(behavior_name=req.behavior_name))
        for step in req.steps:
            if self._aborted():
                self._bus.publish(ActionAborted(
                    behavior_name=req.behavior_name, reason="disarmed"))
                return
            self._exec_step(step, req.roi)
            if self._min_interval:
                time.sleep(self._min_interval)
        self._bus.publish(ActionCompleted(behavior_name=req.behavior_name))

    def _exec_step(self, step: ActionStep, roi: ROI) -> None:
        if step.kind == "move":
            target = self._screen(step.x, step.y, roi)
            for px, py in bezier_path(self._cursor, target,
                                      params=self._motion, rng=self._rng):
                if self._aborted():
                    return
                self._backend.move_to(px, py)
            self._cursor = target
        elif step.kind == "click":
            target = self._screen(step.x, step.y, roi)
            self._backend.move_to(*target)
            self._cursor = target
            time.sleep(reaction_delay(self._motion, self._rng))
            self._backend.click(step.button)
        elif step.kind == "key":
            self._backend.key_down(step.key)
            self._backend.key_up(step.key)
        else:  # wait
            time.sleep(step.duration_s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_action_service.py -q`
Expected: PASS (3 passed). Run 3x for stability (timing).

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/input/service.py tests/unit/test_action_service.py
.venv/bin/mypy
git add src/smartuibot/input/service.py tests/unit/test_action_service.py
git commit -m "feat(input): ActionService (humanized, ROI-confined, abortable)"
```

---

### Task 15: Config — `DecisionConfig` + `InputConfig` + behaviors path

**Files:**
- Modify: `src/smartuibot/core/config.py`
- Modify: `configs/default.yaml`
- Create: `configs/behaviors.yaml`
- Test: `tests/unit/test_config_slice_b.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config_slice_b.py
import textwrap
from pathlib import Path

from smartuibot.core.config import load_config

_CFG = """
capture: {backend: auto, target_fps: 60, monitor: 1}
detection: {model: yolo11n.pt, confidence: 0.35, device: auto, tracking: false, smoothing_frames: 3}
ui: {preview_max_width: 960}
logging: {level: INFO, dir: logs}
hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
decision: {tick_hz: 10.0, anti_loop_window: 30, anti_loop_max_repeats: 3, hesitation_prob: 0.02, rng_seed: 7}
input: {backend: auto, max_actions_per_second: 8.0, roi_confine: true, start_armed: false, move_steps: 24, jitter_px: 2, reaction_min_s: 0.08, reaction_max_s: 0.22, keystroke_min_s: 0.03, keystroke_max_s: 0.09, overshoot_prob: 0.15}
behaviors_path: configs/behaviors.yaml
"""


def test_loads_decision_and_input_config(tmp_path: Path) -> None:
    p = tmp_path / "d.yaml"
    p.write_text(textwrap.dedent(_CFG))
    cfg = load_config(p)
    assert cfg.decision.tick_hz == 10.0
    assert cfg.decision.rng_seed == 7
    assert cfg.input.backend == "auto"
    assert cfg.input.roi_confine is True
    assert cfg.input.start_armed is False
    assert cfg.behaviors_path == "configs/behaviors.yaml"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_config_slice_b.py -q`
Expected: FAIL — `AttributeError: 'AppConfig' object has no attribute 'decision'`

- [ ] **Step 3: Modify `src/smartuibot/core/config.py`**

Add these dataclasses before `AppConfig`:
```python
@dataclass(frozen=True, slots=True)
class DecisionConfig:
    tick_hz: float
    anti_loop_window: int
    anti_loop_max_repeats: int
    hesitation_prob: float
    rng_seed: int


@dataclass(frozen=True, slots=True)
class InputConfig:
    backend: str
    max_actions_per_second: float
    roi_confine: bool
    start_armed: bool
    move_steps: int
    jitter_px: int
    reaction_min_s: float
    reaction_max_s: float
    keystroke_min_s: float
    keystroke_max_s: float
    overshoot_prob: float
```
Add fields to `AppConfig` (after `hotkeys`):
```python
    decision: DecisionConfig
    input: InputConfig
    behaviors_path: str
```
In `load_config`, extend the returned `AppConfig(...)` with:
```python
        decision=DecisionConfig(**data["decision"]),
        input=InputConfig(**data["input"]),
        behaviors_path=str(data["behaviors_path"]),
```
Add to `AppConfig.__post_init__` validation:
```python
        if self.decision.tick_hz <= 0:
            raise ValueError("decision.tick_hz must be positive")
        if not 0.0 <= self.input.overshoot_prob <= 1.0:
            raise ValueError("input.overshoot_prob must be in [0, 1]")
```

- [ ] **Step 4: Modify `configs/default.yaml`** — append:
```yaml
decision:
  tick_hz: 10.0
  anti_loop_window: 30
  anti_loop_max_repeats: 3
  hesitation_prob: 0.02
  rng_seed: 7
input:
  backend: auto        # auto | pynput | pydirectinput
  max_actions_per_second: 8.0
  roi_confine: true
  start_armed: false
  move_steps: 24
  jitter_px: 2
  reaction_min_s: 0.08
  reaction_max_s: 0.22
  keystroke_min_s: 0.03
  keystroke_max_s: 0.09
  overshoot_prob: 0.15
behaviors_path: configs/behaviors.yaml
```

- [ ] **Step 5: Create `configs/behaviors.yaml`**:
```yaml
# Game-agnostic example behaviors. Adapt label names to your trained model.
behaviors:
  - name: attack_enemy
    base_utility: 5.0
    cooldown_s: 0.4
    condition: {labels: [enemy], min_confidence: 0.5}
    steps:
      - {kind: move, target: detection}
      - {kind: click, target: detection, button: left}
  - name: collect_reward
    base_utility: 3.0
    cooldown_s: 1.0
    condition: {labels: [reward], min_confidence: 0.5}
    steps:
      - {kind: click, target: detection, button: left}
  - name: close_popup
    base_utility: 8.0
    cooldown_s: 0.5
    condition: {labels: [popup, close_button], min_confidence: 0.5}
    steps:
      - {kind: click, target: detection, button: left}
  - name: idle
    base_utility: 0.1
    cooldown_s: 0.0
    condition: {labels: [enemy, reward, popup], min_confidence: 0.0, min_count: 0}
    scale_by_confidence: false
    steps:
      - {kind: wait, duration_s: 0.5}
```

- [ ] **Step 6: Run test, gate, commit**

```bash
.venv/bin/pytest tests/unit/test_config_slice_b.py -q
.venv/bin/ruff check src/smartuibot/core/config.py tests/unit/test_config_slice_b.py
.venv/bin/mypy
git add src/smartuibot/core/config.py configs/default.yaml configs/behaviors.yaml tests/unit/test_config_slice_b.py
git commit -m "feat(core): decision/input config + example behaviors"
```
Expected: test passes (1 passed); ruff/mypy clean. (Slice-A `test_config.py` still passes because the new keys are required only when present — verify with `.venv/bin/pytest tests/unit/test_config.py -q`; if Slice-A's minimal fixtures now fail because `decision`/`input` are required, that's expected and handled in Task 16's container test which uses full config. Do NOT modify Slice-A tests here — instead, in Step 3 give `AppConfig` safe defaults: add `= field(default=...)`? No — keep required, and in Step 6 ALSO update `tests/unit/test_config.py` fixtures is NOT allowed. Resolution: make `decision`, `input`, `behaviors_path` OPTIONAL with defaults so Slice-A configs still load.)

> **Resolution (apply in Step 3):** Instead of required fields, give the three new `AppConfig` fields defaults so existing Slice-A configs remain valid:
> ```python
>     decision: DecisionConfig = field(
>         default_factory=lambda: DecisionConfig(10.0, 30, 3, 0.02, 7))
>     input: InputConfig = field(
>         default_factory=lambda: InputConfig("auto", 8.0, True, False, 24, 2,
>                                             0.08, 0.22, 0.03, 0.09, 0.15))
>     behaviors_path: str = "configs/behaviors.yaml"
> ```
> and in `load_config` only override them when present:
> ```python
>         decision=DecisionConfig(**data["decision"]) if "decision" in data
>             else DecisionConfig(10.0, 30, 3, 0.02, 7),
>         input=InputConfig(**data["input"]) if "input" in data
>             else InputConfig("auto", 8.0, True, False, 24, 2, 0.08, 0.22,
>                              0.03, 0.09, 0.15),
>         behaviors_path=str(data.get("behaviors_path", "configs/behaviors.yaml")),
> ```
> Ensure `from dataclasses import dataclass, field` is imported in config.py. Run BOTH `tests/unit/test_config.py` and `tests/unit/test_config_slice_b.py` — both must pass.

---

### Task 16: Wire decision + action into `AppContainer`

**Files:**
- Modify: `src/smartuibot/core/container.py`
- Test: `tests/unit/test_container_slice_b.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_container_slice_b.py
import time
from pathlib import Path

from smartuibot.core.config import load_config
from smartuibot.core.container import AppContainer
from smartuibot.core.events import ActionStarted
from smartuibot.core.types import ROI
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector
from tests.fakes.input import RecordingInputBackend

_CFG = """
capture: {backend: auto, target_fps: 120, monitor: 1}
detection: {model: yolo11n.pt, confidence: 0.1, device: cpu, tracking: false, smoothing_frames: 1}
ui: {preview_max_width: 960}
logging: {level: INFO, dir: %LOGS%}
hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
decision: {tick_hz: 50.0, anti_loop_window: 5, anti_loop_max_repeats: 99, hesitation_prob: 0.0, rng_seed: 1}
input: {backend: auto, max_actions_per_second: 1000.0, roi_confine: true, start_armed: true, move_steps: 3, jitter_px: 0, reaction_min_s: 0.0, reaction_max_s: 0.0, keystroke_min_s: 0.0, keystroke_max_s: 0.0, overshoot_prob: 0.0}
behaviors_path: %BEH%
"""

_BEH = """
behaviors:
  - name: attack
    base_utility: 5.0
    condition: {labels: [enemy], min_confidence: 0.1}
    scale_by_confidence: false
    steps:
      - {kind: click, target: detection, button: left}
"""


def test_container_runs_full_closed_loop_with_fakes(tmp_path: Path) -> None:
    beh = tmp_path / "behaviors.yaml"
    beh.write_text(_BEH)
    cfg_path = tmp_path / "d.yaml"
    cfg_path.write_text(_CFG.replace("%LOGS%", str(tmp_path / "logs"))
                            .replace("%BEH%", str(beh)))
    cfg = load_config(cfg_path)
    backend = RecordingInputBackend()
    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=16, height=16),
        capture_backend=FakeCaptureBackend(),
        detector=FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 8, 8)]] * 400),
        input_backend=backend,
    )
    started: list[ActionStarted] = []
    container.bus.subscribe(ActionStarted, started.append)
    container.mode.arm()
    container.start()
    time.sleep(0.6)
    container.stop()
    assert started, "expected the closed loop to trigger an action"
    assert ("click", ("left",)) in backend.calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_container_slice_b.py -q`
Expected: FAIL — `TypeError: AppContainer.__init__() got an unexpected keyword argument 'input_backend'`

- [ ] **Step 3: Modify `src/smartuibot/core/container.py`**

Add imports:
```python
import random

from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.registry import load_behaviors
from smartuibot.ai.service import DecisionService
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.input.backend import InputBackend
from smartuibot.input.motion import MotionParams
from smartuibot.input.service import ActionService
```
Change `__init__` signature to add `input_backend: InputBackend` (after `detector`). After the existing `self.detection = DetectionService(...)` line and before `self.watchdog = ...`, insert:
```python
        self.mode = ModeFSM()
        if config.input.start_armed:
            self.mode.arm()
        ic = config.input
        dc = config.decision
        rng = random.Random(dc.rng_seed)
        behaviors = load_behaviors(Path(config.behaviors_path))
        policy = UtilityPolicy(
            behaviors, tick_hz=dc.tick_hz,
            anti_loop_window=dc.anti_loop_window,
            anti_loop_max_repeats=dc.anti_loop_max_repeats,
            hesitation_prob=dc.hesitation_prob, rng=rng)
        self.decision = DecisionService(
            bus=self.bus, policy=policy, tracker=WorldStateTracker(),
            mode=self.mode, tick_hz=dc.tick_hz)
        self.action = ActionService(
            bus=self.bus, backend=input_backend, mode=self.mode,
            motion=MotionParams(
                move_steps=ic.move_steps, jitter_px=ic.jitter_px,
                reaction_min_s=ic.reaction_min_s, reaction_max_s=ic.reaction_max_s,
                keystroke_min_s=ic.keystroke_min_s, keystroke_max_s=ic.keystroke_max_s,
                overshoot_prob=ic.overshoot_prob),
            max_actions_per_second=ic.max_actions_per_second,
            roi_confine=ic.roi_confine, rng=random.Random(dc.rng_seed + 1))
```
Change the watchdog line to include the new services:
```python
        self.watchdog = Watchdog(
            [self.capture, self.detection, self.decision, self.action],
            bus=self.bus)
```
Change `start()` to also start them (action + decision before capture):
```python
    def start(self) -> None:
        self.action.start()
        self.decision.start()
        self.detection.start()
        self.capture.start()
        self.watchdog.start()
```
Change `stop()`:
```python
    def stop(self) -> None:
        self.watchdog.stop()
        self.mode.disarm()
        self.capture.stop()
        self.detection.stop()
        self.decision.stop()
        self.action.stop()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_container_slice_b.py -q`
Expected: PASS (1 passed). Run 3x for stability.

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/pytest tests/unit/test_container.py tests/unit/test_container_slice_b.py -q
.venv/bin/ruff check src/smartuibot/core/container.py tests/unit/test_container_slice_b.py
.venv/bin/mypy
git add src/smartuibot/core/container.py tests/unit/test_container_slice_b.py
git commit -m "feat(core): wire decision+action services into AppContainer"
```
Expected: both container tests pass (Slice-A `test_container.py` must construct `AppContainer` — it will now fail because `input_backend` is required. **Resolution:** in Step 3 give `input_backend` no default but UPDATE the Slice-A test is NOT allowed; instead Slice-A `test_container.py` calls `AppContainer(... )` without `input_backend`. To avoid breaking Slice A, make `input_backend` a keyword param with a default of `None` and, when `None`, default to a `RecordingInputBackend`-equivalent no-op: import is circular with tests, so instead default to constructing a safe no-op. Implement a tiny `NoOpInputBackend` in `src/smartuibot/input/backend.py` and use it as the default.)

> **Resolution (apply in Task 9 Step 3 and Task 16 Step 3):**
> Add to `src/smartuibot/input/backend.py` (Task 9) a concrete no-op:
> ```python
> class NoOpInputBackend:
>     def move_to(self, x: int, y: int) -> None: ...
>     def mouse_down(self, button: str) -> None: ...
>     def mouse_up(self, button: str) -> None: ...
>     def click(self, button: str) -> None: ...
>     def key_down(self, key: str) -> None: ...
>     def key_up(self, key: str) -> None: ...
>     def type_text(self, text: str) -> None: ...
> ```
> In Task 16 make the param `input_backend: InputBackend | None = None` and at the top of `__init__` body `if input_backend is None: from smartuibot.input.backend import NoOpInputBackend; input_backend = NoOpInputBackend()`. This keeps Slice-A `test_container.py` (which omits `input_backend`) green while Slice-B tests inject `RecordingInputBackend`. Add a one-line test in `tests/unit/test_recording_backend.py` (Task 9) asserting `isinstance(NoOpInputBackend(), InputBackend)`.

---

### Task 17: End-to-end closed-loop integration test

**Files:**
- Create: `tests/integration/test_closed_loop.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_closed_loop.py
import random
import time

from smartuibot.ai.mode import ModeFSM
from smartuibot.ai.behavior import Behavior, BehaviorStep, Condition
from smartuibot.ai.service import DecisionService
from smartuibot.ai.utility import UtilityPolicy
from smartuibot.ai.world_state import WorldStateTracker
from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ActionAborted, ActionStarted
from smartuibot.core.types import ROI
from smartuibot.input.motion import MotionParams
from smartuibot.input.service import ActionService
from smartuibot.vision.capture.service import CaptureService
from smartuibot.vision.detect.service import DetectionService
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector
from tests.fakes.input import RecordingInputBackend

_ROI = ROI(monitor=1, x=0, y=0, width=16, height=16)


def _wire(mode: ModeFSM, backend: RecordingInputBackend) -> tuple[object, ...]:
    bus = EventBus()
    behaviors = (Behavior(name="attack",
                          condition=Condition(labels=frozenset({"enemy"}),
                                               min_confidence=0.1),
                          base_utility=5.0, scale_by_confidence=False,
                          steps=(BehaviorStep(kind="click", target="detection"),)),)
    policy = UtilityPolicy(behaviors, tick_hz=50.0, anti_loop_window=5,
                           anti_loop_max_repeats=99, hesitation_prob=0.0,
                           rng=random.Random(1))
    decision = DecisionService(bus=bus, policy=policy,
                               tracker=WorldStateTracker(), mode=mode, tick_hz=50.0)
    action = ActionService(bus=bus, backend=backend, mode=mode,
                           motion=MotionParams(2, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
                           max_actions_per_second=1000.0, roi_confine=True,
                           rng=random.Random(2))
    detection = DetectionService(detector=FakeDetector(
        scripted=[[("enemy", 0.9, 4, 4, 12, 12)]] * 500), bus=bus,
        smoothing_frames=1, confidence=0.1)
    capture = CaptureService(backend=FakeCaptureBackend(), bus=bus, roi=_ROI,
                             target_fps=120)
    return bus, capture, detection, decision, action


def test_perceive_decide_act_closed_loop_headless() -> None:
    mode = ModeFSM(); mode.arm()
    backend = RecordingInputBackend()
    bus, capture, detection, decision, action = _wire(mode, backend)
    started: list[ActionStarted] = []
    bus.subscribe(ActionStarted, started.append)
    for s in (action, decision, detection, capture):
        s.start()  # type: ignore[attr-defined]
    time.sleep(0.8)
    for s in (capture, detection, decision, action):
        s.stop()  # type: ignore[attr-defined]
    assert started, "closed loop did not produce an action"
    # centroid of (4,4,12,12) = (8,8); ROI offset 0 → screen (8,8)
    assert ("click", ("left",)) in backend.calls
    assert ("move_to", (8, 8)) in backend.calls


def test_disarm_halts_injection() -> None:
    mode = ModeFSM(); mode.arm()
    backend = RecordingInputBackend()
    bus, capture, detection, decision, action = _wire(mode, backend)
    for s in (action, decision, detection, capture):
        s.start()  # type: ignore[attr-defined]
    time.sleep(0.2)
    mode.disarm()
    n_after_disarm = len(backend.calls)
    time.sleep(0.3)
    for s in (capture, detection, decision, action):
        s.stop()  # type: ignore[attr-defined]
    # after disarm, no significant new injection (allow a tiny in-flight tail)
    assert len(backend.calls) - n_after_disarm <= 3
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_closed_loop.py -q`
Expected: PASS (2 passed). If it fails, debug the offending component with superpowers:systematic-debugging — do NOT weaken assertions. Run 3x for flakiness.

- [ ] **Step 3: Gate + commit**

```bash
.venv/bin/ruff check tests/integration/test_closed_loop.py
.venv/bin/mypy
git add tests/integration/test_closed_loop.py
git commit -m "test(integration): headless perceive-decide-act closed loop"
```

---

### Task 18: `UiController` ARM/DISARM + `ControlBar` button

**Files:**
- Modify: `src/smartuibot/ui/controls.py`
- Test: `tests/unit/test_controls_arm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_controls_arm.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from smartuibot.ui.controls import ControlBar, UiController  # noqa: E402


class _Mode:
    def __init__(self) -> None:
        self.armed = False

    def arm(self) -> bool:
        self.armed = True
        return True

    def disarm(self) -> bool:
        self.armed = False
        return True

    def is_armed(self) -> bool:
        return self.armed


class _Container:
    def __init__(self) -> None:
        self.mode = _Mode()
        self.capture = type("C", (), {"pause": lambda s: None,
                                      "resume": lambda s: None,
                                      "set_roi": lambda s, r: None})()
        self.detection = type("D", (), {"pause": lambda s: None,
                                        "resume": lambda s: None,
                                        "set_confidence": lambda s, v: None,
                                        "reload_model": lambda s, p: None})()

    def start(self) -> None: ...
    def stop(self) -> None: ...


def test_controller_arm_disarm_toggles_mode(tmp_path: Path) -> None:
    c = UiController(container=_Container(), state_path=tmp_path / "s.yaml",
                     save_roi=lambda p, r: None)
    assert c.is_armed() is False
    assert c.toggle_arm() == "armed"
    assert c.container.mode.is_armed() is True
    assert c.toggle_arm() == "disarmed"
    assert c.container.mode.is_armed() is False


def test_control_bar_has_arm_button(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    c = UiController(container=_Container(), state_path=tmp_path / "s.yaml",
                     save_roi=lambda p, r: None)
    bar = ControlBar(controller=c, model_path="yolo11n.pt")
    bar.arm_btn.click()
    assert c.container.mode.is_armed() is True
    assert bar.arm_btn.text() == "Disarm"
    bar.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_controls_arm.py -q`
Expected: FAIL — `AttributeError: 'UiController' object has no attribute 'toggle_arm'`

- [ ] **Step 3: Modify `src/smartuibot/ui/controls.py`**

Add to `UiController` (after `stop`):
```python
    def is_armed(self) -> bool:
        return bool(self.container.mode.is_armed())

    def toggle_arm(self) -> str:
        if self.container.mode.is_armed():
            self.container.mode.disarm()
            return "disarmed"
        self.container.mode.arm()
        return "armed"
```
In `ControlBar.__init__`, after the `reload_btn` line add:
```python
        self.arm_btn = QPushButton("Arm")
        self.arm_btn.clicked.connect(self._on_arm)
```
Add `self.arm_btn` into the `for w in (...)` layout tuple (e.g. right after `self.pause_btn`). Add method:
```python
    def _on_arm(self) -> None:
        state = self._c.toggle_arm()
        self.arm_btn.setText("Disarm" if state == "armed" else "Arm")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_controls_arm.py tests/unit/test_controls.py -q`
Expected: PASS (Slice-A `test_controls.py` still green; new tests pass).

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ui/controls.py tests/unit/test_controls_arm.py
.venv/bin/mypy
git add src/smartuibot/ui/controls.py tests/unit/test_controls_arm.py
git commit -m "feat(ui): ARM/DISARM control"
```

---

### Task 19: `DebugWindow` shows mode + action timeline

**Files:**
- Modify: `src/smartuibot/ui/debug_window.py`
- Test: `tests/unit/test_debug_window_actions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_debug_window_actions.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from smartuibot.core.event_bus import EventBus  # noqa: E402
from smartuibot.core.events import (  # noqa: E402
    ActionCompleted, ActionStarted, ModeChanged,
)
from smartuibot.ui.debug_window import DebugWindow  # noqa: E402


def test_debug_window_tracks_mode_and_action_timeline() -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    bus = EventBus()
    win = DebugWindow(bus=bus, preview_max_width=320)
    bus.publish(ModeChanged(mode="armed"))
    bus.publish(ActionStarted(behavior_name="attack"))
    bus.publish(ActionCompleted(behavior_name="attack"))
    win._drain()
    assert win.mode_text() == "armed"
    assert win.action_count() == 2  # started + completed entries
    win.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_debug_window_actions.py -q`
Expected: FAIL — `AttributeError: 'DebugWindow' object has no attribute 'mode_text'`

- [ ] **Step 3: Modify `src/smartuibot/ui/debug_window.py`**

Add imports (extend the existing events import line):
```python
from smartuibot.core.events import (
    ActionAborted, ActionCompleted, ActionStarted, DetectionsReady, FpsTick,
    LogRecord, ModeChanged,
)
```
In `__init__`, after `self._det_count = 0` add:
```python
        self._mode = "disarmed"
        self._actions: list[str] = []
```
After the existing `self._logs` widget creation, add an actions list widget and put it in the right column (after the logs section):
```python
        self._mode_label = QLabel("mode: disarmed")
        self._action_list = QListWidget()
```
Add `right.addWidget(self._mode_label)` near the top of the `right` layout (before `self._fps_label`) and add `right.addWidget(QLabel("Actions:"))` and `right.addWidget(self._action_list)` after the logs widgets.
Subscribe in `__init__` (next to the other `bus.subscribe(...)` calls):
```python
        bus.subscribe(ModeChanged, self._events.put)
        bus.subscribe(ActionStarted, self._events.put)
        bus.subscribe(ActionCompleted, self._events.put)
        bus.subscribe(ActionAborted, self._events.put)
```
Add introspection helpers (near `detection_count`):
```python
    def mode_text(self) -> str:
        return self._mode

    def action_count(self) -> int:
        return len(self._actions)
```
In `_drain`, extend the `if/elif` chain with:
```python
            elif isinstance(ev, ModeChanged):
                self._mode = ev.mode
                self._mode_label.setText(f"mode: {ev.mode}")
            elif isinstance(ev, ActionStarted):
                self._actions.append(f"▶ {ev.behavior_name}")
                self._action_list.addItem(self._actions[-1])
            elif isinstance(ev, ActionCompleted):
                self._actions.append(f"✓ {ev.behavior_name}")
                self._action_list.addItem(self._actions[-1])
            elif isinstance(ev, ActionAborted):
                self._actions.append(f"✗ {ev.behavior_name} ({ev.reason})")
                self._action_list.addItem(self._actions[-1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_debug_window_actions.py tests/unit/test_debug_window.py -q`
Expected: PASS (both Slice-A and new tests).

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/ui/debug_window.py tests/unit/test_debug_window_actions.py
.venv/bin/mypy
git add src/smartuibot/ui/debug_window.py tests/unit/test_debug_window_actions.py
git commit -m "feat(ui): debug window mode + action timeline"
```

---

### Task 20: `app.py` wiring (input backend factory, e-stop → disarm)

**Files:**
- Modify: `src/smartuibot/app.py`
- Test: `tests/unit/test_app_factory_slice_b.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_app_factory_slice_b.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

from smartuibot.app import _make_input_backend, build_real_container  # noqa: E402
from smartuibot.input.backend import InputBackend  # noqa: E402


def test_make_input_backend_returns_protocol_impl() -> None:
    from smartuibot.core.config import load_config

    cfg = load_config(Path("configs/default.yaml"))
    be = _make_input_backend(cfg)
    assert isinstance(be, InputBackend)


def test_build_real_container_has_mode_and_action(tmp_path: Path, monkeypatch) -> None:
    import smartuibot.app as app_mod

    class _Stub:
        def infer(self, image: object) -> list: return []  # noqa: E704
        def reload(self, p: object) -> None: ...

    monkeypatch.setattr(app_mod, "_make_detector", lambda cfg: _Stub())
    monkeypatch.setattr(app_mod, "_make_capture_backend", lambda cfg: __import__(
        "tests.fakes.capture", fromlist=["FakeCaptureBackend"]).FakeCaptureBackend())
    c = build_real_container(Path("configs/default.yaml"),
                             state_path=tmp_path / "state.yaml")
    assert hasattr(c, "mode") and hasattr(c, "action") and hasattr(c, "decision")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_app_factory_slice_b.py -q`
Expected: FAIL — `ImportError: cannot import name '_make_input_backend'`

- [ ] **Step 3: Modify `src/smartuibot/app.py`**

Add import near the others:
```python
from smartuibot.input.backend import InputBackend
from smartuibot.platform_support.detect import resolve_input_backend_name
```
Add factory next to `_make_detector`:
```python
def _make_input_backend(config: AppConfig) -> InputBackend:
    name = resolve_input_backend_name(config.input.backend)
    if name == "pydirectinput":
        from smartuibot.input.pydirectinput_backend import PyDirectInputBackend

        return PyDirectInputBackend()
    from smartuibot.input.pynput_backend import PynputBackend

    return PynputBackend()
```
In `build_real_container`, pass the input backend:
```python
    return AppContainer(
        config=config,
        roi=roi,
        capture_backend=_make_capture_backend(config),
        detector=_make_detector(config),
        input_backend=_make_input_backend(config),
    )
```
In `main()`, change the `shutdown` closure to disarm before stopping:
```python
    def shutdown(*_a: object) -> None:
        container.mode.disarm()
        container.stop()
        app.quit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_app_factory_slice_b.py tests/unit/test_app_factory.py -q`
Expected: PASS (Slice-A app-factory test still green; new tests pass).

- [ ] **Step 5: Gate + commit**

```bash
.venv/bin/ruff check src/smartuibot/app.py tests/unit/test_app_factory_slice_b.py
.venv/bin/mypy
git add src/smartuibot/app.py tests/unit/test_app_factory_slice_b.py
git commit -m "feat(app): input backend factory + e-stop disarms"
```

---

### Task 21: Full gate + docs update

**Files:**
- Modify: `README.md`, `SETUP.md`

- [ ] **Step 1: Run the entire quality gate**

Run:
```bash
.venv/bin/ruff check src tests
.venv/bin/mypy
.venv/bin/pytest -q -m "not model"
```
Expected: ruff clean; mypy `Success: no issues found`; all non-model tests PASS headless (Slice A + Slice B). If anything fails, fix with superpowers:systematic-debugging — do not weaken assertions or relax the gate.

- [ ] **Step 2: Update `README.md`** — replace the `## Quick start` section body and add a Slice B paragraph. Replace the existing paragraph that begins "On first run (no `configs/state.yaml`)" with:

```markdown
On first run (no `configs/state.yaml`) the ROI selector overlay appears —
drag a rectangle to choose the capture region; it persists across restarts.
The control bar offers Start/Stop, Pause/Resume, **Arm/Disarm**, a confidence
slider, model hot-reload, and re-select ROI — all at runtime.

**Closed loop (Slice B):** when **Armed**, the bot decides via a utility
policy over `configs/behaviors.yaml` (game-agnostic; edit labels to match
your trained model) and performs humanized mouse/keyboard input. It starts
**Disarmed**; nothing is injected until you arm it. Emergency-stop and the
screen-corner fail-safe immediately disarm and abort. **Single-player /
offline use only.**
```

- [ ] **Step 3: Update `SETUP.md`** — append:

```markdown
## Input automation (Slice B)
The bot can move the mouse and press keys when **Armed** (it starts
Disarmed). macOS requires **Accessibility** permission (System Settings →
Privacy & Security → Accessibility) for the input backend in addition to
Screen Recording. Windows uses `pydirectinput` (installed automatically on
Windows only). Behaviors are defined in `configs/behaviors.yaml`; tune
motion/safety in the `input:` block of `configs/default.yaml`
(`max_actions_per_second`, `roi_confine`, `start_armed`).

**Safety:** keep `roi_confine: true` so clicks stay inside the selected
region; the emergency-stop hotkey and moving the cursor to a screen corner
both immediately disarm and abort. Use only on single-player / offline games
you are authorized to automate.
```

- [ ] **Step 4: Re-run the gate**

Run: `.venv/bin/pytest -q -m "not model"`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md SETUP.md
git commit -m "docs: Slice B (closed loop, arming, input safety)"
```

---

## Acceptance Verification (run after Task 21)

Map to spec §8. Use superpowers:verification-before-completion before claiming done.

1. Utility argmax / cooldown / anti-loop / seeded determinism — `tests/unit/test_utility.py`, `test_decision_service.py`.
2. Humanized motion non-instant, jittered, seed-deterministic — `tests/unit/test_motion.py`.
3. Starts DISARMED; arming required; emergency-stop & fail-safe abort mid-action — `test_mode.py`, `test_action_service.py` (disarm-mid-action), `test_closed_loop.py` (disarm halts).
4. Full decide→act loop headless with fakes — `tests/integration/test_closed_loop.py`. Real backend run-tested manually on macOS (env-gated `test_pynput_backend.py`).
5. ROI-confined clicks (`test_action_service.py`), clean shutdown, watchdog covers decision/action (registered in container).
6. `ruff` + `mypy --strict` + `pytest -m "not model"` green; Slice-A read-only tests unchanged and still pass.
7. Debug UI mode + action timeline + ARM/DISARM — `test_debug_window_actions.py`, `test_controls_arm.py`. Visual run is *(manual)*.

> *(manual)* items need a real screen + weights + input permission; validated by the implementer on macOS, intentionally not in headless CI.

---

## Self-Review (completed by plan author)

- **Spec coverage:** §2 decision → Tasks 3–8; §3 action/input → Tasks 9–14; §4 safety → Tasks 6 (ModeFSM), 14 (abort/confine/max-aps), 20 (e-stop disarm), config `start_armed`; §5 events/config/wiring → Tasks 2,15,16,18,19,20; §6 threading → Tasks 8,14,16; §7 testing → every task + Task 17; §8 acceptance → Acceptance section. Windows `pydirectinput` deferred-run = Task 12 (skip marker). No spec requirement left without a task.
- **Backward-compat hazards fixed inline:** `AppConfig` new fields are optional with defaults (Task 15 resolution) so Slice-A `test_config.py` stays green; `AppContainer.input_backend` defaults to `NoOpInputBackend` (Task 16 resolution) so Slice-A `test_container.py` stays green; Slice-A `test_controls.py`/`test_debug_window.py` exercised alongside new tests in Tasks 18/19.
- **Type consistency:** `ActionStep` (core.types) used by events/behavior/service/action; `ActionRequested(behavior_name, steps, roi, priority)` consistent across Tasks 2/8/14/17; `UtilityPolicy.choose → (Behavior, Detection|None, float) | None` consistent Tasks 7/8; `ModeFSM.is_armed/arm/disarm/pause/resume` consistent Tasks 6/8/14/16/18; `InputBackend` 7-method Protocol consistent Tasks 9/12/13/14; `MotionParams` field order consistent Tasks 10/14/16.
- **No placeholders:** every step has concrete code/commands.

---

## Notes for the Executor

- TDD mandatory; commit every task; never weaken a test to make it pass (debug with superpowers:systematic-debugging).
- Out of scope (do not add): behavior-tree engine, full task planner, RAG/persistent memory (S6), training/ONNX/TensorRT (S7), OCR/minimap/replay/multi-profile (S8). The Windows `PyDirectInputBackend` is run-tested later on Windows only.
- Preserve Slice-A read-only guarantee: only `ActionService` + real input backends inject; everything else stays side-effect-free.
- Keep `roi_confine` semantics and DISARMED-by-default intact — they are the core safety guarantees.
