# SmartUIBot — Slice B Design Spec (Decision Engine + Input Automation)

- **Date:** 2026-05-16
- **Status:** Approved (design); pending written-spec review
- **Author:** brainstorming session (Claude Code + user)
- **Topic:** Second vertical slice — close the loop: perceive → decide → act
- **Builds on:** Slice A (`docs/superpowers/specs/2026-05-16-smartuibot-slice-a-design.md`), merged to `master`

---

## 1. Context & Problem

Slice A delivered a read-only pipeline: ROI capture → YOLO11 detection → debug UI, on a
clean hexagonal foundation (DI container, thread-safe `EventBus`, `Service` base with
thread + heartbeat + pause/resume, `Watchdog`, structured logging, emergency-stop hotkey
listener). The `ai/`, `input/`, `memory/` packages are empty placeholders.

**Slice B closes the loop**: consume `DetectionsReady`, decide what to do, and perform
real mouse/keyboard input so the bot actually plays. It bundles the originally-separate
S4 (decision engine) and S5 (human-like input) into one coherent vertical because a
decider with no actuator is only half a loop.

### Decisions established during brainstorming

| Question | Decision |
|---|---|
| Slice B boundary | **Full closed loop**: decision engine **+ real input automation**. |
| Decision paradigm | **Utility-AI core + thin mode FSM + config-driven behavior registry**; cooldowns, anti-loop, human randomness. Behavior-trees and a full task planner are **deferred behind clean seams** (YAGNI). |
| Decision/action structure | **Decoupled services over the event bus** (Approach A): `DecisionService` emits intents, `ActionService` actuates on its own thread. Keeps decision logic headless-testable; only a thin backend touches the OS. |
| Target / ethics | **Single-player / offline games only**, generic & config-driven, user-selected ROI. No competitive/online/anti-cheat automation. Carried forward from Slice A. |
| Input platform | `InputBackend` Protocol; `PynputBackend` (macOS/Linux + universal), `PyDirectInputBackend` (Windows). Selection mirrors Slice-A capture-backend resolution. |

### Out of scope (later slices; seams placed, not implemented)

RAG / persistent memory (S6 — only a small **in-process bounded event ring buffer** for
anti-loop/cooldowns is included), training pipeline + ONNX/TensorRT (S7), OCR / minimap /
template-match / replay / timeline / heatmaps / multi-agent profiles (S8), behavior-tree
engine and full task planner. The Windows `PyDirectInputBackend` adapter is written and
code-reviewed but run-tested only on Windows later (mirrors Slice-A `dxcam` posture; macOS
dev box uses `PynputBackend`).

---

## 2. Decision Architecture

- **`WorldState`** — a pure, immutable per-tick snapshot built from the latest
  `DetectionsReady` (its `frame` + `detections`) plus a bounded ring buffer of recent
  decision events (for anti-loop / cooldown reasoning). No Qt/OS imports.
- **`Behavior`** — a config-defined unit:
  - `name: str`
  - `condition`: predicate over `WorldState` (e.g. *"≥1 detection whose label ∈
    {enemy} and confidence ≥ t"*). Conditions are expressed declaratively in config
    (label set, min confidence, min/max count, optional region) and compiled to a
    predicate — **no arbitrary code from config**.
  - `base_utility: float` and situational `factors` (e.g. scale by detection
    confidence, proximity to ROI center).
  - `cooldown_s: float`.
  - `steps`: ordered action steps — `move` (to a target), `click`, `key`,
    `wait` — where a target resolves from a chosen detection's box (default:
    centroid) or a fixed ROI-relative point.
- **`BehaviorRegistry`** — loads/validates behaviors from YAML
  (`configs/behaviors.yaml`), game-agnostic. The spec's example behaviors
  (enemy→attack, low-HP→heal, popup→close, reward→collect, patrol, idle) are
  expressed purely as config entries.
- **`UtilityPolicy`** — per tick: take behaviors whose `condition` holds and that are
  not in cooldown; score each (`base_utility` × factors); pick the argmax. Apply
  **anti-loop** (penalty when a behavior fired > N times within the recent window) and
  **human randomness** (bounded score jitter + configurable "hesitation" probability
  that yields a no-op tick). On a winner, emit
  `ActionRequested(behavior_name, steps, targets)`.
- **`ModeFSM`** — coarse states: `DISARMED → ARMED ⇄ PAUSED → STOPPING`. Action intents
  are produced only in `ARMED`. Emergency-stop / fail-safe forces `DISARMED`.

Determinism: `UtilityPolicy` randomness uses an injected, seedable RNG so behavior is
reproducible in tests.

---

## 3. Action Architecture (real input)

- **`InputBackend` Protocol**: `move_to(x, y)`, `mouse_down(button)`,
  `mouse_up(button)`, `click(button)`, `key_down(key)`, `key_up(key)`,
  `type_text(text)`. Adapters:
  - `PynputBackend` — macOS/Linux + universal (dev/test platform).
  - `PyDirectInputBackend` — Windows, game-compatible scancodes (deferred run-test).
  - Selected via `platform_support` + a `resolve_input_backend_name` mirroring
    Slice A's `resolve_backend_name`.
- **`HumanizedMotion`** — pure, deterministic-with-seed functions:
  - cubic-bezier path point list between two coordinates,
  - per-segment variable speed + micro-jitter,
  - randomized reaction-time delay before an action,
  - variable inter-keystroke delay,
  - occasional overshoot-then-correct.
  These return *plans* (lists of `(x, y, sleep)` / timing tuples); they do not touch
  the OS, so they are fully unit-tested.
- **`ActionService`** — a `Service` on its **own thread**. Consumes `ActionRequested`
  from a small bounded queue; expands the behavior `steps`; for each step computes a
  `HumanizedMotion` plan and drives the `InputBackend`. Executes **one action at a
  time**, **interruptible at step boundaries** when the bus signals a higher-utility
  preemption (with hysteresis to prevent thrash). ROI-relative targets are converted
  to screen-absolute via the active ROI + monitor.
- **`RecordingInputBackend`** — test fake recording the exact call/timing sequence
  instead of moving the OS cursor, so the entire decide→act loop runs headless in CI.

---

## 4. Safety (critical — real input automation)

- **Arm/disarm gate**: process starts `DISARMED`; a deliberate UI toggle or hotkey is
  required to reach `ARMED`. No injection occurs until `ARMED`.
- **Emergency-stop**: the existing Slice-A global hotkey listener additionally forces
  `ModeFSM → DISARMED`, aborts any in-flight action immediately, then stops services.
- **Fail-safe**: pyautogui-style screen-corner abort (cursor forced to a corner
  disarms), a configurable max-actions-per-second limiter, and auto-disarm on any
  `ServiceError` in the action path.
- **ROI confinement**: injected clicks are clamped to the selected ROI by default
  (configurable) so a misdetection cannot click arbitrary screen locations.
- Slice-A read-only components are unchanged and remain read-only; only `ActionService`
  + input backends inject. README/SETUP reaffirm **single-player / offline only** and
  the new macOS Accessibility/Input-injection permission requirement.

---

## 5. Events, Config, Wiring

- **New events**: `ActionRequested(behavior_name, steps, targets)`, `ActionStarted`,
  `ActionCompleted`, `ActionAborted(reason)`, `ModeChanged(mode)`. The Slice-A
  `DebugWindow` subscribes and shows: current mode, chosen behavior, an action
  timeline, and an **ARM/DISARM** + behavior-profile control in the existing
  `ControlBar`.
- **Config** (`configs/default.yaml` additions, all typed + validated like Slice A):
  - `decision:` — `tick_hz`, `anti_loop_window`, `anti_loop_max_repeats`,
    `hesitation_prob`, `rng_seed`.
  - `input:` — `backend` (auto|pynput|pydirectinput), motion profile params
    (speed range, jitter px, reaction-time range, keystroke delay range,
    overshoot prob), `max_actions_per_second`, `roi_confine` (bool),
    `start_armed` (default false).
  - `behaviors:` — inline or a referenced `configs/behaviors.yaml`.
- **`AppContainer`** gains `decision` + `action` services and the selected
  `InputBackend`, all registered with the `Watchdog` and injected (tests pass fakes),
  consistent with Slice A's composition root.

---

## 6. Threading Model

Adds two threads to Slice A's model:
- `DecisionService` — subscribes to `DetectionsReady`, lightweight, fast tick; emits
  `ActionRequested`. No blocking work.
- `ActionService` — own thread; all human-like sleeps/bezier timing live here so
  perception and decision never block.

The thread-safe `EventBus` remains the cross-thread seam. The action channel is a
small bounded queue with explicit preemption signaling (drop/replace stale intents;
never buffer a long backlog of stale actions).

---

## 7. Testing Strategy

- **Pure unit**: `WorldState` derivation; `UtilityPolicy` scoring, cooldown,
  anti-loop, seeded jitter/hesitation; `BehaviorRegistry` YAML load + validation +
  declarative-condition compilation; `HumanizedMotion` math (deterministic with seed);
  `resolve_input_backend_name`; ROI-relative→absolute coordinate mapping.
- **Fakes**: `FakeDetector` (Slice A) scripts drive scenarios; `RecordingInputBackend`
  asserts the exact humanized call/timing sequence.
- **Integration (headless)**: full `DetectionsReady → DecisionService → ActionService
  → RecordingInputBackend`. Assert: scripted "enemy" → recorded click near its
  centroid; "low_hp" preempts "patrol"; anti-loop caps repeats; emergency-stop /
  disarm halts injection mid-action; ROI confinement clamps an out-of-ROI target.
- **Real OS injection** = the thin untestable edge, env-gated like Slice A's real
  screen-grab test. `ruff` + `mypy --strict` + `pytest -m "not model"` stay green
  headless with no screen/OS input.
- TDD per task during implementation; PEP 695 generics; mypy-overrides extended for
  any new untyped dep (`pynput` already covered; add `pydirectinput`).

---

## 8. Acceptance Criteria

1. Scripted detections drive correct behavior selection — utility argmax, cooldowns,
   anti-loop, seeded determinism — verified by headless tests.
2. `HumanizedMotion` output is non-instant, jittered, and deterministic given a seed —
   unit-tested.
3. Process starts `DISARMED`; arming is required before any injection; emergency-stop
   and fail-safe truly halt injection mid-action — verified with `RecordingInputBackend`.
4. Full decide→act loop runs headless in CI with fakes; the real backend is run-tested
   manually on macOS (documented, env-gated).
5. Injected clicks are ROI-confined; clean shutdown; `Watchdog` restarts a crashed
   `DecisionService`/`ActionService` worker.
6. `ruff` + `mypy --strict` + `pytest -m "not model"` all green; **zero behavior
   change to Slice-A read-only components** (only new services/backends inject).
7. Debug UI shows mode, chosen behavior, and an action timeline; ARM/DISARM control
   works.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Real input automation misbehaves / clicks wrong place | DISARMED-by-default, arm gate, ROI confinement, emergency-stop + corner fail-safe, max-APS limiter, auto-disarm on error |
| Bundling S4+S5 over-scopes the slice | Decision paradigm trimmed to utility+thin-FSM (BT/planner deferred); decoupled services keep each unit small and independently testable |
| Human-like timing stalls perception | `ActionService` on its own thread; bus decoupling; bounded action queue with preemption |
| Non-determinism makes tests flaky | All randomness via injected seedable RNG; timing asserted via plans/`RecordingInputBackend`, not wall-clock |
| macOS input-injection permission missing | Preflight check + explicit `SETUP.md` instructions; auto-disarm + clear log if backend init fails |
| ToS/ethical misuse | Spec + README hard-state single-player/offline only; no anti-cheat/competitive support |

---

## 10. Next Steps

1. User reviews this written spec.
2. On approval → `superpowers:writing-plans` to produce the Slice B implementation plan.
3. Implementation via subagent-driven TDD; later slices (S6–S8) repeat spec → plan →
   build, plugging into this now-closed loop.
