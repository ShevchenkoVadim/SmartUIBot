# SmartUIBot

Real-time computer-vision bot framework: pick a screen region → capture it →
run YOLO11 detection → a utility-driven decision engine chooses a behavior →
humanized mouse/keyboard input executes it — all visible in a live debug
window and gated behind an **Arm/Disarm** safety switch.

Cross-platform (macOS / Windows / Linux), fully dependency-injected so the
entire pipeline runs headless with fakes. **Single-player / offline use only.**

> Версия на русском: [README.ru.md](README.ru.md)

## Quick start
    python -m pip install -e ".[dev]"
    python run.py

On first run (no `configs/state.yaml`) the ROI selector overlay appears —
drag a rectangle to choose the capture region; it persists across restarts.
The control bar offers Start/Stop, Pause/Resume, **Arm/Disarm**, a confidence
slider, model hot-reload, and re-select ROI — all at runtime.

The bot starts **Disarmed**: nothing is injected until you arm it. The
emergency-stop hotkey and the screen-corner fail-safe immediately disarm and
abort. Platform permissions and tuning knobs are documented in
[SETUP.md](SETUP.md).

## Testing
    pytest -q -m "not model"     # fast, headless, no GPU/screen
    pytest -q -m model           # downloads yolo11n.pt, runs real inference

---

# How it works

## The pipeline at a glance

Four worker services, each on its own thread, communicate **only** through a
synchronous in-process event bus. Data flows one direction; backpressure is
"drop old, keep newest" so the bot always acts on the freshest frame.

```
 screen ──grab──> CaptureService ──FrameCaptured──────┐
                                                       ▼
                                          DetectionService  (YOLO11 inference)
                                                       │
                                                       ├─DetectionsReady──> DebugWindow (live preview)
                                                       ▼
                                          DecisionService   (utility policy, ARMED only)
                                                       │
                                                       ├─ActionRequested──┐
                                                       ▼                   ▼
                                          ActionService ──move/click/key──> OS input
                                                       │
                                                       └─ActionStarted / Completed / Aborted ─> DebugWindow
```

Stages between services use a **size-1 latest-wins queue**
([`core/latest_queue.py`](src/smartuibot/core/latest_queue.py) —
`LatestQueue` at `src/smartuibot/core/latest_queue.py:7`): `put()` overwrites
any pending item, so a slow consumer (CPU inference) never lags behind a fast
producer (60 FPS capture); it just skips stale frames.

## Composition root — the container

Everything is wired in one place:
[`core/container.py`](src/smartuibot/core/container.py). `AppContainer`
(`src/smartuibot/core/container.py:26`) takes the config plus the three
platform adapters (capture backend, detector, input backend) as constructor
arguments and builds every singleton: the bus, the four services, the mode
FSM, the utility policy, and the watchdog. `start()`
(`src/smartuibot/core/container.py:80`) launches workers in reverse pipeline
order (action → decision → detection → capture) so every consumer is
subscribed before its producer emits; `stop()`
(`src/smartuibot/core/container.py:87`) disarms first, then tears down.

The real adapters are chosen in
[`app.py`](src/smartuibot/app.py): `build_real_container`
(`src/smartuibot/app.py:67`) calls the `_make_*` factories, and `main`
(`src/smartuibot/app.py:79`) builds the Qt shell, wires the control bar +
debug window, installs the emergency-stop hotkey, and runs the event loop.
Because the platform pieces are injected, tests substitute fakes from
[`tests/fakes/`](tests/fakes) and exercise the whole loop without a screen,
GPU, or real mouse.

## The service framework

All workers extend `Service`
([`core/service.py`](src/smartuibot/core/service.py),
`src/smartuibot/core/service.py:12`). The base class owns a daemon thread,
runs `run_once()` in a loop (`src/smartuibot/core/service.py:52`), updates a
`last_heartbeat` timestamp each iteration, and supports cooperative
`pause()`/`resume()`. Any unhandled exception is converted into a **fatal
`ServiceError`** event and the thread exits cleanly rather than crashing the
process.

The `Watchdog` ([`core/watchdog.py`](src/smartuibot/core/watchdog.py),
`src/smartuibot/core/watchdog.py:15`) supervises all four services: it polls
`is_alive` every second and restarts a dead worker with exponential backoff,
escalating to a fatal `ServiceError` after `max_retries`
(`src/smartuibot/core/watchdog.py:45`).

## The event bus

[`core/event_bus.py`](src/smartuibot/core/event_bus.py) is a thread-safe
**synchronous** pub/sub (`src/smartuibot/core/event_bus.py:15`). `publish()`
calls subscribers inline; a subscriber exception is logged and swallowed
(`src/smartuibot/core/event_bus.py:33`) so one bad handler can never break the
publisher or other subscribers. All message types are frozen dataclasses in
[`core/events.py`](src/smartuibot/core/events.py): `FrameCaptured`,
`DetectionsReady`, `ActionRequested`, `ActionStarted/Completed/Aborted`,
`ModeChanged`, `FpsTick`, `LogRecord`, `ServiceError`, `StateChanged`.

Shared data shapes live in
[`core/types.py`](src/smartuibot/core/types.py): `ROI`
(`src/smartuibot/core/types.py:10`, validated, YAML round-trips via
`as_dict`/`from_dict`), `Frame` (BGR `np.ndarray` + seq + timestamp,
`:44`), `Detection` (label, confidence, bbox, optional track id, `:52`), and
`ActionStep` (the executable primitive: `move|click|key|wait`, `:75`).

## Stage 1 — Capture

[`vision/capture/service.py`](src/smartuibot/vision/capture/service.py)
(`CaptureService` at `src/smartuibot/vision/capture/service.py:15`) grabs the
current ROI through a `CaptureBackend`, wraps it in a `Frame` with a
monotonically increasing `seq`, publishes `FrameCaptured`, and sleeps to hold
`target_fps`. The ROI is swappable at runtime under a lock (`set_roi`,
`:31`) so "Select ROI" works while running.

The backend is a `Protocol`
([`vision/capture/backend.py`](src/smartuibot/vision/capture/backend.py),
`src/smartuibot/vision/capture/backend.py:21`). The shipped implementation is
[`mss_backend.py`](src/smartuibot/vision/capture/mss_backend.py); the faster
Windows `dxcam` path currently falls back to `mss` (see `_make_capture_backend`
in `src/smartuibot/app.py:35`).

## Stage 2 — Detection

[`vision/detect/service.py`](src/smartuibot/vision/detect/service.py)
(`DetectionService` at `src/smartuibot/vision/detect/service.py:16`)
subscribes to `FrameCaptured`, pushes frames into its `LatestQueue`, and on
its own thread pulls the newest frame, runs inference, filters by a
**runtime-adjustable** confidence threshold (`set_confidence`, `:38` — driven
by the UI slider), smooths the result, and publishes `DetectionsReady`.

- `Detector` is a `Protocol`
  ([`vision/detect/detector.py`](src/smartuibot/vision/detect/detector.py),
  `src/smartuibot/vision/detect/detector.py:11`); the real one is
  [`yolo.py`](src/smartuibot/vision/detect/yolo.py) (`Yolo11Detector`,
  Ultralytics YOLO11), hot-reloadable via `reload()`.
- `SmoothingFilter`
  ([`vision/detect/smoothing.py`](src/smartuibot/vision/detect/smoothing.py),
  `src/smartuibot/vision/detect/smoothing.py:7`) keeps a label visible for
  `smoothing_frames` extra frames after it disappears, reducing flicker.

## Stage 3 — Decision

[`ai/service.py`](src/smartuibot/ai/service.py) (`DecisionService` at
`src/smartuibot/ai/service.py:17`) ticks at `decision.tick_hz`. **It
no-ops unless the mode is ARMED** (`src/smartuibot/ai/service.py:41`). Each
tick it snapshots world state, asks the policy to choose, and on a hit
publishes `ActionRequested` with concrete `ActionStep`s.

- **World state**
  ([`ai/world_state.py`](src/smartuibot/ai/world_state.py)):
  `WorldStateTracker` (`src/smartuibot/ai/world_state.py:43`) builds an
  immutable `WorldState` snapshot per tick and records executed behaviors in a
  ring buffer. `best_match` (`:17`) finds the highest-confidence detection
  matching a behavior's labels; `ticks_since`/`recent_count` back the cooldown
  and anti-loop logic.
- **Behaviors**
  ([`ai/behavior.py`](src/smartuibot/ai/behavior.py)): a `Behavior` is a
  `Condition` (label set + thresholds) plus `base_utility`, `cooldown_s`, and
  declarative `BehaviorStep`s. `resolve_steps`
  (`src/smartuibot/ai/behavior.py:41`) turns abstract steps into concrete
  `ActionStep`s, resolving `target: detection | roi_center | fixed` into pixel
  coordinates. Behaviors are loaded from
  [`configs/behaviors.yaml`](configs/behaviors.yaml) by
  [`ai/registry.py`](src/smartuibot/ai/registry.py) (`load_behaviors` at
  `src/smartuibot/ai/registry.py:29`), which validates kinds and values.
- **Utility policy**
  ([`ai/utility.py`](src/smartuibot/ai/utility.py), `UtilityPolicy` at
  `src/smartuibot/ai/utility.py:13`): scores every condition-satisfying
  behavior and picks the argmax. Scoring (`choose`, `:34`) applies
  confidence scaling, a cooldown exclusion, an anti-loop penalty when a
  behavior repeats too often in a window, a small **seeded** random jitter,
  and an occasional "hesitation" skip — so play looks human and is
  reproducible from `decision.rng_seed`.

## Stage 4 — Action (input automation)

[`input/service.py`](src/smartuibot/input/service.py) (`ActionService` at
`src/smartuibot/input/service.py:28`) consumes `ActionRequested` (newest
only), and only if still ARMED, executes steps one at a time, rate-limited by
`max_actions_per_second`. It re-checks `_aborted()`
(`src/smartuibot/input/service.py:62`) **before every step and every motion
sub-point**, so disarming or emergency-stop halts mid-action and emits
`ActionAborted`; otherwise `ActionStarted` → `ActionCompleted`.

- **Humanized motion**
  ([`input/motion.py`](src/smartuibot/input/motion.py)): `bezier_path`
  (`src/smartuibot/input/motion.py:19`) draws a quadratic Bézier with a
  randomized control point and per-point jitter; plus randomized reaction
  delays and optional overshoot. All randomness is from an injected seeded
  `random.Random`.
- **ROI confinement**: when `roi_confine` is on, `_screen`/`_confine`
  (`src/smartuibot/input/service.py:57`) clamp every target inside the
  selected region before mapping to absolute screen coordinates — clicks can't
  escape the box.
- **Backends**: `InputBackend` is a `Protocol`
  ([`input/backend.py`](src/smartuibot/input/backend.py),
  `src/smartuibot/input/backend.py:8`). The default is the safe
  `NoOpInputBackend` (`:18`, injects nothing);
  [`pynput_backend.py`](src/smartuibot/input/pynput_backend.py) (mac/Linux)
  and [`pydirectinput_backend.py`](src/smartuibot/input/pydirectinput_backend.py)
  (Windows) are selected by `resolve_input_backend_name`.

## Safety: the mode FSM

[`ai/mode.py`](src/smartuibot/ai/mode.py) (`ModeFSM` at
`src/smartuibot/ai/mode.py:13`) is a thread-safe gate with three states:
`DISARMED` (default), `ARMED`, `PAUSED`. **Injection is possible only in
ARMED**, and both the decision and action services check it independently.
The Arm/Disarm button (`UiController.toggle_arm`,
`src/smartuibot/ui/controls.py:67`) and the emergency-stop hotkey / corner
fail-safe (`src/smartuibot/app.py:110`) flip it to disarm and abort. It starts
DISARMED unless `input.start_armed: true`
(`src/smartuibot/core/container.py:53`).

## The UI layer (Qt, main thread)

Qt rule: all GUI work on the main thread. Worker threads only publish events.

- [`ui/controls.py`](src/smartuibot/ui/controls.py): `UiController`
  (`src/smartuibot/ui/controls.py:16`) is **pure glue with no Qt imports** —
  every action (start/stop, pause, confidence, reload, ROI, arm) is
  unit-testable against a fake container. `ControlBar` (`:75`) is the thin Qt
  widget on top.
- [`ui/debug_window.py`](src/smartuibot/ui/debug_window.py): `DebugWindow`
  (`src/smartuibot/ui/debug_window.py:48`) subscribes to events from worker
  threads, buffers them in a `queue.Queue`, and drains on a ~30 Hz `QTimer`
  (`_drain`, `:113`) — the thread-safe handoff. It draws detection boxes,
  FPS, logs, mode, and the action log.
- [`ui/roi_selector.py`](src/smartuibot/ui/roi_selector.py): a fullscreen
  translucent overlay sized to the configured monitor's `QScreen`
  (`_resolve_screen`). Drag-release maps logical points into the capture
  backend's pixel space in the pure `selection_to_roi`, scaling by the
  **measured ratio** of the mss monitor size to the Qt screen's logical size
  (not `devicePixelRatio`, which double-counts on macOS+mss), clamped to the
  monitor — correct on Retina/HiDPI. Result persists to `configs/state.yaml`
  (`save_roi`). Esc or a sub-minimum drag cancels with no state change.

## Configuration & logging

- [`core/config.py`](src/smartuibot/core/config.py): `load_config`
  (`src/smartuibot/core/config.py:104`) loads
  [`configs/default.yaml`](configs/default.yaml), deep-merges an optional user
  override, and builds a validated frozen `AppConfig`
  (`src/smartuibot/core/config.py:68`, fail-fast in `__post_init__`).
- [`core/logging_setup.py`](src/smartuibot/core/logging_setup.py):
  `setup_logging` (`src/smartuibot/core/logging_setup.py:50`) installs three
  handlers — colorized console, rotating **JSON** file in `logs/`, and a
  `_BusHandler` that republishes log records as `LogRecord` events so they
  show up live in the debug window.
- [`platform_support/detect.py`](src/smartuibot/platform_support/detect.py):
  `current_os` / `resolve_backend_name` / `resolve_input_backend_name`
  (`src/smartuibot/platform_support/detect.py:15`) turn `auto` into the right
  per-OS capture and input backend.

## Testing strategy

[`tests/`](tests) mirrors the package. Fakes in
[`tests/fakes/`](tests/fakes) (capture, detector, input) let
[`tests/integration/test_closed_loop.py`](tests/integration/test_closed_loop.py)
and [`test_pipeline.py`](tests/integration/test_pipeline.py) run the full
capture→decision→action loop headlessly and deterministically (seeded RNG).
Unit tests cover every module; the `model` marker isolates the one suite that
needs real YOLO weights so the default run stays offline and fast.

## Design docs

Deeper rationale and slice plans live in
[`docs/superpowers/specs/`](docs/superpowers/specs) and
[`docs/superpowers/plans/`](docs/superpowers/plans) — Slice A (CV pipeline),
Slice B (decision engine + input), and the capture-region selector.
