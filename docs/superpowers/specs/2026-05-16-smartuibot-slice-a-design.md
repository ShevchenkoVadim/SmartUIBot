# SmartUIBot — Slice A Design Spec

- **Date:** 2026-05-16
- **Status:** Approved (design); pending written-spec review
- **Author:** brainstorming session (Claude Code + user)
- **Topic:** First vertical slice of the SmartUIBot cross-platform CV automation framework

---

## 1. Context & Problem

The long-term goal is a production-grade, modular, cross-platform desktop AI bot framework
for 2D applications/games: ROI screen capture → real-time scene analysis → YOLO11
detection → context-aware decisions → human-like input → memory/RAG, with a debug UI,
training pipeline, and stability tooling.

### Decisions established during brainstorming

| Question | Decision |
|---|---|
| Runtime platform | **Cross-platform (Windows + macOS)** with a first-class platform-abstraction layer |
| Target application | **Generic / game-agnostic**: everything is driven by a **user-selected screen ROI**; no hardcoded game logic. Users adapt it to a specific game by training a custom YOLO11 model (later slice). |
| Game nature | **Single-player / offline** games only. Low ToS risk; no anti-cheat / competitive multiplayer automation. |
| Dev environment | macOS 25.4 (Darwin), **Intel x86_64** (no CUDA, no Apple-Silicon MPS → CPU-only inference locally). Python 3.12 venv (PyCharm). Greenfield, not yet a git repo. |
| Build strategy | The 18-section vision is decomposed into ~8 sub-projects (S0–S8). Each gets its own spec → plan → build cycle. **This spec covers Slice A only.** |

### Platform reality table

| Spec component | Windows + NVIDIA | Intel macOS (dev box) |
|---|---|---|
| DXcam capture | ✅ supported | ❌ Windows-only API → use `mss` |
| pydirectinput | ✅ (later slice S5) | ❌ Windows-only → use `pynput`/Quartz (S5) |
| CUDA / TensorRT | ✅ | ❌ CPU-only |
| YOLO11 FPS | high (GPU) | low (~3–10 FPS CPU); capture stays 30–60 |

Capture FPS and inference FPS are therefore measured and surfaced **separately** in the UI;
the gap is expected behavior on CPU, not a defect.

### Full-system decomposition (north star)

| # | Sub-project | Depends on |
|---|---|---|
| **S0** | Foundation: scaffold, DI/service container, event bus, YAML config, structured logging, platform-abstraction interfaces, lifecycle + watchdog + emergency-stop | — |
| **S1** | Capture engine: ROI selector overlay, persisted coords, dedicated capture thread, mss/dxcam behind one interface, FPS control, pause/resume | S0 |
| **S2** | Vision: YOLO11 inference service, model load/hot-reload, thresholds, tracking IDs, temporal smoothing, ONNX/TensorRT optional | S0, S1 |
| **S3** | Debug/Overlay UI: separate window — live preview, boxes, detections list, FPS, AI state, logs console, controls | S0–S2 |
| **S4** | Decision engine: FSM + utility AI + behavior trees + planner, cooldowns, anti-loop, human randomness | S0–S2 |
| **S5** | Human-like input: bezier paths, jitter, reaction times, typing; pynput(mac)/pydirectinput(win) behind one interface | S0 |
| **S6** | Memory/RAG: SQLite episodic store, embeddings, FAISS/Chroma retrieval, strategy reuse | S0, S4 |
| **S7 / S8** | Training pipeline; Advanced (OCR, minimap, template match, replay/timeline/heatmaps, multi-profile) | varies |

**Slice A = S0 + S1 + S2 + S3.**

---

## 2. Slice A Scope

A runnable, **read-only** real-time pipeline:

> select a screen ROI → capture it at high FPS → run YOLO11 detection →
> render boxes / FPS / detections / logs in a separate debug window,

plus the production backbone (DI, event bus, config, logging, watchdog, emergency-stop)
that all later slices plug into.

**In scope:** S0 backbone; S1 ROI capture (mss/dxcam abstraction, pause/resume, live ROI
change, multi-monitor); S2 YOLO11 detection (load, hot-reload, confidence threshold,
optional ByteTrack IDs, temporal smoothing); S3 debug window + ROI selector overlay.

**Explicitly out of scope (later slices; seams/interfaces placed now so they drop in
without rework):** input injection (S5), decision engine / FSM / behavior trees / utility
AI / planner (S4), RAG / memory (S6), training pipeline + ONNX/TensorRT implementation
(S7), OCR / minimap / template matching / motion detection / scene classification /
replay / timeline / heatmaps / multi-agent profiles (S8).

---

## 3. Architecture

Clean / hexagonal. The domain core has no knowledge of mss, dxcam, PyTorch, or Qt; those
are **adapters** behind `Protocol` interfaces, selected at runtime by platform.

```
UI LAYER (PyQt6, main thread)
  ROISelectorOverlay   DebugWindow(preview, boxes, list, logs)
        ▲ Qt signals            ▲ subscribes
APPLICATION / ORCHESTRATION
  AppContainer (DI)   EventBus (pub/sub)   Lifecycle + Watchdog
        ▲                ▲                      ▲
  CaptureService    DetectionService    ConfigService / LoggingService
   (thread)           (thread)
        ▲ Protocol       ▲ Protocol
  CaptureBackend     Detector
  mss / dxcam        YOLO11 (torch)
```

- **DI container** — explicit, lightweight, constructor injection (no magic framework);
  wires services and selects platform adapters from config.
- **Event bus** — thread-safe pub/sub; decouples producers (capture, detection) from
  consumers (UI, logs, future S4). Events: `FrameCaptured`, `DetectionsReady`, `FpsTick`,
  `ServiceError`, `LogRecord`, `StateChanged`. A subscriber exception is isolated, logged,
  and never kills the publisher.
- **Platform abstraction** — `CaptureBackend` and `Detector` are `Protocol`s.
  `InputBackend` is defined now but its only Slice-A implementation is `NoOpInput`
  (real input = S5).

---

## 4. Processing Pipeline & Data Flow

```
[ROI rect from config/state]
   → CaptureService(thread): grab(roi) → Frame(np.ndarray BGRA, ts, seq)
   → LatestFrameQueue (size=1, drop-old)    ◄── backpressure point
   → DetectionService(thread): pop newest → YOLO11 infer → Detections[]
        + SmoothingFilter (EMA / N-frame persistence)
        + Tracker IDs (ByteTrack, optional)
   → EventBus.publish(DetectionsReady, FpsTick)
   → DebugWindow (Qt slot): draw preview + boxes + table + FPS + logs
```

**Backpressure decision (core performance rationale):** capture and inference run at
independent rates. On Intel-Mac CPU, inference is far slower than capture. A **size-1
latest-wins queue** means the capturer overwrites the pending frame and the detector
always works on the freshest frame; stale frames are dropped. This *is* the
adaptive-FPS / frame-skip behavior — no growing latency, no unbounded memory. Capture FPS
and inference FPS are measured and displayed separately.

---

## 5. Threading Model (highest-risk decision — explicit)

| Unit | Mechanism | Rationale |
|---|---|---|
| Capture | dedicated `threading.Thread` | mss/dxcam release the GIL during the OS grab; thread = simple, zero IPC frame-copy |
| Inference | dedicated `threading.Thread` | PyTorch releases the GIL in native ops; a process would force pickling/copying every frame |
| UI | **main thread** | PyQt6 mandates GUI on main thread; workers reach it via Qt signals / event bus |
| Lifecycle / Watchdog | supervisor `threading.Thread` | monitors per-service heartbeats; restarts a crashed worker with backoff |
| Emergency-stop | `pynput` global **listener** (read-only) | a hotkey listener is not input injection; allowed in Slice A |

**Why threads (not multiprocessing / asyncio) in the hot path:** the hot path is dominated
by native code that releases the GIL (OS capture, Torch). Multiprocessing adds a
frame-serialization tax per frame; asyncio does not help CPU/native-bound work. We use
**threads + bounded queues + a thread-safe event bus**. asyncio/multiprocessing are
deliberately deferred to S4's orchestration layer where they pay off — an intentional
choice, recorded so it is not accidental.

---

## 6. Project Folder Structure (Slice A; later slices fill empty packages)

```
SmartUIBot/
├── pyproject.toml            # deps, ruff, mypy(strict), pytest config
├── README.md  SETUP.md
├── run.py                    # launch script
├── configs/
│   ├── default.yaml          # thresholds, fps, model, backend, hotkeys
│   └── state.yaml            # persisted ROI + monitor index (runtime-written)
├── src/smartuibot/
│   ├── core/                 # S0: container, event_bus, config, logging,
│   │                         #     lifecycle, watchdog, errors, types
│   ├── platform/             # platform detection + adapter selection
│   ├── vision/
│   │   ├── capture/          # S1: CaptureService + mss/dxcam backends + ROI model
│   │   └── detect/           # S2: DetectionService, YOLO adapter, smoothing, tracking
│   ├── ui/                   # S3: DebugWindow, ROISelectorOverlay, log panel
│   ├── ai/   input/   memory/   # later slices — empty packages w/ interfaces only
│   └── app.py                # composition root: builds container, starts services
├── models/                   # YOLO weights (yolo11n.pt auto-downloaded)
├── logs/                     # rotating structured logs
├── datasets/                 # later (S7)
└── tests/
    ├── unit/   integration/   fakes/   fixtures/
```

---

## 7. Key Modules & Interfaces

- **`core.container.AppContainer`** — builds & owns service singletons; constructor
  injection; `start()` / `stop()`.
- **`core.event_bus.EventBus`** — `subscribe(EventType, handler)`, `publish(event)`;
  thread-safe; subscriber exceptions isolated + logged.
- **`core.config`** — YAML → typed dataclasses with validation; layered
  (defaults ← user ← runtime state); `reload()` for hot-reloadable keys (thresholds, FPS).
- **`core.logging`** — structured JSON rotating file + colored console + in-memory ring
  buffer published as `LogRecord` events for the UI panel.
- **`core.lifecycle` / `Watchdog`** — per-service heartbeat; crashed worker → isolate,
  log, restart with exponential backoff (max retries → degrade + surface in UI); global
  emergency-stop → graceful shutdown.
- **`vision.capture.CaptureBackend`** (Protocol): `grab(roi) -> Frame`,
  `list_monitors()`. Adapters: `MssBackend` (macOS + universal fallback),
  `DxcamBackend` (Windows).
- **`vision.capture.CaptureService`** — owns backend + capture thread + FPS meter;
  publishes `FrameCaptured`; pause/resume; live ROI change without restart.
- **`vision.detect.Detector`** (Protocol): `infer(frame) -> list[Detection]`,
  `reload(path)`. Adapter: `Yolo11Detector` (Ultralytics; CUDA if available else CPU;
  ONNX/TensorRT export hook = interface only in Slice A).
- **`vision.detect.DetectionService`** — consumes latest frame, runs detector, applies
  `SmoothingFilter` + optional ByteTrack IDs, publishes `DetectionsReady`.
- **`ui.ROISelectorOverlay`** — frameless translucent PyQt6 window for drag-select;
  writes ROI + monitor index to `state.yaml`.
- **`ui.DebugWindow`** — live preview + boxes, detections table, capture/infer FPS,
  current state, real-time log console, controls (start/stop, pause, model picker,
  re-select ROI, confidence slider, debug toggle).

---

## 8. Tech-Stack Decisions (locked for Slice A)

- **UI = PyQt6** (not DearPyGui): mature worker→GUI signaling, reliable
  frameless/translucent overlay for ROI selection, cross-platform.
- **Capture = mss (macOS x86 + universal fallback) / dxcam (Windows)**; abstraction hides
  the difference.
- **Detector = Ultralytics YOLO11**, default weights `yolo11n.pt` (auto-downloaded to
  `models/`). Device auto: CUDA → else CPU. On the Intel-Mac dev box this is CPU
  (~3–10 FPS inference vs 30–60 capture — separate meters make this visible).
- **Config = YAML → typed dataclasses**; **Python 3.12** (matches existing venv).
- **Quality gates = ruff + mypy --strict + pytest**; TDD during implementation
  (separate plan/execution phase).

---

## 9. Error Handling & Safety

- Every worker thread wrapped in an **exception boundary**: catch → publish
  `ServiceError` → structured log → watchdog restarts with backoff (max retries → degrade
  + surface in UI).
- EventBus isolates subscriber exceptions (a bad UI handler cannot kill capture).
- **Emergency-stop hotkey** (default `Ctrl+Alt+Q`) → immediate graceful shutdown of all
  services.
- macOS **Screen Recording (TCC)** and global-hotkey/Accessibility permissions are
  required; documented in `SETUP.md` with a friendly preflight check that fails loudly
  with instructions.
- Clean shutdown: cooperative stop events; threads joined with timeout; Qt app quits last.

---

## 10. Testing Strategy

- **Unit:** config load/validate; ROI math + multi-monitor mapping; latest-wins queue;
  EventBus (incl. subscriber-exception isolation); SmoothingFilter; FPS meter; container
  wiring.
- **Fakes:** `FakeCaptureBackend` (synthetic frames), `FakeDetector` (scripted
  detections) → whole pipeline runs **headless in CI**, no screen/GPU.
- **Integration:** real services + fakes; assert frames flow capture → detect → bus and
  stale frames are dropped under a slow-detector simulation.
- **Model test (marked/skippable):** download `yolo11n`, one inference on a fixture image,
  assert `Detection` schema; skipped if offline.
- **UI smoke:** Qt app builds and tears down with `QT_QPA_PLATFORM=offscreen`.

---

## 11. Acceptance Criteria for Slice A

1. `python run.py` launches; user selects an ROI via the overlay; ROI persists across
   restarts in `state.yaml`.
2. Live preview of the ROI renders in the debug window with YOLO11 bounding boxes,
   per-object labels + confidence, and a detections table.
3. Capture FPS and inference FPS are displayed separately and update in real time.
4. ROI can be re-selected and confidence threshold changed at runtime without restart;
   model can be hot-reloaded via the UI.
5. Pause/resume works; emergency-stop hotkey cleanly shuts everything down.
6. Killing a worker (simulated exception) triggers watchdog restart with backoff, visible
   in the log panel.
7. `ruff`, `mypy --strict`, and `pytest` (unit + integration with fakes, headless) all
   pass in CI without a screen or GPU.
8. No mouse/keyboard injection occurs anywhere (read-only guarantee).

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| CPU-only inference on Intel Mac is slow | Latest-wins backpressure; separate FPS meters; choose `yolo11n`; document expectation |
| macOS TCC permission blocks capture silently | Preflight permission check with explicit user-facing instructions in `SETUP.md` |
| Qt threading mistakes (GUI off main thread) | Strict rule: workers publish events; only Qt slots touch widgets; covered by review + offscreen smoke test |
| Cross-platform drift (only macOS tested locally) | Platform-abstraction interfaces + fakes; Windows path code-reviewed and CI-linted even though run-tested later |
| Scope creep into S4/S5 | This spec hard-bounds Slice A; later subsystems are interfaces/stubs only |

---

## 13. Next Steps

1. User reviews this written spec.
2. On approval → `superpowers:writing-plans` to produce the Slice A implementation plan.
3. Implementation via TDD; later slices (S4–S8) each repeat spec → plan → build.
