# PaddleOCR Integration — Design

**Date:** 2026-05-17
**Status:** Approved (design), pending implementation plan
**Area:** `src/smartuibot/vision/ocr/` (new), `src/smartuibot/core/types.py`,
`src/smartuibot/core/events.py`, `src/smartuibot/ai/` (behavior, world_state,
registry), `src/smartuibot/core/config.py`, `src/smartuibot/core/container.py`,
`src/smartuibot/app.py`, `src/smartuibot/ui/debug_window.py`,
`configs/default.yaml`, `configs/behaviors.yaml`

## Problem

The bot only perceives objects via YOLO class labels. Many UI elements are
distinguished by their **text** (a "Close" vs "Buy" button, an "OK" vs
"Cancel" dialog), which a class label cannot express. We want to recognize
the text inside YOLO-detected boxes with PaddleOCR and let behaviors trigger
on that text.

## Scope

In scope:

- Run OCR **only inside YOLO detection boxes** whose label is in a
  configured set (`ocr.labels`); attach recognized text to the `Detection`.
- Make text matchable in `configs/behaviors.yaml` via a new optional
  `text_any` condition key (case-insensitive substring, whitespace-
  normalized, any-of).
- A dedicated `OcrService` pipeline stage (own thread, drop-old
  backpressure, watchdog-supervised), an `OcrEngine` Protocol, a real
  `PaddleOcrEngine`, and a `FakeOcrEngine` for headless tests.
- OCR is **off by default** (opt-in via `ocr.enabled`).

Out of scope (explicit, per design decisions):

- Full-frame / free-form screen OCR (only YOLO-box crops).
- Reading specific HUD numeric values, text-region detection without YOLO.
- Regex or exact-match condition semantics (substring any-of only).
- Auto-deriving the OCR label set from behavior config (explicit
  `ocr.labels` list only).
- Per-cycle throttling beyond the existing size-1 drop-old queue.
- Changing the `state.yaml` persistence format.

## Approach

**Approach A (chosen): a dedicated `OcrService` enrichment stage**, mirroring
`DetectionService` (`vision/detect/service.py:16`). It subscribes to
`DetectionsReady`, crops configured-label boxes, OCRs them via the
`OcrEngine` Protocol, and republishes a new `DetectionsEnriched` event that
`DecisionService` and `DebugWindow` consume. When OCR is disabled it is a
pure pass-through, so pipeline wiring stays uniform.

Rejected alternatives:

- **B — OCR inline in `DetectionService`** (after `infer`, before publish):
  simplest data flow, but serializes OCR into the already CPU-bound
  detection loop, dropping YOLO FPS every cycle on the Intel-mac CPU and
  breaking the project's per-stage isolation. Cannot throttle OCR
  independently.
- **C — OCR on demand inside `DecisionService`**: blocks the 10 Hz decision
  tick on OCR (blows the tick budget) and mixes responsibilities; hard to
  test deterministically.

Approach A is the only option that protects YOLO throughput on CPU-only
inference and matches the established Service / Protocol / fake / event
pattern used by every other stage.

## Design

### 1. Data model & event

`Detection` (`core/types.py:52`) gains two fields, defaulted so every
existing construction and test stays valid:

```
text: str | None = None        # OCR result for this box; None = not OCR'd
text_confidence: float = 0.0   # [0, 1]
```

`__post_init__` additionally validates `0.0 <= text_confidence <= 1.0`
(alongside the existing `confidence` check).

New event in `core/events.py`, identical shape to `DetectionsReady`:

```
@dataclass(frozen=True, slots=True)
class DetectionsEnriched(Event):
    frame: Frame
    detections: tuple[Detection, ...]
```

`DetectionsReady` remains the detection→OCR hop. `DecisionService`
(`ai/service.py:33`) and `DebugWindow` (`ui/debug_window.py:86`) switch their
subscription from `DetectionsReady` to `DetectionsEnriched`.

### 2. OCR engine — Protocol + implementations

New package `src/smartuibot/vision/ocr/`.

`engine.py` — Protocol parallel to `Detector` (`vision/detect/detector.py:11`):

```
@runtime_checkable
class OcrEngine(Protocol):
    def recognize(self, image: Image) -> tuple[str, float]:
        """(text, confidence in [0,1]) for one cropped box image;
        ('', 0.0) when nothing is read."""
```

`paddle.py` — `PaddleOcrEngine`: **lazily imports `paddleocr` inside
`__init__`** (exactly as `Yolo11Detector.__init__` lazily imports
`ultralytics`, `vision/detect/yolo.py:33`), so the heavy dependency never
loads in headless tests. It joins multi-line PaddleOCR output into one
space-separated string and reports the **minimum** line confidence;
configurable `lang`.

`tests/fakes/ocr.py` — `FakeOcrEngine`: returns deterministic text +
confidence (e.g. configurable mapping / fixed string) so unit and
integration tests are reproducible (mirrors `tests/fakes/detector.py`).

### 3. `OcrService` (the enrichment stage)

`src/smartuibot/vision/ocr/service.py`, extends `Service`
(`core/service.py:12`), mirrors `DetectionService`:

- Constructor: `(engine: OcrEngine | None, bus: EventBus,
  labels: frozenset[str], min_confidence: float, enabled: bool)`.
- Subscribes `DetectionsReady` → pushes into a
  `LatestQueue[DetectionsReady]` (`core/latest_queue.py:7`, drop-old).
- `run_once()`: pull newest (timeout 0.1; `None` → return). If
  `not enabled` **or** `engine is None` **or** no detection's label is in
  `labels` → publish `DetectionsEnriched` with the **unchanged** detections
  (pure pass-through). Otherwise build a new detections tuple containing
  **every** detection (those with `label ∉ labels` pass through unchanged in
  place); for each detection whose `label ∈ labels`:
  - Compute the crop `frame.image[y1:y2, x1:x2]` with coordinates clamped to
    the image bounds; skip degenerate crops (`x2<=x1`, `y2<=y1`, or zero
    size after clamp) leaving `text=None`.
  - Call `engine.recognize(crop)` inside a per-detection `try/except`; on
    exception leave `text=None` and log once at WARNING.
  - If `conf >= min_confidence`, replace the detection via
    `dataclasses.replace(det, text=…, text_confidence=conf)`; else leave
    `text=None`.
  - Publish `DetectionsEnriched(frame, enriched_tuple)` and
    `FpsTick(name="ocr", fps=…)` (same FpsMeter pattern as detection).

### 4. Decision-engine text matching

`Condition` (`ai/behavior.py:11`) gains `text_any: frozenset[str] =
frozenset()`. Empty (the default) means **no text constraint** → all
existing behaviors are byte-for-byte unchanged in behavior.

`WorldState.best_match` (`ai/world_state.py:17`) takes an extra
`text_any: frozenset[str]`. A detection matches when:

1. `d.label in labels` and `d.confidence >= min_confidence` (unchanged), and
2. `text_any` is empty, **or** `d.text is not None` and
   `_normalize(d.text)` contains any needle in `text_any`.

`min_count` applies to the post-text-filter matches; ordering stays by
`confidence` descending (text confidence is not used for ranking — YAGNI).
`_normalize(s)` = `" ".join(s.split()).lower()` (strip, collapse internal
whitespace, lowercase). Needles are pre-normalized at load.

`Condition.match` passes `self.text_any` through.

`registry.py` `load_behaviors` (`ai/registry.py:29`) parses optional
`text_any` from the condition mapping:
`frozenset(str(x).strip().lower() for x in cond_raw.get("text_any", []))`.
Validation (same fail-fast style as existing checks): value must be a list;
every entry non-empty after strip; otherwise raise `ValueError` naming the
behavior.

`configs/behaviors.yaml` — update `close_popup` as the worked example and
add a comment that `text_any` requires `ocr.enabled: true` and the label in
`ocr.labels`, else `text` is `None` and the condition never matches:

```
- name: close_popup
  base_utility: 8.0
  cooldown_s: 0.5
  condition: {labels: [popup, close_button], min_confidence: 0.5,
              text_any: ["close", "x", "ok"]}
  steps:
    - {kind: click, target: detection, button: left}
```

### 5. Configuration & wiring

New frozen `OcrConfig` in `core/config.py`:

```
@dataclass(frozen=True, slots=True)
class OcrConfig:
    enabled: bool
    labels: list[str]
    lang: str
    min_confidence: float
```

`AppConfig` gets an `ocr` field with a default factory
`OcrConfig(False, [], "en", 0.5)` and parses `data["ocr"]` if present (same
pattern as `decision`/`input`, `core/config.py:74-79,115-119`), so existing
config files without an `ocr:` block still load. `__post_init__` validates
`0.0 <= ocr.min_confidence <= 1.0`.

`configs/default.yaml` gains:

```
ocr:
  enabled: false        # opt-in: heavy dep + CPU cost
  labels: [button, popup, dialog]
  lang: en
  min_confidence: 0.5
```

`AppContainer` (`core/container.py:26`): construct `OcrService` **always**
(pass-through when disabled) between detection and decision; wire
`DecisionService` and `DebugWindow` to `DetectionsEnriched`. `start()` order
(reverse pipeline, consumers before producers): action, decision, ocr,
detection, capture. Watchdog list becomes
`[capture, detection, ocr, decision, action]` (`core/container.py:76`).

`app.py` adds `_make_ocr_engine(config) -> OcrEngine | None`: returns a
`PaddleOcrEngine` only when `config.ocr.enabled`, wrapped in try/except —
construction failure logs and returns `None` (bot runs OCR-disabled, like
the best-effort e-stop hotkey, `app.py:118-124`). Otherwise returns `None`.

`pyproject.toml`: add `paddleocr` and `paddlepaddle` to dependencies; add
`paddleocr.*`, `paddle.*` to the mypy `ignore_missing_imports` overrides
block. Add a new pytest marker `ocr` (like `model`).

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `OcrEngine` (Protocol) | crop image → `(text, conf)` | `Image` (value only) |
| `PaddleOcrEngine` | real OCR; lazy `paddleocr` import | `paddleocr` (lazy) |
| `FakeOcrEngine` (tests) | deterministic `(text, conf)` | — |
| `OcrService` | crop configured-label boxes, enrich, republish; pass-through when disabled | `OcrEngine`, `EventBus`, `LatestQueue`, `Service` |
| `DetectionsEnriched` (event) | detection stream carrying optional text | `Frame`, `Detection` |
| `Condition.text_any` + `best_match` | substring any-of text filter | `WorldState`, `Detection` |
| `registry.load_behaviors` | parse/validate `text_any` | `Condition` |
| `OcrConfig` + container wiring | config + always-present stage | `AppConfig`, `AppContainer` |

Each unit is understandable and testable in isolation; the heavy dependency
is confined to `PaddleOcrEngine` behind the Protocol.

## Error handling

- Per-detection `recognize()` exception → that box `text=None`, logged once
  at WARNING; one bad crop never fails the frame.
- `PaddleOcrEngine` construction failure (import/init) → caught in the
  `app.py` factory, logged, returns `None` → OCR-disabled pass-through; the
  bot keeps running.
- Degenerate / empty crop → skipped, `text=None`.
- Disabled or `None` engine → pass-through; `DecisionService` still receives
  `DetectionsEnriched` so the loop never stalls.
- An uncaught error inside `OcrService.run_once` still flows through the
  existing fatal `ServiceError` → `Watchdog` restart path, identical to
  every other service.

## Testing

Headless-first, mirroring the existing suite (`pytest -m "not model"`):

- `tests/fakes/ocr.py` `FakeOcrEngine`.
- `test_ocr_engine` — `OcrEngine` is `runtime_checkable`; fake satisfies it.
- `test_ocr_service` — enrich configured labels; pass-through when
  disabled / `None` engine / no matching labels; `min_confidence` gating;
  degenerate-crop skip; `recognize` raising → `text=None` + still publishes.
- `test_world_state` / `test_behavior` additions — `text_any` substring +
  normalization; empty `text_any` unchanged; `None` text never matches;
  `min_count` after text filter.
- `test_registry` — parse `text_any`; reject non-list / empty entries.
- `test_config*` / `test_container*` — `OcrConfig` defaults; container wires
  5 services, watchdog list, Decision/Debug on `DetectionsEnriched`.
- `test_debug_window` — text rendered in the detections panel.
- `tests/integration/test_closed_loop` — a text-gated behavior fires only
  when `FakeOcrEngine` yields matching text; an OCR-disabled run still
  drives the loop via pass-through.
- Real `PaddleOcrEngine` test marked `ocr`, deselected by default. Default
  command becomes `pytest -m "not model and not ocr"`; update README/SETUP.

The real Paddle path needs the heavy dependency and is verified separately;
the project already follows this pure-core / thin-adapter split.

## Touched files

- New: `src/smartuibot/vision/ocr/__init__.py`, `engine.py`, `paddle.py`,
  `service.py`; `tests/fakes/ocr.py`; new unit/integration tests.
- `src/smartuibot/core/types.py` — `Detection.text`, `text_confidence`
  (+ validation).
- `src/smartuibot/core/events.py` — `DetectionsEnriched`.
- `src/smartuibot/ai/behavior.py` — `Condition.text_any`, `match`.
- `src/smartuibot/ai/world_state.py` — `best_match` text filter +
  `_normalize`.
- `src/smartuibot/ai/registry.py` — parse/validate `text_any`.
- `src/smartuibot/ai/service.py` — subscribe `DetectionsEnriched`.
- `src/smartuibot/ui/debug_window.py` — subscribe `DetectionsEnriched`;
  show text.
- `src/smartuibot/core/config.py` — `OcrConfig` + `AppConfig` wiring +
  validation.
- `src/smartuibot/core/container.py` — construct/wire `OcrService`; watchdog
  list; `start()` order.
- `src/smartuibot/app.py` — `_make_ocr_engine` factory.
- `configs/default.yaml` — `ocr:` block.
- `configs/behaviors.yaml` — `close_popup` `text_any` example + comment.
- `pyproject.toml` — `paddleocr`/`paddlepaddle` deps, mypy overrides, `ocr`
  marker.
- `README.md` / `README.ru.md` / `SETUP.md` — OCR stage + paddlepaddle
  Intel-mac/CPU caveat + updated test command.

No changes to `ROI`, the capture/`Detector` Protocols, the mss backend, the
input stack, or the `state.yaml` format.
