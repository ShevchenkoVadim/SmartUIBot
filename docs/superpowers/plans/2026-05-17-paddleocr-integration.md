# PaddleOCR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognize text inside YOLO detection boxes with PaddleOCR, attach it to `Detection`, and let `behaviors.yaml` trigger on that text via a new `text_any` condition.

**Architecture:** A dedicated `OcrService` enrichment stage (own thread, size-1 drop-old `LatestQueue`, watchdog-supervised) sits between detection and decision. It subscribes to `DetectionsReady`, OCRs boxes whose label is in `ocr.labels`, and republishes `DetectionsEnriched`, which `DecisionService` and `DebugWindow` consume. OCR runs behind an `OcrEngine` Protocol (real `PaddleOcrEngine` with a lazy `paddleocr` import; `FakeOcrEngine` for headless tests). Off by default; pure pass-through when disabled.

**Tech Stack:** Python 3.12, PyQt6, mss, ultralytics, numpy (pinned `>=1.26,<2`), pytest. Gates (run from repo root, use the venv): `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`.

Spec: `docs/superpowers/specs/2026-05-17-paddleocr-integration-design.md`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/smartuibot/core/types.py` | `Detection` gains `text`/`text_confidence` | modify |
| `src/smartuibot/core/events.py` | `DetectionsEnriched` event | modify |
| `src/smartuibot/vision/ocr/__init__.py` | package marker | create |
| `src/smartuibot/vision/ocr/engine.py` | `OcrEngine` Protocol | create |
| `src/smartuibot/vision/ocr/paddle.py` | `PaddleOcrEngine` (lazy import) | create |
| `src/smartuibot/vision/ocr/service.py` | `OcrService` enrichment stage | create |
| `tests/fakes/ocr.py` | `FakeOcrEngine` | create |
| `src/smartuibot/core/config.py` | `OcrConfig` + `AppConfig` wiring | modify |
| `src/smartuibot/ai/world_state.py` | `best_match` text filter + `_normalize` | modify |
| `src/smartuibot/ai/behavior.py` | `Condition.text_any` | modify |
| `src/smartuibot/ai/registry.py` | parse/validate `text_any` | modify |
| `src/smartuibot/ai/service.py` | subscribe `DetectionsEnriched` | modify |
| `src/smartuibot/ui/debug_window.py` | subscribe `DetectionsEnriched`; show text | modify |
| `src/smartuibot/core/container.py` | construct/wire `OcrService` | modify |
| `src/smartuibot/app.py` | `_make_ocr_engine` factory | modify |
| `configs/default.yaml` | `ocr:` block | modify |
| `configs/behaviors.yaml` | `close_popup` `text_any` example | modify |
| `pyproject.toml` | optional `[ocr]` extra, mypy overrides, `ocr` marker | modify |
| `README.md` / `README.ru.md` / `SETUP.md` | OCR stage + caveat + test cmd | modify |

**Note on dependency packaging:** the spec says "add paddleocr/paddlepaddle to dependencies". This plan installs them as an **optional extra** `[ocr]`, not hard deps. This is the faithful realization of the spec's own "off by default / opt-in / lazy import / FakeOcrEngine tests" intent: a hard dep would force every `pip install -e .[dev]` and CI to install paddlepaddle (large; fragile x86_64-darwin wheels) for a feature that is disabled by default. Flagged again in Self-Review.

---

### Task 1: `Detection.text` / `text_confidence`

**Files:**
- Modify: `src/smartuibot/core/types.py`
- Test: `tests/unit/test_types.py`

- [ ] **Step 1: Append the failing tests**

Add to the end of `tests/unit/test_types.py`:

```python
def test_detection_defaults_have_no_text() -> None:
    from smartuibot.core.types import Detection

    d = Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2, y2=2)
    assert d.text is None
    assert d.text_confidence == 0.0


def test_detection_with_text_and_validation() -> None:
    import pytest

    from smartuibot.core.types import Detection

    d = Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2,
                  y2=2, text="Close", text_confidence=0.9)
    assert d.text == "Close" and d.text_confidence == 0.9
    with pytest.raises(ValueError):
        Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2,
                  y2=2, text="x", text_confidence=1.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_types.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'text'`.

- [ ] **Step 3: Add the fields**

In `src/smartuibot/core/types.py`, in the `Detection` dataclass, add the two fields immediately after `track_id: int | None = None`:

```python
    track_id: int | None = None
    text: str | None = None
    text_confidence: float = 0.0
```

And extend its `__post_init__` (currently only the confidence check):

```python
    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.text_confidence <= 1.0:
            raise ValueError("text_confidence must be in [0, 1]")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_types.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: ruff clean; mypy `Success`; full suite green.

```bash
git add src/smartuibot/core/types.py tests/unit/test_types.py
git commit -m "feat(types): Detection.text + text_confidence (validated)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `DetectionsEnriched` event

**Files:**
- Modify: `src/smartuibot/core/events.py`
- Test: `tests/unit/test_events.py`

- [ ] **Step 1: Append the failing test**

Add to the end of `tests/unit/test_events.py`:

```python
def test_detections_enriched_event_shape() -> None:
    import numpy as np

    from smartuibot.core.events import DetectionsEnriched
    from smartuibot.core.types import ROI, Detection, Frame

    roi = ROI(monitor=1, x=0, y=0, width=4, height=4)
    frame = Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), timestamp=0.0,
                  seq=1, roi=roi)
    det = Detection(label="b", confidence=0.5, class_id=0, x1=0, y1=0, x2=2,
                    y2=2, text="Close", text_confidence=0.9)
    ev = DetectionsEnriched(frame=frame, detections=(det,))
    assert ev.detections[0].text == "Close"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_events.py -q`
Expected: FAIL — `ImportError: cannot import name 'DetectionsEnriched'`.

- [ ] **Step 3: Add the event**

In `src/smartuibot/core/events.py`, directly after the `DetectionsReady` class, add:

```python
@dataclass(frozen=True, slots=True)
class DetectionsEnriched(Event):
    frame: Frame
    detections: tuple[Detection, ...]
```

(`Frame` and `Detection` are already imported at the top of `events.py`.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_events.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/core/events.py tests/unit/test_events.py
git commit -m "feat(events): DetectionsEnriched (detection stream + OCR text)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `OcrEngine` Protocol + `FakeOcrEngine`

**Files:**
- Create: `src/smartuibot/vision/ocr/__init__.py`, `src/smartuibot/vision/ocr/engine.py`
- Create: `tests/fakes/ocr.py`
- Test: `tests/unit/test_ocr_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ocr_engine.py`:

```python
import numpy as np

from smartuibot.vision.ocr.engine import OcrEngine
from tests.fakes.ocr import FakeOcrEngine


def test_fake_ocr_engine_satisfies_protocol_and_returns_scripted() -> None:
    eng = FakeOcrEngine(text="Close", confidence=0.92)
    assert isinstance(eng, OcrEngine)
    text, conf = eng.recognize(np.zeros((4, 4, 3), dtype=np.uint8))
    assert text == "Close" and conf == 0.92


def test_fake_ocr_engine_default_is_empty() -> None:
    text, conf = FakeOcrEngine().recognize(np.zeros((2, 2, 3), dtype=np.uint8))
    assert text == "" and conf == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_ocr_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'smartuibot.vision.ocr'`.

- [ ] **Step 3: Create the package + Protocol**

Create `src/smartuibot/vision/ocr/__init__.py` (empty file).

Create `src/smartuibot/vision/ocr/engine.py`:

```python
# src/smartuibot/vision/ocr/engine.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from smartuibot.core.types import Image


@runtime_checkable
class OcrEngine(Protocol):
    def recognize(self, image: Image) -> tuple[str, float]:
        """Return (text, confidence in [0,1]) for one cropped box image.
        ('', 0.0) when nothing is read."""
```

Create `tests/fakes/ocr.py`:

```python
# tests/fakes/ocr.py
from __future__ import annotations

from smartuibot.core.types import Image


class FakeOcrEngine:
    """Deterministic OCR for headless tests: returns a fixed (text, conf)."""

    def __init__(self, text: str = "", confidence: float = 0.0) -> None:
        self._text = text
        self._conf = confidence

    def recognize(self, image: Image) -> tuple[str, float]:
        return self._text, self._conf
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_ocr_engine.py -q`
Expected: PASS — 2 passed.

- [ ] **Step 5: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/vision/ocr/__init__.py src/smartuibot/vision/ocr/engine.py tests/fakes/ocr.py tests/unit/test_ocr_engine.py
git commit -m "feat(ocr): OcrEngine Protocol + FakeOcrEngine

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `PaddleOcrEngine` (real, lazy import)

**Files:**
- Create: `src/smartuibot/vision/ocr/paddle.py`
- Test: `tests/unit/test_paddle_ocr.py`

- [ ] **Step 1: Write the tests**

Create `tests/unit/test_paddle_ocr.py`:

```python
import numpy as np
import pytest


def test_paddle_engine_module_imports_without_paddle() -> None:
    # Module import must NOT import paddleocr (lazy in __init__).
    from smartuibot.vision.ocr.paddle import PaddleOcrEngine

    assert PaddleOcrEngine is not None


@pytest.mark.ocr
def test_paddle_engine_reads_text() -> None:
    from smartuibot.vision.ocr.paddle import PaddleOcrEngine

    eng = PaddleOcrEngine(lang="en")
    img = np.full((40, 160, 3), 255, dtype=np.uint8)  # blank -> '' is fine
    text, conf = eng.recognize(img)
    assert isinstance(text, str) and 0.0 <= conf <= 1.0
```

- [ ] **Step 2: Run to verify the non-marked test fails**

Run: `.venv/bin/python -m pytest tests/unit/test_paddle_ocr.py -q -m "not ocr"`
Expected: FAIL — `ModuleNotFoundError: No module named 'smartuibot.vision.ocr.paddle'`.

- [ ] **Step 3: Create `PaddleOcrEngine`**

Create `src/smartuibot/vision/ocr/paddle.py`:

```python
# src/smartuibot/vision/ocr/paddle.py
from __future__ import annotations

from typing import Any

from smartuibot.core.types import Image


class PaddleOcrEngine:
    """Real OCR via PaddleOCR. `paddleocr` is imported lazily in __init__ so
    the heavy dependency never loads for headless tests / disabled OCR."""

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR  # lazy: heavy optional dependency

        self._ocr: Any = PaddleOCR(lang=lang, show_log=False, use_gpu=False)

    def recognize(self, image: Image) -> tuple[str, float]:
        result = self._ocr.ocr(image, cls=False)
        if not result or not result[0]:
            return "", 0.0
        texts: list[str] = []
        confs: list[float] = []
        for line in result[0]:
            txt, conf = line[1]
            texts.append(str(txt))
            confs.append(float(conf))
        if not texts:
            return "", 0.0
        return " ".join(texts), min(confs)
```

- [ ] **Step 4: Run to verify the non-marked test passes**

Run: `.venv/bin/python -m pytest tests/unit/test_paddle_ocr.py -q -m "not ocr"`
Expected: PASS — `test_paddle_engine_module_imports_without_paddle` passes; the `ocr`-marked test is deselected.

- [ ] **Step 5: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green (the `ocr` marker is registered in Task 12; until then pytest emits an unknown-marker warning but does not fail — acceptable between tasks).

```bash
git add src/smartuibot/vision/ocr/paddle.py tests/unit/test_paddle_ocr.py
git commit -m "feat(ocr): PaddleOcrEngine with lazy paddleocr import

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `OcrService` enrichment stage

**Files:**
- Create: `src/smartuibot/vision/ocr/service.py`
- Test: `tests/unit/test_ocr_service.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ocr_service.py`:

```python
import time

import numpy as np

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsEnriched, DetectionsReady
from smartuibot.core.types import ROI, Detection, Frame
from smartuibot.vision.ocr.service import OcrService
from tests.fakes.ocr import FakeOcrEngine

_ROI = ROI(monitor=1, x=0, y=0, width=20, height=20)


def _frame() -> Frame:
    return Frame(image=np.zeros((20, 20, 3), dtype=np.uint8),
                 timestamp=time.monotonic(), seq=1, roi=_ROI)


def _det(label: str, box: tuple[int, int, int, int]) -> Detection:
    return Detection(label=label, confidence=0.9, class_id=0,
                     x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def _run(svc: OcrService, bus: EventBus, ev: DetectionsReady
         ) -> DetectionsEnriched:
    out: list[DetectionsEnriched] = []
    bus.subscribe(DetectionsEnriched, out.append)
    svc.start()
    bus.publish(ev)
    time.sleep(0.2)
    svc.stop()
    assert out, "OcrService did not publish DetectionsEnriched"
    return out[-1]


def test_enriches_configured_label_only() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),
                                     _det("enemy", (2, 2, 8, 8))))
    res = _run(svc, bus, ev)
    by_label = {d.label: d for d in res.detections}
    assert by_label["button"].text == "Close"
    assert by_label["button"].text_confidence == 0.9
    assert by_label["enemy"].text is None  # not a configured label


def test_pass_through_when_disabled() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=False)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None


def test_pass_through_when_engine_none() -> None:
    bus = EventBus()
    svc = OcrService(engine=None, bus=bus, labels=frozenset({"button"}),
                     min_confidence=0.5, enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None


def test_min_confidence_gates_text() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.3), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None  # 0.3 < 0.5


def test_degenerate_crop_is_skipped() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (5, 5, 5, 5)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None


def test_recognize_exception_leaves_text_none_and_still_publishes() -> None:
    class _Boom:
        def recognize(self, image: object) -> tuple[str, float]:
            raise RuntimeError("ocr exploded")

    bus = EventBus()
    svc = OcrService(engine=_Boom(), bus=bus, labels=frozenset({"button"}),
                     min_confidence=0.5, enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_ocr_service.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'smartuibot.vision.ocr.service'`.

- [ ] **Step 3: Create `OcrService`**

Create `src/smartuibot/vision/ocr/service.py`:

```python
# src/smartuibot/vision/ocr/service.py
from __future__ import annotations

import logging
from dataclasses import replace

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsEnriched, DetectionsReady, FpsTick
from smartuibot.core.fps import FpsMeter
from smartuibot.core.latest_queue import LatestQueue
from smartuibot.core.service import Service
from smartuibot.core.types import Detection, Image
from smartuibot.vision.ocr.engine import OcrEngine

_log = logging.getLogger("smartuibot.ocr")


def _crop(img: Image, d: Detection) -> Image | None:
    h, w = img.shape[:2]
    x1 = max(0, min(d.x1, w))
    x2 = max(0, min(d.x2, w))
    y1 = max(0, min(d.y1, h))
    y2 = max(0, min(d.y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


class OcrService(Service):
    """Crops configured-label YOLO boxes, OCRs them, and republishes the
    detection stream as DetectionsEnriched. Pure pass-through when disabled,
    engine is None, or no detection matches the configured labels."""

    def __init__(
        self,
        engine: OcrEngine | None,
        bus: EventBus,
        labels: frozenset[str],
        min_confidence: float,
        enabled: bool,
    ) -> None:
        super().__init__(name="ocr", bus=bus)
        self._engine = engine
        self._labels = labels
        self._min_confidence = min_confidence
        self._enabled = enabled
        self._queue: LatestQueue[DetectionsReady] = LatestQueue()
        self._fps = FpsMeter(window=30)
        self._warned = False
        bus.subscribe(DetectionsReady, self._on_detections)

    def _on_detections(self, event: DetectionsReady) -> None:
        self._queue.put(event)

    def run_once(self) -> None:
        ev = self._queue.get(timeout=0.1)
        if ev is None:
            return
        dets = ev.detections
        active = (
            self._enabled
            and self._engine is not None
            and any(d.label in self._labels for d in dets)
        )
        if active:
            assert self._engine is not None
            out: list[Detection] = []
            for d in dets:
                if d.label not in self._labels:
                    out.append(d)
                    continue
                crop = _crop(ev.frame.image, d)
                if crop is None:
                    out.append(d)
                    continue
                try:
                    text, conf = self._engine.recognize(crop)
                except Exception:  # noqa: BLE001 - one bad crop must not fail the frame
                    if not self._warned:
                        _log.warning("OCR recognize failed; leaving text empty",
                                     exc_info=True)
                        self._warned = True
                    out.append(d)
                    continue
                if text and conf >= self._min_confidence:
                    out.append(replace(d, text=text, text_confidence=conf))
                else:
                    out.append(d)
            dets = tuple(out)
        self._bus.publish(DetectionsEnriched(frame=ev.frame, detections=dets))
        self._fps.tick()
        self._bus.publish(FpsTick(name="ocr", fps=self._fps.fps))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_ocr_service.py -q`
Expected: PASS — 6 passed.

- [ ] **Step 5: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/vision/ocr/service.py tests/unit/test_ocr_service.py
git commit -m "feat(ocr): OcrService enrichment stage (pass-through when off)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `OcrConfig` + `AppConfig` wiring + `default.yaml`

**Files:**
- Modify: `src/smartuibot/core/config.py`
- Modify: `configs/default.yaml`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Append the failing tests**

Add to the end of `tests/unit/test_config.py`:

```python
def test_ocr_defaults_when_block_absent(tmp_path: Path) -> None:
    default = _write(tmp_path, "d.yaml", """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {
            model: yolo11n.pt, confidence: 0.35, device: auto,
            tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """)
    cfg = load_config(default)
    assert cfg.ocr.enabled is False
    assert cfg.ocr.labels == []
    assert cfg.ocr.lang == "en"
    assert cfg.ocr.min_confidence == 0.5


def test_ocr_block_parsed_and_validated(tmp_path: Path) -> None:
    base = """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {
            model: yolo11n.pt, confidence: 0.35, device: auto,
            tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """
    good = _write(tmp_path, "g.yaml", base + """
        ocr: {enabled: true, labels: [button, popup], lang: en,
              min_confidence: 0.6}
    """)
    cfg = load_config(good)
    assert cfg.ocr.enabled is True
    assert cfg.ocr.labels == ["button", "popup"]
    assert cfg.ocr.min_confidence == 0.6

    bad = _write(tmp_path, "b.yaml", base + """
        ocr: {enabled: true, labels: [], lang: en, min_confidence: 9.0}
    """)
    with pytest.raises(ValueError):
        load_config(bad)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py -q`
Expected: FAIL — `AttributeError: 'AppConfig' object has no attribute 'ocr'`.

- [ ] **Step 3: Add `OcrConfig` and wire it**

In `src/smartuibot/core/config.py`, add this dataclass directly after the `InputConfig` dataclass:

```python
@dataclass(frozen=True, slots=True)
class OcrConfig:
    enabled: bool
    labels: list[str]
    lang: str
    min_confidence: float
```

In the `AppConfig` dataclass, add an `ocr` field next to the existing
`input` default-factory field (keep it above `behaviors_path`):

```python
    ocr: OcrConfig = field(
        default_factory=lambda: OcrConfig(False, [], "en", 0.5))
    behaviors_path: str = "configs/behaviors.yaml"
```

Add to `AppConfig.__post_init__`, after the existing `overshoot_prob` check:

```python
        if not 0.0 <= self.ocr.min_confidence <= 1.0:
            raise ValueError("ocr.min_confidence must be in [0, 1]")
```

In `load_config`, in the `AppConfig(...)` construction, add the `ocr`
argument directly before `behaviors_path=...`:

```python
        ocr=OcrConfig(**data["ocr"]) if "ocr" in data
            else OcrConfig(False, [], "en", 0.5),
        behaviors_path=str(data.get("behaviors_path", "configs/behaviors.yaml")),
```

- [ ] **Step 4: Add the `ocr:` block to `configs/default.yaml`**

In `configs/default.yaml`, insert this block immediately before the final
`behaviors_path: configs/behaviors.yaml` line:

```yaml
ocr:
  enabled: false       # opt-in: heavy optional dep + CPU cost
  labels: [button, popup, dialog]
  lang: en
  min_confidence: 0.5
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py -q`
Expected: PASS.

- [ ] **Step 6: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/core/config.py configs/default.yaml tests/unit/test_config.py
git commit -m "feat(config): OcrConfig block (off by default)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `Condition.text_any` + `WorldState.best_match` text filter

**Files:**
- Modify: `src/smartuibot/ai/world_state.py`
- Modify: `src/smartuibot/ai/behavior.py`
- Test: `tests/unit/test_world_state.py`, `tests/unit/test_behavior.py`

- [ ] **Step 1: Append the failing tests**

Add to the end of `tests/unit/test_world_state.py`:

```python
def _det_t(label: str, conf: float, text: str | None) -> Detection:
    return Detection(label=label, confidence=conf, class_id=0, x1=0, y1=0,
                     x2=10, y2=10, text=text)


def test_best_match_text_any_substring_normalized() -> None:
    ws = WorldState(
        detections=(_det_t("button", 0.9, "  Close  Window "),),
        roi=_ROI, tick=1, recent=())
    assert ws.best_match(frozenset({"button"}), 0.0, 1,
                         frozenset({"close window"})) is not None
    assert ws.best_match(frozenset({"button"}), 0.0, 1,
                         frozenset({"cancel"})) is None


def test_best_match_text_any_empty_is_unconstrained() -> None:
    ws = WorldState(detections=(_det_t("button", 0.9, None),),
                    roi=_ROI, tick=1, recent=())
    assert ws.best_match(frozenset({"button"}), 0.0, 1) is not None


def test_best_match_none_text_never_matches_text_any() -> None:
    ws = WorldState(detections=(_det_t("button", 0.9, None),),
                    roi=_ROI, tick=1, recent=())
    assert ws.best_match(frozenset({"button"}), 0.0, 1,
                         frozenset({"close"})) is None
```

Add to the end of `tests/unit/test_behavior.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_world_state.py tests/unit/test_behavior.py -q`
Expected: FAIL — `TypeError: best_match() takes ... positional arguments` / `Condition.__init__() got an unexpected keyword argument 'text_any'`.

- [ ] **Step 3: Extend `WorldState.best_match` + add `_normalize`**

In `src/smartuibot/ai/world_state.py`, add at module level (after the imports, before the `WorldState` dataclass):

```python
def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()
```

Replace the `best_match` method with:

```python
    def best_match(
        self,
        labels: frozenset[str],
        min_confidence: float,
        min_count: int,
        text_any: frozenset[str] = frozenset(),
    ) -> Detection | None:
        matches = sorted(
            (d for d in self.detections
             if d.label in labels and d.confidence >= min_confidence
             and (not text_any or (
                 d.text is not None
                 and any(n in _normalize(d.text) for n in text_any)))),
            key=lambda d: d.confidence,
            reverse=True,
        )
        if not matches or len(matches) < min_count:
            return None
        return matches[0]
```

- [ ] **Step 4: Add `text_any` to `Condition`**

In `src/smartuibot/ai/behavior.py`, replace the `Condition` dataclass with:

```python
@dataclass(frozen=True, slots=True)
class Condition:
    labels: frozenset[str]
    min_confidence: float = 0.0
    min_count: int = 1
    text_any: frozenset[str] = frozenset()

    def match(self, ws: WorldState) -> Detection | None:
        return ws.best_match(self.labels, self.min_confidence,
                             self.min_count, self.text_any)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_world_state.py tests/unit/test_behavior.py -q`
Expected: PASS.

- [ ] **Step 6: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green (existing `test_utility.py` / `test_decision_service.py` unaffected — `text_any` defaults to empty).

```bash
git add src/smartuibot/ai/world_state.py src/smartuibot/ai/behavior.py tests/unit/test_world_state.py tests/unit/test_behavior.py
git commit -m "feat(ai): Condition.text_any substring filter (normalized)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `registry` parses + validates `text_any`

**Files:**
- Modify: `src/smartuibot/ai/registry.py`
- Test: `tests/unit/test_registry.py`

- [ ] **Step 1: Append the failing tests**

Add to the end of `tests/unit/test_registry.py`:

```python
def test_parses_text_any_normalized(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text(
        "behaviors:\n - name: close\n   base_utility: 1.0\n"
        "   condition: {labels: [popup], text_any: ['  Close ', 'OK']}\n"
        "   steps:\n    - {kind: click, target: detection}\n"
    )
    bs = load_behaviors(p)
    assert bs[0].condition.text_any == frozenset({"close", "ok"})


def test_text_any_absent_defaults_empty(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text(
        "behaviors:\n - name: x\n   base_utility: 1.0\n"
        "   condition: {labels: [a]}\n   steps: []\n"
    )
    assert load_behaviors(p)[0].condition.text_any == frozenset()


def test_rejects_empty_text_any_entry(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text(
        "behaviors:\n - name: x\n   base_utility: 1.0\n"
        "   condition: {labels: [a], text_any: ['ok', '  ']}\n   steps: []\n"
    )
    with pytest.raises(ValueError):
        load_behaviors(p)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_registry.py -q`
Expected: FAIL — `AssertionError` (`text_any` is `frozenset()` because not parsed) on `test_parses_text_any_normalized`.

- [ ] **Step 3: Parse + validate `text_any`**

In `src/smartuibot/ai/registry.py`, in `load_behaviors`, locate the block that
builds `cond` (currently):

```python
        cond_raw = raw["condition"]
        cond = Condition(
            labels=frozenset(str(x) for x in cond_raw["labels"]),
            min_confidence=float(cond_raw.get("min_confidence", 0.0)),
            min_count=int(cond_raw.get("min_count", 1)),
        )
```

Replace it with:

```python
        cond_raw = raw["condition"]
        raw_text_any = cond_raw.get("text_any", [])
        if not isinstance(raw_text_any, list):
            raise ValueError(
                f"behavior {raw.get('name')!r}: text_any must be a list")
        text_any: set[str] = set()
        for entry in raw_text_any:
            norm = " ".join(str(entry).split()).lower()
            if not norm:
                raise ValueError(
                    f"behavior {raw.get('name')!r}: text_any entries must "
                    f"be non-empty")
            text_any.add(norm)
        cond = Condition(
            labels=frozenset(str(x) for x in cond_raw["labels"]),
            min_confidence=float(cond_raw.get("min_confidence", 0.0)),
            min_count=int(cond_raw.get("min_count", 1)),
            text_any=frozenset(text_any),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/ai/registry.py tests/unit/test_registry.py
git commit -m "feat(ai): registry parses/validates condition.text_any

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Route `DecisionService` to `DetectionsEnriched`

This switch breaks direct-publish tests; the test updates are part of this
task so the suite stays green at commit.

**Files:**
- Modify: `src/smartuibot/ai/service.py`
- Modify: `tests/unit/test_decision_service.py`
- Modify: `tests/integration/test_closed_loop.py`
- Test: `tests/integration/test_closed_loop.py` (new text-gated case)

- [ ] **Step 1: Update `test_decision_service.py` to the new event**

In `tests/unit/test_decision_service.py`:

Change the import line
`from smartuibot.core.events import ActionRequested, DetectionsReady`
to:
```python
from smartuibot.core.events import ActionRequested, DetectionsEnriched
```
Replace both occurrences of
`bus.publish(DetectionsReady(frame=_frame(), detections=(_enemy(),)))`
with:
```python
    bus.publish(DetectionsEnriched(frame=_frame(), detections=(_enemy(),)))
```

- [ ] **Step 2: Update `test_closed_loop.py` to wire `OcrService`**

In `tests/integration/test_closed_loop.py`:

Add import after the existing `DetectionService` import:
```python
from smartuibot.vision.ocr.service import OcrService
```
In `_wire`, change the return type annotation and add an `ocr` service.
Replace the `_wire` signature line and the `return` line:

Signature →
```python
def _wire(
    mode: ModeFSM, backend: RecordingInputBackend
) -> tuple[EventBus, CaptureService, DetectionService, OcrService,
           DecisionService, ActionService]:
```
Directly before the existing `capture = CaptureService(...)` line, add:
```python
    ocr = OcrService(engine=None, bus=bus, labels=frozenset(),
                     min_confidence=0.0, enabled=False)
```
Change `return bus, capture, detection, decision, action` to:
```python
    return bus, capture, detection, ocr, decision, action
```
In `test_perceive_decide_act_closed_loop_headless`, change
`bus, capture, detection, decision, action = _wire(mode, backend)` to:
```python
    bus, capture, detection, ocr, decision, action = _wire(mode, backend)
```
and the start/stop loops:
```python
    for s in (action, decision, ocr, detection, capture):
        s.start()
    time.sleep(0.8)
    for s in (capture, detection, ocr, decision, action):
        s.stop()
```
In `test_disarm_halts_injection`, apply the same three changes (unpack with
`ocr`, start `(action, decision, ocr, detection, capture)`, stop
`(capture, detection, ocr, decision, action)`).

- [ ] **Step 3: Add a text-gated closed-loop test**

Append to `tests/integration/test_closed_loop.py`:

```python
def test_text_gated_behavior_fires_only_with_matching_ocr_text() -> None:
    import random as _random

    from smartuibot.ai.utility import UtilityPolicy
    from tests.fakes.ocr import FakeOcrEngine

    mode = ModeFSM()
    mode.arm()
    backend = RecordingInputBackend()
    bus = EventBus()
    behaviors = (Behavior(name="close",
                          condition=Condition(labels=frozenset({"button"}),
                                               min_confidence=0.1,
                                               text_any=frozenset({"close"})),
                          base_utility=5.0, scale_by_confidence=False,
                          steps=(BehaviorStep(kind="click",
                                              target="detection"),)),)
    policy = UtilityPolicy(behaviors, tick_hz=50.0, anti_loop_window=5,
                           anti_loop_max_repeats=99, hesitation_prob=0.0,
                           rng=_random.Random(1))
    decision = DecisionService(bus=bus, policy=policy,
                               tracker=WorldStateTracker(), mode=mode,
                               tick_hz=50.0)
    action = ActionService(bus=bus, backend=backend, mode=mode,
                           motion=MotionParams(2, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
                           max_actions_per_second=1000.0, roi_confine=True,
                           rng=_random.Random(2))
    ocr = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    detection = DetectionService(detector=FakeDetector(
        scripted=[[("button", 0.9, 4, 4, 12, 12)]] * 500), bus=bus,
        smoothing_frames=1, confidence=0.1)
    capture = CaptureService(backend=FakeCaptureBackend(), bus=bus, roi=_ROI,
                             target_fps=120)
    started: list[ActionStarted] = []
    bus.subscribe(ActionStarted, started.append)
    for s in (action, decision, ocr, detection, capture):
        s.start()
    time.sleep(0.8)
    for s in (capture, detection, ocr, decision, action):
        s.stop()
    assert started, "text-gated behavior did not fire with matching OCR text"
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_decision_service.py tests/integration/test_closed_loop.py -q`
Expected: FAIL — `DecisionService` still subscribes to `DetectionsReady`, so no `ActionRequested`/`ActionStarted` is produced (`assert out` / `assert started` fail).

- [ ] **Step 5: Switch `DecisionService` to `DetectionsEnriched`**

In `src/smartuibot/ai/service.py`:

Change the import
`from smartuibot.core.events import ActionRequested, DetectionsReady`
to:
```python
from smartuibot.core.events import ActionRequested, DetectionsEnriched
```
Change the subscription line
`bus.subscribe(DetectionsReady, self._on_detections)` to:
```python
        bus.subscribe(DetectionsEnriched, self._on_detections)
```
Change the handler signature
`def _on_detections(self, event: DetectionsReady) -> None:` to:
```python
    def _on_detections(self, event: DetectionsEnriched) -> None:
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_decision_service.py tests/integration/test_closed_loop.py -q`
Expected: PASS — including the new text-gated test.

- [ ] **Step 7: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/ai/service.py tests/unit/test_decision_service.py tests/integration/test_closed_loop.py
git commit -m "feat(ai): DecisionService consumes DetectionsEnriched

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: `DebugWindow` consumes `DetectionsEnriched` + shows text

**Files:**
- Modify: `src/smartuibot/ui/debug_window.py`
- Test: `tests/unit/test_debug_window.py`

- [ ] **Step 1: Update + extend the test**

In `tests/unit/test_debug_window.py`:

Change the import
`from smartuibot.core.events import DetectionsReady, FpsTick, LogRecord`
to:
```python
from smartuibot.core.events import DetectionsEnriched, FpsTick, LogRecord  # noqa: E402
```
In `test_debug_window_consumes_bus_events_without_error`, replace the
`bus.publish(DetectionsReady(...))` call with a detection carrying text and
add an assertion. Replace:

```python
    bus.publish(DetectionsReady(frame=frame, detections=(
        Detection(label="enemy", confidence=0.8, class_id=0, x1=1, y1=1, x2=5, y2=5),)))
```
with:
```python
    bus.publish(DetectionsEnriched(frame=frame, detections=(
        Detection(label="button", confidence=0.8, class_id=0, x1=1, y1=1,
                  x2=5, y2=5, text="Close", text_confidence=0.9),)))
```
After `assert win.detection_count() == 1`, add:
```python
    assert win.last_detection_texts() == ["Close"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_debug_window.py -q`
Expected: FAIL — `AttributeError: 'DebugWindow' object has no attribute 'last_detection_texts'` (and the window no longer receives the event, since it still subscribes to `DetectionsReady`).

- [ ] **Step 3: Update `DebugWindow`**

In `src/smartuibot/ui/debug_window.py`:

In the events import block, replace `DetectionsReady` with
`DetectionsEnriched` (keep the other names):

```python
from smartuibot.core.events import (
    ActionAborted,
    ActionCompleted,
    ActionStarted,
    DetectionsEnriched,
    FpsTick,
    LogRecord,
    ModeChanged,
)
```
Change the subscription line
`bus.subscribe(DetectionsReady, self._events.put)` to:
```python
        bus.subscribe(DetectionsEnriched, self._events.put)
```
In `_drain`, change `if isinstance(ev, DetectionsReady):` to:
```python
            if isinstance(ev, DetectionsEnriched):
```
In `__init__`, add a text cache next to the other state (e.g. after
`self._det_count = 0`):
```python
        self._det_texts: list[str] = []
```
Add this accessor method (next to `detection_count`):
```python
    def last_detection_texts(self) -> list[str]:
        return self._det_texts
```
Replace the `_on_detections` method with:
```python
    def _on_detections(self, ev: DetectionsEnriched) -> None:
        self._det_count = len(ev.detections)
        self._det_texts = [d.text for d in ev.detections if d.text]
        self._table.clear()
        for d in ev.detections:
            txt = f'  "{d.text}"' if d.text else ""
            self._table.addItem(
                f"{d.label}  {d.confidence:.2f}{txt}"
                f"  [{d.x1},{d.y1},{d.x2},{d.y2}]"
            )
        rendered = draw_boxes(ev.frame.image, list(ev.detections))
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        h = rgb.shape[0]
        w = rgb.shape[1]
        qimg = QImage(
            bytes(rgb.data), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)
        if w > self._max_w:
            pix = pix.scaledToWidth(
                self._max_w, Qt.TransformationMode.SmoothTransformation
            )
        self._preview.setPixmap(pix)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_debug_window.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/ui/debug_window.py tests/unit/test_debug_window.py
git commit -m "feat(ui): debug window shows OCR text; consumes DetectionsEnriched

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Container + app wiring

> **Plan revised after Task 9.** Task 9 already added a *minimal disabled
> pass-through* `self.ocr = OcrService(engine=None, bus=self.bus,
> labels=frozenset(), min_confidence=0.0, enabled=False)` to
> `AppContainer.__init__` (between `self.detection` and `self.mode`),
> already added `self.ocr` to the `Watchdog([...])` list (between
> `self.detection` and `self.decision`), and already added `self.ocr.start()`
> / `self.ocr.stop()` in the correct pipeline order. The `OcrService` import
> already exists in `container.py`. So this task only adds the
> **`ocr_engine` injection parameter** and replaces the hardcoded shim values
> with **config-driven** ones, plus the `app.py` factory + wiring + tests. Do
> NOT re-add the watchdog/start/stop wiring — verify it is already correct.

**Files:**
- Modify: `src/smartuibot/core/container.py`
- Modify: `src/smartuibot/app.py`
- Test: `tests/unit/test_container.py`

- [ ] **Step 1: Append the failing tests**

Add to the end of `tests/unit/test_container.py`:

```python
def test_container_injects_ocr_engine_and_enriches(tmp_path: Path) -> None:
    from smartuibot.core.events import DetectionsEnriched
    from tests.fakes.ocr import FakeOcrEngine

    default = tmp_path / "d.yaml"
    default.write_text(
        "capture: {backend: auto, target_fps: 120, monitor: 1}\n"
        "detection: {model: yolo11n.pt, confidence: 0.3, device: cpu, "
        "tracking: false, smoothing_frames: 1}\n"
        "ui: {preview_max_width: 960}\n"
        "logging: {level: INFO, dir: " + str(tmp_path / 'logs') + "}\n"
        "hotkeys: {emergency_stop: \"<ctrl>+<alt>+q\"}\n"
        "ocr: {enabled: true, labels: [enemy], lang: en, "
        "min_confidence: 0.5}\n"
    )
    cfg = load_config(default)
    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=16, height=16),
        capture_backend=FakeCaptureBackend(),
        detector=FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 4, 4)]] * 40),
        ocr_engine=FakeOcrEngine("Hello", 0.9),
    )
    assert container.ocr in container.watchdog._services
    enriched: list[DetectionsEnriched] = []
    container.bus.subscribe(DetectionsEnriched, enriched.append)
    container.start()
    time.sleep(0.3)
    container.stop()
    assert enriched, "expected DetectionsEnriched to flow through the pipeline"
    assert enriched[-1].detections[0].label == "enemy"
    assert enriched[-1].detections[0].text == "Hello"  # config-enabled + injected


def test_container_ocr_disabled_by_default_passthrough(tmp_path: Path) -> None:
    from smartuibot.core.events import DetectionsEnriched
    from tests.fakes.ocr import FakeOcrEngine

    default = tmp_path / "d.yaml"
    default.write_text(
        "capture: {backend: auto, target_fps: 120, monitor: 1}\n"
        "detection: {model: yolo11n.pt, confidence: 0.3, device: cpu, "
        "tracking: false, smoothing_frames: 1}\n"
        "ui: {preview_max_width: 960}\n"
        "logging: {level: INFO, dir: " + str(tmp_path / 'logs') + "}\n"
        "hotkeys: {emergency_stop: \"<ctrl>+<alt>+q\"}\n"
    )  # no ocr: block -> OcrConfig defaults (enabled=False)
    cfg = load_config(default)
    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=16, height=16),
        capture_backend=FakeCaptureBackend(),
        detector=FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 4, 4)]] * 40),
        ocr_engine=FakeOcrEngine("Hello", 0.9),
    )
    enriched: list[DetectionsEnriched] = []
    container.bus.subscribe(DetectionsEnriched, enriched.append)
    container.start()
    time.sleep(0.3)
    container.stop()
    assert enriched, "pass-through must still emit DetectionsEnriched"
    assert enriched[-1].detections[0].label == "enemy"
    assert enriched[-1].detections[0].text is None  # disabled -> no OCR
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_container.py -q`
Expected: FAIL — `TypeError: AppContainer.__init__() got an unexpected
keyword argument 'ocr_engine'` (the injection parameter does not exist yet;
the shim added in Task 9 hardcodes `engine=None`/`enabled=False`).

- [ ] **Step 3: Add the `ocr_engine` param + make the shim config-driven**

In `src/smartuibot/core/container.py`:

Add the `OcrEngine` import next to the existing
`from smartuibot.vision.ocr.service import OcrService` line (added in
Task 9):
```python
from smartuibot.vision.ocr.engine import OcrEngine
```
Add an `ocr_engine` parameter to `AppContainer.__init__` (after
`input_backend`):
```python
        input_backend: InputBackend | None = None,
        ocr_engine: OcrEngine | None = None,
```
Replace the Task-9 shim line
```python
        self.ocr = OcrService(engine=None, bus=self.bus, labels=frozenset(),
                              min_confidence=0.0, enabled=False)
```
with the config/engine-driven construction (same location, between
`self.detection` and `self.mode`):
```python
        self.ocr = OcrService(
            engine=ocr_engine, bus=self.bus,
            labels=frozenset(config.ocr.labels),
            min_confidence=config.ocr.min_confidence,
            enabled=config.ocr.enabled)
```
**Do not change** the `Watchdog([...])` list, `start()`, or `stop()` — Task 9
already placed `self.ocr` correctly there. Verify (read the file) that the
watchdog list is `[self.capture, self.detection, self.ocr, self.decision,
self.action]`, `start()` order is action→decision→ocr→detection→capture→
watchdog, and `stop()` order is watchdog→(disarm)→capture→detection→ocr→
decision→action. If any of that is missing, add it; otherwise leave it.

- [ ] **Step 4: Add the `_make_ocr_engine` factory in `app.py`**

In `src/smartuibot/app.py`:

Add imports (top-level; `ruff -I` orders them):
```python
import logging
from smartuibot.vision.ocr.engine import OcrEngine
```
Add this factory next to the other `_make_*` factories:
```python
def _make_ocr_engine(config: AppConfig) -> OcrEngine | None:
    if not config.ocr.enabled:
        return None
    try:
        from smartuibot.vision.ocr.paddle import PaddleOcrEngine

        return PaddleOcrEngine(lang=config.ocr.lang)
    except Exception:  # noqa: BLE001 - OCR is best-effort; degrade gracefully
        logging.getLogger("smartuibot.ocr").warning(
            "PaddleOCR unavailable; running with OCR disabled", exc_info=True)
        return None
```
In `build_real_container`, pass it through:
```python
    return AppContainer(
        config=config,
        roi=roi,
        capture_backend=_make_capture_backend(config),
        detector=_make_detector(config),
        input_backend=_make_input_backend(config),
        ocr_engine=_make_ocr_engine(config),
    )
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/test_container.py tests/integration/test_pipeline.py -q`
Expected: PASS (`test_pipeline` still subscribes to `DetectionsReady`, which `DetectionService` still emits — unaffected by the pass-through stage).

- [ ] **Step 6: Gate + commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: all green.

```bash
git add src/smartuibot/core/container.py src/smartuibot/app.py tests/unit/test_container.py
git commit -m "feat(app): wire OcrService into the pipeline (pass-through default)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Packaging, behaviors example, docs, final gate

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/behaviors.yaml`
- Modify: `README.md`, `README.ru.md`, `SETUP.md`

- [ ] **Step 1: Update `pyproject.toml`**

Add an optional extra to `[project.optional-dependencies]` (alongside the
existing `dev` entry):
```toml
ocr = ["paddleocr>=2.7", "paddlepaddle>=2.5"]
```
The `paddleocr.*` / `paddle.*` mypy `ignore_missing_imports` override was
already added in Task 4 (moved earlier to avoid an unused-`type: ignore`
under strict `warn_unused_ignores`). **Do not re-add it.** Just verify the
existing `[[tool.mypy.overrides]]` block already reads:
```toml
[[tool.mypy.overrides]]
module = ["mss.*", "ultralytics.*", "cv2.*", "pynput.*", "pydirectinput.*", "paddleocr.*", "paddle.*"]
ignore_missing_imports = true
```
Add the `ocr` marker to `[tool.pytest.ini_options]` markers:
```toml
markers = [
    "model: requires downloading YOLO weights (skipped offline)",
    "ocr: requires PaddleOCR + paddlepaddle installed (skipped by default)",
]
```

- [ ] **Step 2: Update `configs/behaviors.yaml`**

Replace the `close_popup` behavior with (note the comment):

```yaml
  # text_any requires ocr.enabled: true and the label listed in ocr.labels;
  # otherwise Detection.text is None and the text condition never matches.
  - name: close_popup
    base_utility: 8.0
    cooldown_s: 0.5
    condition: {labels: [popup, close_button], min_confidence: 0.5,
                text_any: [close, x, ok]}
    steps:
      - {kind: click, target: detection, button: left}
```

- [ ] **Step 3: Update docs**

In `README.md` "Testing" section, change the fast command to:
```
    pytest -q -m "not model and not ocr"   # fast, headless, no GPU/screen/OCR
```
In the README "How it works" pipeline section, add an OCR stage paragraph
after the Detection stage description:
> **Stage 2.5 — OCR enrichment** ([`vision/ocr/service.py`](src/smartuibot/vision/ocr/service.py)): `OcrService` subscribes to `DetectionsReady`, crops boxes whose label is in `ocr.labels`, recognizes text via the `OcrEngine` Protocol (`PaddleOcrEngine`, lazy import), attaches it to the `Detection`, and republishes `DetectionsEnriched` (consumed by decision + debug). Pure pass-through when `ocr.enabled` is false (the default).

Mirror the same paragraph (translated) into `README.ru.md` after its
Detection stage, and update its test command line identically.

In `SETUP.md`, add a section:
```markdown
## OCR (optional, off by default)
Text-in-detection-box OCR uses PaddleOCR. Install the extra:
`python -m pip install -e ".[ocr]"`. On Intel x86_64 macOS, paddlepaddle
ships only older CPU wheels and inference is slow — keep `ocr.labels` small.
Enable via `ocr.enabled: true` in `configs/default.yaml`.
```

- [ ] **Step 4: Final full gate**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q -m "not model and not ocr"`
Expected: ruff `All checks passed!`; mypy `Success: no issues found`; full suite green with **no unknown-marker warnings** (the `ocr` marker is now registered). Also run `grep -rn "DetectionsReady" src/smartuibot/ai src/smartuibot/ui` and confirm neither `ai/service.py` nor `ui/debug_window.py` still references `DetectionsReady`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/behaviors.yaml README.md README.ru.md SETUP.md
git commit -m "build(ocr): optional [ocr] extra, marker, mypy overrides; docs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Data model (`Detection.text`/`text_confidence` + validation) → Task 1. ✓
- `DetectionsEnriched` event → Task 2; Decision/Debug switched → Tasks 9, 10. ✓
- `OcrEngine` Protocol + `PaddleOcrEngine` (lazy) + `FakeOcrEngine` → Tasks 3, 4. ✓
- `OcrService` (configured-label crop, pass-through, min_confidence, degenerate crop, recognize-raises) → Task 5. ✓
- `OcrConfig` + default factory + `default.yaml` → Task 6. ✓
- `Condition.text_any` + `best_match` + `_normalize` → Task 7; registry parse/validate → Task 8. ✓
- Container always-present stage, watchdog 5 services, start/stop order; `app._make_ocr_engine` graceful fallback → Task 11. ✓
- Error handling: per-detection try/except + warn-once (Task 5); engine-construction fallback (Task 11). ✓
- behaviors.yaml example, mypy overrides, `ocr` marker, docs → Task 12. ✓
- Testing: fakes + unit + integration text-gated + pass-through; real path behind `ocr` marker → Tasks 3–12. ✓
- **Deliberate spec refinement:** paddle is an optional `[ocr]` extra, not a hard dependency. Faithful to the spec's "off by default / opt-in / lazy / FakeOcrEngine" intent; avoids forcing paddlepaddle into every dev/CI install. Documented in File Structure note and SETUP.md.

**Placeholder scan:** every code step shows complete code; every command has expected output; no TBD/TODO. ✓

**Type consistency:** `OcrEngine.recognize(image: Image) -> tuple[str, float]` is identical in Task 3 (Protocol), Task 4 (`PaddleOcrEngine`), Task 5 (`OcrService` call site), and the `tests/fakes/ocr.py` fake. `OcrService.__init__(engine: OcrEngine | None, bus, labels: frozenset[str], min_confidence: float, enabled: bool)` is identical across Tasks 5, 9, 11 and every test. `DetectionsEnriched(frame, detections)` matches `DetectionsReady`'s shape and all consumers. `Condition(labels, min_confidence, min_count, text_any)` and `best_match(labels, min_confidence, min_count, text_any=frozenset())` match across Tasks 7, 8, 9. `AppContainer(..., ocr_engine: OcrEngine | None = None)` matches the `app.py` call and the container tests (which omit it → `None` → pass-through). ✓

**Scope check:** single feature, one pipeline stage + its config/condition surface; appropriately sized for one plan. ✓
