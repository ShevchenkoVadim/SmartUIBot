# SmartUIBot Slice A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, read-only real-time pipeline — select a screen ROI, capture it at high FPS, run YOLO11 detection, and render boxes/FPS/detections/logs in a separate PyQt6 debug window — on a clean cross-platform foundation (DI, event bus, config, logging, watchdog, emergency-stop).

**Architecture:** Clean/hexagonal. A thread-safe event bus decouples worker threads (capture, detection) from the Qt UI on the main thread. Platform-specific capture is hidden behind a `CaptureBackend` Protocol. A size-1 latest-wins queue provides backpressure so inference always runs on the freshest frame. No mouse/keyboard injection anywhere (read-only guarantee).

**Tech Stack:** Python 3.12, PyQt6, Ultralytics YOLO11 (PyTorch), `mss` (capture), `numpy`, `PyYAML`, `pynput` (emergency-stop listener only), pytest/ruff/mypy.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Deps, ruff, mypy(strict), pytest config |
| `src/smartuibot/core/types.py` | `ROI`, `Frame`, `Detection` domain types |
| `src/smartuibot/core/events.py` | Event dataclasses |
| `src/smartuibot/core/event_bus.py` | Thread-safe pub/sub with subscriber isolation |
| `src/smartuibot/core/latest_queue.py` | Size-1 latest-wins queue (backpressure) |
| `src/smartuibot/core/fps.py` | Rolling FPS meter |
| `src/smartuibot/core/config.py` | YAML → typed dataclasses, layered + validated |
| `src/smartuibot/core/logging_setup.py` | Structured JSON + colored console + ring buffer→bus |
| `src/smartuibot/core/service.py` | `Service` base (thread, heartbeat, exception boundary) |
| `src/smartuibot/core/watchdog.py` | Heartbeat monitor + backoff restart |
| `src/smartuibot/core/container.py` | DI composition |
| `src/smartuibot/platform_support/detect.py` | OS detection + backend selection |
| `src/smartuibot/vision/capture/backend.py` | `CaptureBackend` Protocol, `Monitor` |
| `src/smartuibot/vision/capture/mss_backend.py` | macOS/universal mss adapter |
| `src/smartuibot/vision/capture/service.py` | `CaptureService` worker |
| `src/smartuibot/vision/detect/detector.py` | `Detector` Protocol |
| `src/smartuibot/vision/detect/smoothing.py` | Temporal smoothing filter |
| `src/smartuibot/vision/detect/yolo.py` | `Yolo11Detector` adapter |
| `src/smartuibot/vision/detect/service.py` | `DetectionService` worker |
| `src/smartuibot/ui/roi_selector.py` | Frameless translucent ROI drag-select |
| `src/smartuibot/ui/debug_window.py` | Debug window (preview/boxes/table/FPS/logs/controls) |
| `src/smartuibot/app.py` | Composition root |
| `run.py` | Launch script |
| `configs/default.yaml` | Default config |
| `tests/fakes/` | `FakeCaptureBackend`, `FakeDetector` |

**Conventions (apply to every task):** package importable as `smartuibot` (src layout); all code fully type-annotated; tests under `tests/` mirroring package paths; commit after every task with the message shown.

---

### Task 1: Project scaffold & tooling

**Files:**
- Create: `pyproject.toml`, `src/smartuibot/__init__.py`, `tests/__init__.py`, `configs/default.yaml`, `.python-version`
- Create empty packages: `src/smartuibot/{core,platform_support,vision,vision/capture,vision/detect,ui,ai,input,memory}/__init__.py`, `tests/{unit,integration,fakes,fixtures}/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "smartuibot"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "PyYAML>=6.0",
    "mss>=9.0",
    "opencv-python>=4.10",
    "PyQt6>=6.7",
    "ultralytics>=8.3",
    "pynput>=1.7",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-qt>=4.4", "ruff>=0.6", "mypy>=1.11"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["smartuibot"]
mypy_path = "src"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["model: requires downloading YOLO weights (skipped offline)"]
```

- [ ] **Step 2: Create package skeleton**

Run:
```bash
cd /Users/vadimshevchenko/tf_projects/SmartUIBot
mkdir -p src/smartuibot/{core,platform_support,vision/capture,vision/detect,ui,ai,input,memory} tests/{unit,integration,fakes,fixtures} configs
echo "3.12" > .python-version
find src/smartuibot tests -type d -exec touch {}/__init__.py \;
touch src/smartuibot/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create `configs/default.yaml`**

```yaml
capture:
  backend: auto        # auto | mss | dxcam
  target_fps: 60
  monitor: 1
detection:
  model: yolo11n.pt
  confidence: 0.35
  device: auto         # auto | cpu | cuda
  tracking: false
  smoothing_frames: 3
ui:
  preview_max_width: 960
logging:
  level: INFO
  dir: logs
hotkeys:
  emergency_stop: "<ctrl>+<alt>+q"
```

- [ ] **Step 4: Install dev environment**

Run:
```bash
.venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -c "import smartuibot, numpy, yaml, mss; print('ok')"
```
Expected: prints `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version configs/ src/ tests/
git commit -m "chore: project scaffold, tooling, config defaults"
```

---

### Task 2: Core domain types (`ROI`, `Frame`, `Detection`)

**Files:**
- Create: `src/smartuibot/core/types.py`
- Test: `tests/unit/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_types.py
import numpy as np
import pytest
from smartuibot.core.types import ROI, Frame, Detection


def test_roi_roundtrip_dict():
    roi = ROI(monitor=1, x=10, y=20, width=300, height=200)
    assert ROI.from_dict(roi.as_dict()) == roi


def test_roi_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        ROI(monitor=1, x=0, y=0, width=0, height=100)


def test_frame_carries_image_and_metadata():
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    roi = ROI(monitor=1, x=0, y=0, width=5, height=4)
    f = Frame(image=img, timestamp=1.0, seq=7, roi=roi)
    assert f.seq == 7 and f.image.shape == (4, 5, 3)


def test_detection_box_area_and_validation():
    d = Detection(label="x", confidence=0.9, class_id=0, x1=0, y1=0, x2=10, y2=4)
    assert d.area == 40
    with pytest.raises(ValueError):
        Detection(label="x", confidence=1.5, class_id=0, x1=0, y1=0, x2=1, y2=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_types.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.types`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ROI:
    monitor: int
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if self.monitor < 0:
            raise ValueError("ROI monitor index must be >= 0")

    def as_dict(self) -> dict[str, int]:
        return {
            "monitor": self.monitor,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ROI:
        return cls(
            monitor=int(d["monitor"]),
            x=int(d["x"]),
            y=int(d["y"]),
            width=int(d["width"]),
            height=int(d["height"]),
        )


@dataclass(slots=True)
class Frame:
    image: np.ndarray  # HxWx3 BGR uint8
    timestamp: float
    seq: int
    roi: ROI


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_types.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/types.py tests/unit/test_types.py
git commit -m "feat(core): ROI, Frame, Detection domain types"
```

---

### Task 3: Event definitions

**Files:**
- Create: `src/smartuibot/core/events.py`
- Test: `tests/unit/test_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_events.py
from smartuibot.core.events import (
    DetectionsReady, Event, FpsTick, FrameCaptured, LogRecord, ServiceError, StateChanged,
)


def test_events_are_subclasses_of_event():
    for cls in (FrameCaptured, DetectionsReady, FpsTick, ServiceError, LogRecord, StateChanged):
        assert issubclass(cls, Event)


def test_fps_tick_fields():
    e = FpsTick(name="capture", fps=42.5)
    assert e.name == "capture" and e.fps == 42.5


def test_service_error_defaults_nonfatal():
    e = ServiceError(service="capture", error="boom")
    assert e.fatal is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_events.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.events`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/events.py
from __future__ import annotations

from dataclasses import dataclass, field

from smartuibot.core.types import Detection, Frame


@dataclass(frozen=True, slots=True)
class Event:
    pass


@dataclass(frozen=True, slots=True)
class FrameCaptured(Event):
    frame: Frame


@dataclass(frozen=True, slots=True)
class DetectionsReady(Event):
    frame: Frame
    detections: tuple[Detection, ...]


@dataclass(frozen=True, slots=True)
class FpsTick(Event):
    name: str
    fps: float


@dataclass(frozen=True, slots=True)
class ServiceError(Event):
    service: str
    error: str
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class LogRecord(Event):
    level: str
    logger: str
    message: str
    ts: float


@dataclass(frozen=True, slots=True)
class StateChanged(Event):
    service: str
    state: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_events.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/events.py tests/unit/test_events.py
git commit -m "feat(core): event dataclasses"
```

---

### Task 4: Thread-safe EventBus with subscriber isolation

**Files:**
- Create: `src/smartuibot/core/event_bus.py`
- Test: `tests/unit/test_event_bus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_event_bus.py
import threading

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import FpsTick, ServiceError


def test_publish_delivers_to_matching_subscribers():
    bus = EventBus()
    seen: list[float] = []
    bus.subscribe(FpsTick, lambda e: seen.append(e.fps))
    bus.publish(FpsTick(name="c", fps=10.0))
    bus.publish(ServiceError(service="x", error="y"))  # no subscriber, must not raise
    assert seen == [10.0]


def test_subscriber_exception_is_isolated():
    bus = EventBus()
    seen: list[float] = []
    bus.subscribe(FpsTick, lambda e: (_ for _ in ()).throw(RuntimeError("bad")))
    bus.subscribe(FpsTick, lambda e: seen.append(e.fps))
    bus.publish(FpsTick(name="c", fps=5.0))  # must not raise
    assert seen == [5.0]


def test_publish_is_thread_safe():
    bus = EventBus()
    count = [0]
    lock = threading.Lock()

    def handler(_e: FpsTick) -> None:
        with lock:
            count[0] += 1

    bus.subscribe(FpsTick, handler)
    threads = [threading.Thread(target=bus.publish, args=(FpsTick(name="c", fps=1.0),))
               for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count[0] == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_event_bus.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.event_bus`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/event_bus.py
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

from smartuibot.core.events import Event

E = TypeVar("E", bound=Event)
_log = logging.getLogger("smartuibot.event_bus")


class EventBus:
    """Thread-safe synchronous pub/sub. A subscriber exception is logged and
    swallowed so it can never break the publisher or other subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[type[Event], list[Callable[..., None]]] = {}

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        with self._lock:
            self._subs.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._subs.get(type(event), ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - isolation is the whole point
                _log.exception("subscriber for %s failed", type(event).__name__)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_event_bus.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/event_bus.py tests/unit/test_event_bus.py
git commit -m "feat(core): thread-safe EventBus with subscriber isolation"
```

---

### Task 5: Size-1 latest-wins queue (backpressure)

**Files:**
- Create: `src/smartuibot/core/latest_queue.py`
- Test: `tests/unit/test_latest_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_latest_queue.py
import threading
import time

from smartuibot.core.latest_queue import LatestQueue


def test_put_overwrites_old_value():
    q: LatestQueue[int] = LatestQueue()
    q.put(1)
    q.put(2)
    q.put(3)
    assert q.get(timeout=0.1) == 3


def test_get_blocks_until_value_then_times_out():
    q: LatestQueue[int] = LatestQueue()
    assert q.get(timeout=0.05) is None
    result: list[int | None] = []

    def consumer() -> None:
        result.append(q.get(timeout=1.0))

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)
    q.put(99)
    t.join(timeout=1.0)
    assert result == [99]


def test_clear_discards_pending():
    q: LatestQueue[int] = LatestQueue()
    q.put(5)
    q.clear()
    assert q.get(timeout=0.05) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_latest_queue.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.latest_queue`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/latest_queue.py
# NOTE: Use PEP 695 generic syntax (`class LatestQueue[T]:`). The project's
# ruff UP ruleset (UP046) rejects `class X(Generic[T])`. Function-level
# TypeVar (as in event_bus.py) remains fine.
from __future__ import annotations

import threading


class LatestQueue[T]:
    """Holds at most one item. put() overwrites any pending item so consumers
    always see the freshest value (drop-old backpressure)."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._item: T | None = None
        self._has_item = False

    def put(self, item: T) -> None:
        with self._cv:
            self._item = item
            self._has_item = True
            self._cv.notify()

    def get(self, timeout: float) -> T | None:
        with self._cv:
            if not self._has_item:
                self._cv.wait(timeout)
            if not self._has_item:
                return None
            item = self._item
            self._item = None
            self._has_item = False
            return item

    def clear(self) -> None:
        with self._cv:
            self._item = None
            self._has_item = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_latest_queue.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/latest_queue.py tests/unit/test_latest_queue.py
git commit -m "feat(core): size-1 latest-wins queue for backpressure"
```

---

### Task 6: Rolling FPS meter

**Files:**
- Create: `src/smartuibot/core/fps.py`
- Test: `tests/unit/test_fps.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fps.py
from smartuibot.core.fps import FpsMeter


def test_fps_zero_before_two_ticks():
    m = FpsMeter(window=10)
    assert m.fps == 0.0
    m.tick(now=100.0)
    assert m.fps == 0.0


def test_fps_computed_over_window():
    m = FpsMeter(window=5)
    for i in range(5):
        m.tick(now=float(i) * 0.5)  # 2 fps spacing
    assert round(m.fps, 1) == 2.0


def test_window_drops_old_samples():
    m = FpsMeter(window=3)
    for i in range(10):
        m.tick(now=float(i))  # 1 fps spacing
    assert round(m.fps, 1) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_fps.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.fps`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/fps.py
from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Rolling FPS over the last `window` tick timestamps."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> None:
        self._times.append(time.monotonic() if now is None else now)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_fps.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/fps.py tests/unit/test_fps.py
git commit -m "feat(core): rolling FPS meter"
```

---

### Task 7: Config system (YAML → typed dataclasses, layered, validated)

**Files:**
- Create: `src/smartuibot/core/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
import textwrap

import pytest

from smartuibot.core.config import AppConfig, load_config


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_loads_defaults(tmp_path):
    default = _write(tmp_path, "d.yaml", """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {model: yolo11n.pt, confidence: 0.35, device: auto, tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """)
    cfg = load_config(default)
    assert isinstance(cfg, AppConfig)
    assert cfg.detection.confidence == 0.35
    assert cfg.capture.target_fps == 60


def test_user_overrides_merge_over_defaults(tmp_path):
    default = _write(tmp_path, "d.yaml", """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {model: yolo11n.pt, confidence: 0.35, device: auto, tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """)
    user = _write(tmp_path, "u.yaml", "detection: {confidence: 0.7}\n")
    cfg = load_config(default, user)
    assert cfg.detection.confidence == 0.7
    assert cfg.detection.model == "yolo11n.pt"  # untouched


def test_invalid_confidence_rejected(tmp_path):
    default = _write(tmp_path, "d.yaml", """
        capture: {backend: auto, target_fps: 60, monitor: 1}
        detection: {model: yolo11n.pt, confidence: 9.0, device: auto, tracking: false, smoothing_frames: 3}
        ui: {preview_max_width: 960}
        logging: {level: INFO, dir: logs}
        hotkeys: {emergency_stop: "<ctrl>+<alt>+q"}
    """)
    with pytest.raises(ValueError):
        load_config(default)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.config`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    backend: str
    target_fps: int
    monitor: int


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    model: str
    confidence: float
    device: str
    tracking: bool
    smoothing_frames: int


@dataclass(frozen=True, slots=True)
class UIConfig:
    preview_max_width: int


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str
    dir: str


@dataclass(frozen=True, slots=True)
class HotkeyConfig:
    emergency_stop: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    capture: CaptureConfig
    detection: DetectionConfig
    ui: UIConfig
    logging: LoggingConfig
    hotkeys: HotkeyConfig

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection.confidence <= 1.0:
            raise ValueError("detection.confidence must be in [0, 1]")
        if self.capture.target_fps <= 0:
            raise ValueError("capture.target_fps must be positive")
        if self.detection.smoothing_frames < 1:
            raise ValueError("detection.smoothing_frames must be >= 1")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(default_path: Path, user_path: Path | None = None) -> AppConfig:
    data: dict[str, Any] = yaml.safe_load(Path(default_path).read_text()) or {}
    if user_path is not None and Path(user_path).exists():
        user = yaml.safe_load(Path(user_path).read_text()) or {}
        data = _deep_merge(data, user)
    return AppConfig(
        capture=CaptureConfig(**data["capture"]),
        detection=DetectionConfig(**data["detection"]),
        ui=UIConfig(**data["ui"]),
        logging=LoggingConfig(**data["logging"]),
        hotkeys=HotkeyConfig(**data["hotkeys"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_config.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/config.py tests/unit/test_config.py
git commit -m "feat(core): layered validated YAML config"
```

---

### Task 8: Structured logging → ring buffer → bus

**Files:**
- Create: `src/smartuibot/core/logging_setup.py`
- Test: `tests/unit/test_logging_setup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_logging_setup.py
import logging

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import LogRecord
from smartuibot.core.logging_setup import setup_logging


def test_log_messages_are_published_as_events(tmp_path):
    bus = EventBus()
    seen: list[LogRecord] = []
    bus.subscribe(LogRecord, seen.append)
    setup_logging(level="INFO", log_dir=tmp_path, bus=bus)
    logging.getLogger("smartuibot.test").info("hello")
    assert any(r.message == "hello" and r.level == "INFO" for r in seen)


def test_rotating_file_is_written(tmp_path):
    bus = EventBus()
    setup_logging(level="DEBUG", log_dir=tmp_path, bus=bus)
    logging.getLogger("smartuibot.test").warning("disk-me")
    files = list(tmp_path.glob("*.log"))
    assert files and "disk-me" in files[0].read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_logging_setup.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.logging_setup`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/logging_setup.py
from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import LogRecord

_RESET = "\033[0m"
_COLORS = {"DEBUG": "\033[37m", "INFO": "\033[36m",
           "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[41m"}


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        base = f"{self.formatTime(record)} [{record.levelname}] {record.name}: {record.getMessage()}"
        return f"{color}{base}{_RESET}"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


class _BusHandler(logging.Handler):
    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        self._bus.publish(LogRecord(
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            ts=record.created,
        ))


def setup_logging(level: str, log_dir: Path, bus: EventBus) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("smartuibot")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(_ColorFormatter())
    root.addHandler(console)

    fname = log_dir / f"smartuibot-{time.strftime('%Y%m%d')}.log"
    fileh = RotatingFileHandler(fname, maxBytes=5_000_000, backupCount=5)
    fileh.setFormatter(_JsonFormatter())
    root.addHandler(fileh)

    root.addHandler(_BusHandler(bus))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_logging_setup.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/logging_setup.py tests/unit/test_logging_setup.py
git commit -m "feat(core): structured logging with bus + rotating file"
```

---

### Task 9: Service base (thread, heartbeat, exception boundary)

**Files:**
- Create: `src/smartuibot/core/service.py`
- Test: `tests/unit/test_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_service.py
import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ServiceError
from smartuibot.core.service import Service


class _Counter(Service):
    def __init__(self, bus: EventBus) -> None:
        super().__init__(name="counter", bus=bus)
        self.n = 0

    def run_once(self) -> None:
        self.n += 1
        time.sleep(0.01)


class _Boom(Service):
    def __init__(self, bus: EventBus) -> None:
        super().__init__(name="boom", bus=bus)

    def run_once(self) -> None:
        raise RuntimeError("explode")


def test_service_runs_loop_and_heartbeats():
    bus = EventBus()
    s = _Counter(bus)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert s.n > 1
    assert s.last_heartbeat > 0.0
    assert not s.is_alive


def test_service_exception_published_and_loop_stops():
    bus = EventBus()
    errs: list[ServiceError] = []
    bus.subscribe(ServiceError, errs.append)
    s = _Boom(bus)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert any(e.service == "boom" and "explode" in e.error for e in errs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_service.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.service`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/service.py
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ServiceError, StateChanged


class Service(ABC):
    """Base worker: owns a thread, updates a heartbeat each loop, and converts
    an unhandled exception into a fatal ServiceError then stops cleanly."""

    def __init__(self, name: str, bus: EventBus) -> None:
        self.name = name
        self._bus = bus
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.last_heartbeat: float = 0.0

    @abstractmethod
    def run_once(self) -> None:
        """One unit of work. Called repeatedly until stop()."""

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()
        self._bus.publish(StateChanged(service=self.name, state="running"))

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._bus.publish(StateChanged(service=self.name, state="stopped"))

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.02)
                continue
            try:
                self.run_once()
                self.last_heartbeat = time.monotonic()
            except Exception as exc:  # noqa: BLE001 - boundary
                self._bus.publish(ServiceError(service=self.name, error=repr(exc), fatal=True))
                return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_service.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/service.py tests/unit/test_service.py
git commit -m "feat(core): Service base with heartbeat + exception boundary"
```

---

### Task 10: Watchdog (heartbeat monitor + backoff restart)

**Files:**
- Create: `src/smartuibot/core/watchdog.py`
- Test: `tests/unit/test_watchdog.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_watchdog.py
import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.service import Service
from smartuibot.core.watchdog import Watchdog


class _Flaky(Service):
    def __init__(self, bus: EventBus) -> None:
        super().__init__(name="flaky", bus=bus)
        self.starts = 0
        self._fail_once = True

    def start(self) -> None:  # count restarts
        self.starts += 1
        super().start()

    def run_once(self) -> None:
        if self._fail_once:
            self._fail_once = False
            raise RuntimeError("first run dies")
        time.sleep(0.01)


def test_watchdog_restarts_crashed_service():
    bus = EventBus()
    s = _Flaky(bus)
    wd = Watchdog([s], bus=bus, check_interval=0.02, base_backoff=0.01)
    s.start()
    wd.start()
    time.sleep(0.3)
    wd.stop()
    s.stop()
    assert s.starts >= 2  # restarted at least once
    assert s.is_alive or s.starts >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_watchdog.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.watchdog`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/watchdog.py
from __future__ import annotations

import logging
import threading
import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ServiceError
from smartuibot.core.service import Service

_log = logging.getLogger("smartuibot.watchdog")


class Watchdog:
    """Restarts a Service whose thread has died, with exponential backoff."""

    def __init__(
        self,
        services: list[Service],
        bus: EventBus,
        check_interval: float = 1.0,
        base_backoff: float = 0.5,
        max_retries: int = 5,
    ) -> None:
        self._services = services
        self._bus = bus
        self._interval = check_interval
        self._base = base_backoff
        self._max_retries = max_retries
        self._retries: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            for svc in self._services:
                if svc.is_alive:
                    self._retries[svc.name] = 0
                    continue
                n = self._retries.get(svc.name, 0)
                if n >= self._max_retries:
                    self._bus.publish(ServiceError(
                        service=svc.name, error="max restarts exceeded", fatal=True))
                    continue
                self._retries[svc.name] = n + 1
                backoff = self._base * (2 ** n)
                _log.warning("restarting %s (attempt %d) after %.2fs", svc.name, n + 1, backoff)
                time.sleep(backoff)
                svc.start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_watchdog.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/watchdog.py tests/unit/test_watchdog.py
git commit -m "feat(core): watchdog with exponential-backoff restart"
```

---

### Task 11: Platform detection + capture-backend selection

**Files:**
- Create: `src/smartuibot/platform_support/detect.py`
- Test: `tests/unit/test_platform.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_platform.py
from smartuibot.platform_support.detect import current_os, resolve_backend_name


def test_current_os_known_value():
    assert current_os() in {"windows", "macos", "linux"}


def test_auto_resolves_to_mss_off_windows():
    assert resolve_backend_name("auto", os_name="macos") == "mss"
    assert resolve_backend_name("auto", os_name="linux") == "mss"
    assert resolve_backend_name("auto", os_name="windows") == "dxcam"


def test_explicit_backend_is_respected():
    assert resolve_backend_name("mss", os_name="windows") == "mss"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_platform.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.platform_support.detect`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/platform_support/detect.py
from __future__ import annotations

import sys


def current_os() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def resolve_backend_name(configured: str, os_name: str | None = None) -> str:
    os_name = os_name or current_os()
    if configured != "auto":
        return configured
    return "dxcam" if os_name == "windows" else "mss"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_platform.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/platform_support/detect.py tests/unit/test_platform.py
git commit -m "feat(platform): OS detection + backend selection"
```

---

### Task 12: CaptureBackend Protocol + FakeCaptureBackend

**Files:**
- Create: `src/smartuibot/vision/capture/backend.py`
- Create: `tests/fakes/capture.py`
- Test: `tests/unit/test_fake_capture.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fake_capture.py
from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import CaptureBackend
from tests.fakes.capture import FakeCaptureBackend


def test_fake_backend_satisfies_protocol():
    fake: CaptureBackend = FakeCaptureBackend(width=20, height=10)
    assert fake.list_monitors()[0].index == 1


def test_fake_grab_returns_roi_sized_bgr_image():
    fake = FakeCaptureBackend(width=20, height=10)
    img = fake.grab(ROI(monitor=1, x=0, y=0, width=8, height=6))
    assert img.shape == (6, 8, 3)
    assert img.dtype.name == "uint8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_fake_capture.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.vision.capture.backend`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/vision/capture/backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from smartuibot.core.types import ROI


@dataclass(frozen=True, slots=True)
class Monitor:
    index: int
    x: int
    y: int
    width: int
    height: int


@runtime_checkable
class CaptureBackend(Protocol):
    def list_monitors(self) -> list[Monitor]: ...

    def grab(self, roi: ROI) -> np.ndarray:
        """Return an HxWx3 BGR uint8 array for the ROI."""
```

```python
# tests/fakes/capture.py
from __future__ import annotations

import numpy as np

from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import Monitor


class FakeCaptureBackend:
    """Deterministic synthetic frames; lets the pipeline run headless."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._w = width
        self._h = height
        self._seq = 0

    def list_monitors(self) -> list[Monitor]:
        return [Monitor(index=1, x=0, y=0, width=self._w, height=self._h)]

    def grab(self, roi: ROI) -> np.ndarray:
        self._seq += 1
        img = np.full((roi.height, roi.width, 3), self._seq % 256, dtype=np.uint8)
        return img
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_fake_capture.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/vision/capture/backend.py tests/fakes/capture.py tests/unit/test_fake_capture.py
git commit -m "feat(capture): CaptureBackend protocol + fake backend"
```

---

### Task 13: MssBackend (real macOS/universal capture)

**Files:**
- Create: `src/smartuibot/vision/capture/mss_backend.py`
- Test: `tests/unit/test_mss_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_mss_backend.py
import numpy as np
import pytest

from smartuibot.core.types import ROI
from smartuibot.vision.capture.mss_backend import MssBackend


def test_bgra_to_bgr_conversion_shape():
    # Exercise the pure conversion helper without needing a real screen.
    bgra = np.zeros((4, 5, 4), dtype=np.uint8)
    bgr = MssBackend._to_bgr(bgra)
    assert bgr.shape == (4, 5, 3)


@pytest.mark.skipif(
    __import__("os").environ.get("CI") == "true",
    reason="real screen capture not available in CI",
)
def test_real_grab_returns_requested_size():
    be = MssBackend()
    mons = be.list_monitors()
    assert mons
    img = be.grab(ROI(monitor=mons[0].index, x=0, y=0, width=16, height=12))
    assert img.shape == (12, 16, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_mss_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.vision.capture.mss_backend`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/vision/capture/mss_backend.py
from __future__ import annotations

import numpy as np

from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import Monitor


class MssBackend:
    """Screen capture via `mss`. Works on macOS (Intel/ARM), Windows, Linux."""

    def __init__(self) -> None:
        import mss

        self._sct = mss.mss()

    def list_monitors(self) -> list[Monitor]:
        # mss.monitors[0] is the virtual "all monitors" rect; real ones start at 1.
        out: list[Monitor] = []
        for i, m in enumerate(self._sct.monitors[1:], start=1):
            out.append(Monitor(index=i, x=m["left"], y=m["top"],
                                width=m["width"], height=m["height"]))
        return out

    @staticmethod
    def _to_bgr(bgra: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(bgra[:, :, :3])

    def grab(self, roi: ROI) -> np.ndarray:
        mon = self._sct.monitors[roi.monitor]
        region = {
            "left": mon["left"] + roi.x,
            "top": mon["top"] + roi.y,
            "width": roi.width,
            "height": roi.height,
        }
        shot = self._sct.grab(region)
        bgra = np.asarray(shot, dtype=np.uint8)  # mss returns BGRA
        return self._to_bgr(bgra)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_mss_backend.py -q`
Expected: PASS (1 passed, 1 may be skipped if no screen / CI)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/vision/capture/mss_backend.py tests/unit/test_mss_backend.py
git commit -m "feat(capture): mss backend with BGRA->BGR conversion"
```

---

### Task 14: CaptureService (worker thread → FrameCaptured)

**Files:**
- Create: `src/smartuibot/vision/capture/service.py`
- Test: `tests/unit/test_capture_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_capture_service.py
import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import FpsTick, FrameCaptured
from smartuibot.core.types import ROI
from smartuibot.vision.capture.service import CaptureService
from tests.fakes.capture import FakeCaptureBackend


def test_capture_service_publishes_frames_and_fps():
    bus = EventBus()
    frames: list[FrameCaptured] = []
    fps: list[FpsTick] = []
    bus.subscribe(FrameCaptured, frames.append)
    bus.subscribe(FpsTick, lambda e: fps.append(e) if e.name == "capture" else None)
    svc = CaptureService(
        backend=FakeCaptureBackend(),
        bus=bus,
        roi=ROI(monitor=1, x=0, y=0, width=32, height=24),
        target_fps=120,
    )
    svc.start()
    time.sleep(0.2)
    svc.stop()
    assert len(frames) > 2
    assert frames[0].frame.seq < frames[-1].frame.seq
    assert any(f.fps > 0 for f in fps)


def test_set_roi_changes_frame_size_without_restart():
    bus = EventBus()
    frames: list[FrameCaptured] = []
    bus.subscribe(FrameCaptured, frames.append)
    svc = CaptureService(FakeCaptureBackend(), bus,
                          ROI(monitor=1, x=0, y=0, width=10, height=10), target_fps=120)
    svc.start()
    time.sleep(0.05)
    svc.set_roi(ROI(monitor=1, x=0, y=0, width=20, height=15))
    time.sleep(0.1)
    svc.stop()
    assert frames[-1].frame.image.shape == (15, 20, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_capture_service.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.vision.capture.service`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/vision/capture/service.py
from __future__ import annotations

import threading
import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import FpsTick, FrameCaptured
from smartuibot.core.fps import FpsMeter
from smartuibot.core.service import Service
from smartuibot.core.types import ROI, Frame
from smartuibot.vision.capture.backend import CaptureBackend


class CaptureService(Service):
    def __init__(
        self,
        backend: CaptureBackend,
        bus: EventBus,
        roi: ROI,
        target_fps: int = 60,
    ) -> None:
        super().__init__(name="capture", bus=bus)
        self._backend = backend
        self._roi = roi
        self._roi_lock = threading.Lock()
        self._period = 1.0 / float(target_fps)
        self._seq = 0
        self._fps = FpsMeter(window=30)

    def set_roi(self, roi: ROI) -> None:
        with self._roi_lock:
            self._roi = roi

    def run_once(self) -> None:
        start = time.monotonic()
        with self._roi_lock:
            roi = self._roi
        image = self._backend.grab(roi)
        self._seq += 1
        frame = Frame(image=image, timestamp=time.monotonic(), seq=self._seq, roi=roi)
        self._bus.publish(FrameCaptured(frame=frame))
        self._fps.tick()
        self._bus.publish(FpsTick(name="capture", fps=self._fps.fps))
        elapsed = time.monotonic() - start
        if elapsed < self._period:
            time.sleep(self._period - elapsed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_capture_service.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/vision/capture/service.py tests/unit/test_capture_service.py
git commit -m "feat(capture): CaptureService worker with live ROI + FPS"
```

---

### Task 15: Detector Protocol + FakeDetector

**Files:**
- Create: `src/smartuibot/vision/detect/detector.py`
- Create: `tests/fakes/detector.py`
- Test: `tests/unit/test_fake_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fake_detector.py
import numpy as np

from smartuibot.vision.detect.detector import Detector
from tests.fakes.detector import FakeDetector


def test_fake_detector_satisfies_protocol_and_returns_scripted():
    det: Detector = FakeDetector(scripted=[
        [("enemy", 0.9, 0, 0, 10, 10)],
        [],
    ])
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    first = det.infer(img)
    assert first[0].label == "enemy" and first[0].confidence == 0.9
    assert det.infer(img) == []
    det.reload("ignored.pt")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_fake_detector.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.vision.detect.detector`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/vision/detect/detector.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from smartuibot.core.types import Detection


@runtime_checkable
class Detector(Protocol):
    def infer(self, image: np.ndarray) -> list[Detection]: ...

    def reload(self, model_path: str) -> None: ...
```

```python
# tests/fakes/detector.py
from __future__ import annotations

import numpy as np

from smartuibot.core.types import Detection

_Script = list[tuple[str, float, int, int, int, int]]


class FakeDetector:
    """Returns scripted detections per call, then []. Deterministic."""

    def __init__(self, scripted: list[_Script]) -> None:
        self._scripted = scripted
        self._i = 0

    def infer(self, image: np.ndarray) -> list[Detection]:
        if self._i >= len(self._scripted):
            return []
        spec = self._scripted[self._i]
        self._i += 1
        return [
            Detection(label=lbl, confidence=conf, class_id=0,
                      x1=x1, y1=y1, x2=x2, y2=y2)
            for (lbl, conf, x1, y1, x2, y2) in spec
        ]

    def reload(self, model_path: str) -> None:
        self._i = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_fake_detector.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/vision/detect/detector.py tests/fakes/detector.py tests/unit/test_fake_detector.py
git commit -m "feat(detect): Detector protocol + fake detector"
```

---

### Task 16: SmoothingFilter (temporal persistence)

**Files:**
- Create: `src/smartuibot/vision/detect/smoothing.py`
- Test: `tests/unit/test_smoothing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_smoothing.py
from smartuibot.core.types import Detection
from smartuibot.vision.detect.smoothing import SmoothingFilter


def _d(label: str) -> Detection:
    return Detection(label=label, confidence=0.9, class_id=0, x1=0, y1=0, x2=5, y2=5)


def test_detection_persists_for_n_frames_after_disappearing():
    f = SmoothingFilter(persist_frames=2)
    assert [d.label for d in f.update([_d("a")])] == ["a"]
    assert [d.label for d in f.update([])] == ["a"]      # frame 1 missing -> kept
    assert [d.label for d in f.update([])] == ["a"]      # frame 2 missing -> kept
    assert [d.label for d in f.update([])] == []         # frame 3 missing -> dropped


def test_reappearing_detection_resets_persistence():
    f = SmoothingFilter(persist_frames=1)
    f.update([_d("a")])
    f.update([])
    assert [d.label for d in f.update([_d("a")])] == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_smoothing.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.vision.detect.smoothing`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/vision/detect/smoothing.py
from __future__ import annotations

from smartuibot.core.types import Detection


class SmoothingFilter:
    """Keeps a detection visible for `persist_frames` extra frames after it
    stops being reported, reducing flicker."""

    def __init__(self, persist_frames: int = 3) -> None:
        self._persist = persist_frames
        self._ages: dict[str, int] = {}
        self._last: dict[str, Detection] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        present = {d.label: d for d in detections}
        for label, det in present.items():
            self._ages[label] = 0
            self._last[label] = det

        out: list[Detection] = []
        for label in list(self._ages.keys()):
            if label in present:
                out.append(present[label])
                continue
            self._ages[label] += 1
            if self._ages[label] > self._persist:
                del self._ages[label]
                del self._last[label]
            else:
                out.append(self._last[label])
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_smoothing.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/vision/detect/smoothing.py tests/unit/test_smoothing.py
git commit -m "feat(detect): temporal smoothing filter"
```

---

### Task 17: Yolo11Detector adapter (marked/skippable model test)

**Files:**
- Create: `src/smartuibot/vision/detect/yolo.py`
- Test: `tests/unit/test_yolo_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_yolo_detector.py
import numpy as np
import pytest

from smartuibot.core.types import Detection
from smartuibot.vision.detect.yolo import Yolo11Detector, _results_to_detections


def test_results_to_detections_pure_mapping():
    class _Box:
        def __init__(self):
            self.xyxy = [np.array([1.0, 2.0, 11.0, 12.0])]
            self.conf = [np.array(0.81)]
            self.cls = [np.array(0.0)]

    class _Res:
        names = {0: "person"}
        boxes = [_Box()]

    dets = _results_to_detections([_Res()], conf_threshold=0.5)
    assert dets == [Detection(label="person", confidence=pytest.approx(0.81),
                              class_id=0, x1=1, y1=2, x2=11, y2=12)]


def test_results_to_detections_filters_low_confidence():
    class _Box:
        def __init__(self):
            self.xyxy = [np.array([0.0, 0.0, 1.0, 1.0])]
            self.conf = [np.array(0.10)]
            self.cls = [np.array(0.0)]

    class _Res:
        names = {0: "person"}
        boxes = [_Box()]

    assert _results_to_detections([_Res()], conf_threshold=0.5) == []


@pytest.mark.model
def test_real_yolo_infers_schema():
    det = Yolo11Detector(model_path="yolo11n.pt", device="cpu", confidence=0.25)
    out = det.infer(np.zeros((320, 320, 3), dtype=np.uint8))
    assert isinstance(out, list)
    for d in out:
        assert isinstance(d, Detection)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_yolo_detector.py -q -m "not model"`
Expected: FAIL — `ModuleNotFoundError: smartuibot.vision.detect.yolo`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/vision/detect/yolo.py
from __future__ import annotations

from typing import Any

import numpy as np

from smartuibot.core.types import Detection


def _results_to_detections(results: list[Any], conf_threshold: float) -> list[Detection]:
    out: list[Detection] = []
    for res in results:
        names = res.names
        for box in res.boxes:
            conf = float(np.asarray(box.conf[0]))
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in np.asarray(box.xyxy[0]))
            cls_id = int(np.asarray(box.cls[0]))
            out.append(Detection(
                label=str(names[cls_id]),
                confidence=conf,
                class_id=cls_id,
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            ))
    return out


class Yolo11Detector:
    def __init__(self, model_path: str, device: str = "auto",
                 confidence: float = 0.35) -> None:
        from ultralytics import YOLO

        self._confidence = confidence
        self._device = self._resolve_device(device)
        self._model = YOLO(model_path)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def infer(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            image, device=self._device, conf=self._confidence, verbose=False)
        return _results_to_detections(results, self._confidence)

    def reload(self, model_path: str) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_yolo_detector.py -q -m "not model"`
Expected: PASS (2 passed, model test deselected)

Optionally, with network: `.venv/bin/pytest tests/unit/test_yolo_detector.py -q -m model`
Expected: PASS (downloads `yolo11n.pt`)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/vision/detect/yolo.py tests/unit/test_yolo_detector.py
git commit -m "feat(detect): YOLO11 adapter with pure result mapping"
```

---

### Task 18: DetectionService (latest frame → DetectionsReady)

**Files:**
- Create: `src/smartuibot/vision/detect/service.py`
- Test: `tests/unit/test_detection_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_detection_service.py
import time

import numpy as np

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsReady, FrameCaptured
from smartuibot.core.types import ROI, Frame
from smartuibot.vision.detect.service import DetectionService
from tests.fakes.detector import FakeDetector


def _frame(seq: int) -> Frame:
    return Frame(image=np.zeros((8, 8, 3), dtype=np.uint8),
                 timestamp=time.monotonic(), seq=seq,
                 roi=ROI(monitor=1, x=0, y=0, width=8, height=8))


def test_detection_service_consumes_frames_and_publishes():
    bus = EventBus()
    out: list[DetectionsReady] = []
    bus.subscribe(DetectionsReady, out.append)
    det = FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 4, 4)]] * 3)
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1)
    svc.start()
    bus.publish(FrameCaptured(frame=_frame(1)))
    time.sleep(0.2)
    svc.stop()
    assert out
    assert out[0].detections[0].label == "enemy"


def test_only_latest_frame_is_processed_under_backlog():
    bus = EventBus()
    out: list[DetectionsReady] = []
    bus.subscribe(DetectionsReady, out.append)
    det = FakeDetector(scripted=[[("x", 0.9, 0, 0, 1, 1)]] * 50)
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1)
    for s in range(10):
        bus.publish(FrameCaptured(frame=_frame(s)))  # before start: only last kept
    svc.start()
    time.sleep(0.15)
    svc.stop()
    # Latest-wins queue means we did NOT emit 10 results for the backlog.
    assert 1 <= len(out) <= 3
    assert out[-1].frame.seq == 9


def test_runtime_confidence_filters_low_confidence_detections():
    bus = EventBus()
    out: list[DetectionsReady] = []
    bus.subscribe(DetectionsReady, out.append)
    det = FakeDetector(scripted=[[("hi", 0.9, 0, 0, 1, 1), ("lo", 0.2, 0, 0, 1, 1)]] * 5)
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1, confidence=0.5)
    svc.start()
    bus.publish(FrameCaptured(frame=_frame(1)))
    time.sleep(0.15)
    svc.stop()
    labels = {d.label for d in out[0].detections}
    assert labels == {"hi"}  # 0.2 < 0.5 threshold filtered out


def test_set_confidence_and_reload_model_are_thread_safe_noops_on_fake():
    bus = EventBus()
    det = FakeDetector(scripted=[[("x", 0.9, 0, 0, 1, 1)]])
    svc = DetectionService(detector=det, bus=bus, smoothing_frames=1, confidence=0.3)
    svc.set_confidence(0.95)        # must not raise
    svc.reload_model("ignored.pt")  # delegates to detector.reload
    assert svc.confidence == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_detection_service.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.vision.detect.service`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/vision/detect/service.py
from __future__ import annotations

import threading

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsReady, FpsTick, FrameCaptured
from smartuibot.core.fps import FpsMeter
from smartuibot.core.latest_queue import LatestQueue
from smartuibot.core.service import Service
from smartuibot.core.types import Frame
from smartuibot.vision.detect.detector import Detector
from smartuibot.vision.detect.smoothing import SmoothingFilter


class DetectionService(Service):
    def __init__(
        self,
        detector: Detector,
        bus: EventBus,
        smoothing_frames: int = 3,
        confidence: float = 0.0,
    ) -> None:
        super().__init__(name="detection", bus=bus)
        self._detector = detector
        self._queue: LatestQueue[Frame] = LatestQueue()
        self._smoothing = SmoothingFilter(persist_frames=smoothing_frames)
        self._fps = FpsMeter(window=30)
        self._lock = threading.Lock()
        self._confidence = confidence
        bus.subscribe(FrameCaptured, self._on_frame)

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._confidence

    def set_confidence(self, value: float) -> None:
        """Runtime-adjustable post-inference threshold (called from UI thread)."""
        with self._lock:
            self._confidence = max(0.0, min(1.0, value))

    def reload_model(self, model_path: str) -> None:
        """Hot-reload the detector model (called from UI thread)."""
        self._detector.reload(model_path)

    def _on_frame(self, event: FrameCaptured) -> None:
        self._queue.put(event.frame)

    def run_once(self) -> None:
        frame = self._queue.get(timeout=0.1)
        if frame is None:
            return
        threshold = self.confidence
        raw = [d for d in self._detector.infer(frame.image) if d.confidence >= threshold]
        smoothed = self._smoothing.update(raw)
        self._bus.publish(DetectionsReady(frame=frame, detections=tuple(smoothed)))
        self._fps.tick()
        self._bus.publish(FpsTick(name="detection", fps=self._fps.fps))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_detection_service.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/vision/detect/service.py tests/unit/test_detection_service.py
git commit -m "feat(detect): DetectionService with latest-wins backpressure"
```

---

### Task 19: AppContainer (DI composition root)

**Files:**
- Create: `src/smartuibot/core/container.py`
- Test: `tests/unit/test_container.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_container.py
import time

from smartuibot.core.config import load_config
from smartuibot.core.container import AppContainer
from smartuibot.core.events import DetectionsReady
from smartuibot.core.types import ROI
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector


def test_container_wires_pipeline_with_injected_fakes(tmp_path):
    default = tmp_path / "d.yaml"
    default.write_text(
        "capture: {backend: auto, target_fps: 120, monitor: 1}\n"
        "detection: {model: yolo11n.pt, confidence: 0.3, device: cpu, tracking: false, smoothing_frames: 1}\n"
        "ui: {preview_max_width: 960}\n"
        "logging: {level: INFO, dir: " + str(tmp_path / 'logs') + "}\n"
        "hotkeys: {emergency_stop: \"<ctrl>+<alt>+q\"}\n"
    )
    cfg = load_config(default)
    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=16, height=16),
        capture_backend=FakeCaptureBackend(),
        detector=FakeDetector(scripted=[[("enemy", 0.9, 0, 0, 4, 4)]] * 40),
    )
    out: list[DetectionsReady] = []
    container.bus.subscribe(DetectionsReady, out.append)
    container.start()
    time.sleep(0.3)
    container.stop()
    assert out, "expected detections to flow capture->detect->bus"
    assert out[-1].detections[0].label == "enemy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_container.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.core.container`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/core/container.py
from __future__ import annotations

from pathlib import Path

from smartuibot.core.config import AppConfig
from smartuibot.core.event_bus import EventBus
from smartuibot.core.logging_setup import setup_logging
from smartuibot.core.types import ROI
from smartuibot.core.watchdog import Watchdog
from smartuibot.vision.capture.backend import CaptureBackend
from smartuibot.vision.capture.service import CaptureService
from smartuibot.vision.detect.detector import Detector
from smartuibot.vision.detect.service import DetectionService


class AppContainer:
    """Owns and wires every singleton. Platform adapters are injected so the
    whole pipeline can run headless with fakes."""

    def __init__(
        self,
        config: AppConfig,
        roi: ROI,
        capture_backend: CaptureBackend,
        detector: Detector,
    ) -> None:
        self.config = config
        self.bus = EventBus()
        setup_logging(level=config.logging.level,
                      log_dir=Path(config.logging.dir), bus=self.bus)
        self.capture = CaptureService(
            backend=capture_backend, bus=self.bus, roi=roi,
            target_fps=config.capture.target_fps)
        self.detection = DetectionService(
            detector=detector, bus=self.bus,
            smoothing_frames=config.detection.smoothing_frames,
            confidence=config.detection.confidence)
        self.watchdog = Watchdog([self.capture, self.detection], bus=self.bus)

    def start(self) -> None:
        self.detection.start()
        self.capture.start()
        self.watchdog.start()

    def stop(self) -> None:
        self.watchdog.stop()
        self.capture.stop()
        self.detection.stop()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_container.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/core/container.py tests/unit/test_container.py
git commit -m "feat(core): AppContainer DI composition root"
```

---

### Task 20: End-to-end integration test (headless, with fakes)

**Files:**
- Create: `tests/integration/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_pipeline.py
import time

from smartuibot.core.config import load_config
from smartuibot.core.container import AppContainer
from smartuibot.core.events import DetectionsReady, ServiceError
from smartuibot.core.types import ROI
from tests.fakes.capture import FakeCaptureBackend
from tests.fakes.detector import FakeDetector


def _cfg(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(
        "capture: {backend: auto, target_fps: 90, monitor: 1}\n"
        "detection: {model: yolo11n.pt, confidence: 0.3, device: cpu, tracking: false, smoothing_frames: 2}\n"
        "ui: {preview_max_width: 960}\n"
        "logging: {level: INFO, dir: " + str(tmp_path / 'logs') + "}\n"
        "hotkeys: {emergency_stop: \"<ctrl>+<alt>+q\"}\n"
    )
    return load_config(p)


def test_frames_flow_and_stale_frames_are_dropped(tmp_path):
    cfg = _cfg(tmp_path)

    class _SlowDetector(FakeDetector):
        def infer(self, image):
            time.sleep(0.05)  # detector slower than capture
            return super().infer(image)

    container = AppContainer(
        config=cfg,
        roi=ROI(monitor=1, x=0, y=0, width=24, height=24),
        capture_backend=FakeCaptureBackend(),
        detector=_SlowDetector(scripted=[[("e", 0.9, 0, 0, 2, 2)]] * 200),
    )
    errors: list[ServiceError] = []
    results: list[DetectionsReady] = []
    container.bus.subscribe(ServiceError, errors.append)
    container.bus.subscribe(DetectionsReady, results.append)

    container.start()
    time.sleep(1.0)
    container.stop()

    assert not errors, f"unexpected service errors: {errors}"
    assert results, "no detections produced"
    # Detector (~20 fps) far slower than capture (~90 fps): far fewer results
    # than frames, proving stale frames were dropped, not buffered.
    assert len(results) < 60
    seqs = [r.frame.seq for r in results]
    assert seqs == sorted(seqs), "results must be in monotonically increasing seq order"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_pipeline.py -q`
Expected: FAIL initially only if any upstream task incomplete; otherwise it should PASS once Tasks 1–19 are done.

- [ ] **Step 3: No new implementation**

This test exercises already-built components. If it fails, debug the offending component using superpowers:systematic-debugging — do not weaken the assertions.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_pipeline.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_pipeline.py
git commit -m "test(integration): headless pipeline + stale-frame drop"
```

---

### Task 21: ROISelectorOverlay (frameless translucent drag-select)

**Files:**
- Create: `src/smartuibot/ui/roi_selector.py`
- Test: `tests/unit/test_roi_selector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_roi_selector.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint  # noqa: E402

from smartuibot.core.types import ROI  # noqa: E402
from smartuibot.ui.roi_selector import rect_to_roi  # noqa: E402


def test_rect_to_roi_normalizes_drag_direction():
    # drag from bottom-right to top-left still yields a positive ROI
    roi = rect_to_roi(QPoint(100, 90), QPoint(20, 10), monitor=1)
    assert roi == ROI(monitor=1, x=20, y=10, width=80, height=80)


def test_rect_to_roi_minimum_size_enforced():
    roi = rect_to_roi(QPoint(5, 5), QPoint(6, 6), monitor=2)
    assert roi.width >= 1 and roi.height >= 1 and roi.monitor == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_roi_selector.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ui.roi_selector`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/ui/roi_selector.py
from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QWidget

from smartuibot.core.types import ROI


def rect_to_roi(p1: QPoint, p2: QPoint, monitor: int) -> ROI:
    x = min(p1.x(), p2.x())
    y = min(p1.y(), p2.y())
    w = max(1, abs(p1.x() - p2.x()))
    h = max(1, abs(p1.y() - p2.y()))
    return ROI(monitor=monitor, x=x, y=y, width=w, height=h)


class ROISelectorOverlay(QWidget):
    """Fullscreen translucent overlay; drag a rectangle, release to confirm."""

    def __init__(self, monitor: int, on_selected: Callable[[ROI], None]) -> None:
        super().__init__()
        self._monitor = monitor
        self._on_selected = on_selected
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.35)

    # NOTE: PyQt6 stubs type these handler params as `... | None` and mypy
    # --strict enforces override compatibility, so annotate with `| None` and
    # guard. Use event.position().toPoint() (Qt6 API; pos() is deprecated).
    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        self._origin = event.position().toPoint()
        self._current = event.position().toPoint()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        self._current = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self._origin is not None:
            roi = rect_to_roi(self._origin, event.position().toPoint(), self._monitor)
            self._on_selected(roi)
        self.close()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if self._origin is None or self._current is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 200, 0), 2))
        painter.drawRect(QRect(self._origin, self._current))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_roi_selector.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/ui/roi_selector.py tests/unit/test_roi_selector.py
git commit -m "feat(ui): translucent ROI selector overlay + rect_to_roi"
```

---

### Task 22: DebugWindow (preview/boxes/table/FPS/logs/controls)

**Files:**
- Create: `src/smartuibot/ui/debug_window.py`
- Test: `tests/unit/test_debug_window.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_debug_window.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from smartuibot.core.event_bus import EventBus  # noqa: E402
from smartuibot.core.events import DetectionsReady, FpsTick, LogRecord  # noqa: E402
from smartuibot.core.types import ROI, Detection, Frame  # noqa: E402
from smartuibot.ui.debug_window import DebugWindow, draw_boxes  # noqa: E402


def test_draw_boxes_does_not_mutate_input_and_keeps_shape():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    dets = [Detection(label="e", confidence=0.9, class_id=0, x1=2, y1=2, x2=20, y2=20)]
    out = draw_boxes(img, dets)
    assert out.shape == img.shape
    assert img.sum() == 0  # original untouched (copy made)
    assert out.sum() > 0   # something drawn


def test_debug_window_consumes_bus_events_without_error():
    app = QApplication.instance() or QApplication([])
    bus = EventBus()
    win = DebugWindow(bus=bus, preview_max_width=320)
    frame = Frame(image=np.zeros((30, 40, 3), dtype=np.uint8),
                  timestamp=0.0, seq=1, roi=ROI(monitor=1, x=0, y=0, width=40, height=30))
    bus.publish(DetectionsReady(frame=frame, detections=(
        Detection(label="enemy", confidence=0.8, class_id=0, x1=1, y1=1, x2=5, y2=5),)))
    bus.publish(FpsTick(name="capture", fps=55.0))
    bus.publish(FpsTick(name="detection", fps=7.0))
    bus.publish(LogRecord(level="INFO", logger="t", message="hi", ts=0.0))
    win._drain()  # process queued events synchronously
    assert win.detection_count() == 1
    assert "55.0" in win.fps_text()
    win.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_debug_window.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ui.debug_window`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/ui/debug_window.py
from __future__ import annotations

import queue

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QPlainTextEdit, QVBoxLayout, QWidget,
)

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsReady, FpsTick, LogRecord
from smartuibot.core.types import Detection


def draw_boxes(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image.copy()
    for d in detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (d.x1, max(0, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


class DebugWindow(QWidget):
    """Read-only debug view. Events arrive on worker threads; they are queued
    and drained on the Qt thread via a timer (Qt rule: GUI on main thread)."""

    def __init__(self, bus: EventBus, preview_max_width: int = 960) -> None:
        super().__init__()
        self.setWindowTitle("SmartUIBot — Debug")
        self._max_w = preview_max_width
        self._events: queue.Queue = queue.Queue()
        self._fps = {"capture": 0.0, "detection": 0.0}
        self._det_count = 0

        self._preview = QLabel("waiting for frames…")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fps_label = QLabel(self.fps_text())
        self._table = QListWidget()
        self._logs = QPlainTextEdit()
        self._logs.setReadOnly(True)

        right = QVBoxLayout()
        right.addWidget(self._fps_label)
        right.addWidget(QLabel("Detections:"))
        right.addWidget(self._table)
        right.addWidget(QLabel("Logs:"))
        right.addWidget(self._logs)
        root = QHBoxLayout(self)
        root.addWidget(self._preview, stretch=3)
        right_box = QWidget()
        right_box.setLayout(right)
        root.addWidget(right_box, stretch=2)

        bus.subscribe(DetectionsReady, self._events.put)
        bus.subscribe(FpsTick, self._events.put)
        bus.subscribe(LogRecord, self._events.put)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain)
        self._timer.start(33)  # ~30 Hz UI refresh

    # --- introspection helpers (used by tests) ---
    def detection_count(self) -> int:
        return self._det_count

    def fps_text(self) -> str:
        return f"capture {self._fps['capture']:.1f} fps | detection {self._fps['detection']:.1f} fps"

    # --- event handling on the Qt thread ---
    def _drain(self) -> None:
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            if isinstance(ev, DetectionsReady):
                self._on_detections(ev)
            elif isinstance(ev, FpsTick):
                self._fps[ev.name] = ev.fps
                self._fps_label.setText(self.fps_text())
            elif isinstance(ev, LogRecord):
                self._logs.appendPlainText(f"[{ev.level}] {ev.logger}: {ev.message}")

    def _on_detections(self, ev: DetectionsReady) -> None:
        self._det_count = len(ev.detections)
        self._table.clear()
        for d in ev.detections:
            self._table.addItem(f"{d.label}  {d.confidence:.2f}  "
                                f"[{d.x1},{d.y1},{d.x2},{d.y2}]")
        rendered = draw_boxes(ev.frame.image, list(ev.detections))
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if w > self._max_w:
            pix = pix.scaledToWidth(self._max_w, Qt.TransformationMode.SmoothTransformation)
        self._preview.setPixmap(pix)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_debug_window.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/ui/debug_window.py tests/unit/test_debug_window.py
git commit -m "feat(ui): debug window with preview/table/fps/logs"
```

---

### Task 23: Composition root `app.py` + `run.py` + emergency-stop

**Files:**
- Create: `src/smartuibot/app.py`
- Create: `run.py`
- Test: `tests/unit/test_app_factory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_app_factory.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

from smartuibot.app import build_real_container, load_or_default_roi  # noqa: E402
from smartuibot.core.types import ROI  # noqa: E402


def test_load_or_default_roi_uses_state_when_present(tmp_path):
    state = tmp_path / "state.yaml"
    state.write_text("roi: {monitor: 1, x: 5, y: 6, width: 100, height: 80}\n")
    assert load_or_default_roi(state) == ROI(monitor=1, x=5, y=6, width=100, height=80)


def test_load_or_default_roi_falls_back_when_missing(tmp_path):
    roi = load_or_default_roi(tmp_path / "absent.yaml")
    assert isinstance(roi, ROI) and roi.width > 0 and roi.height > 0


def test_build_real_container_constructs_without_starting(tmp_path, monkeypatch):
    # Avoid importing torch/ultralytics: stub the YOLO detector factory.
    import smartuibot.app as app_mod

    class _StubDetector:
        def infer(self, image): return []
        def reload(self, p): ...

    monkeypatch.setattr(app_mod, "_make_detector", lambda cfg: _StubDetector())
    monkeypatch.setattr(app_mod, "_make_capture_backend", lambda cfg: __import__(
        "tests.fakes.capture", fromlist=["FakeCaptureBackend"]).FakeCaptureBackend())
    cfg_path = Path("configs/default.yaml")
    container = build_real_container(cfg_path, state_path=tmp_path / "state.yaml")
    assert container.config.detection.model.endswith(".pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_app_factory.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.app`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/app.py
from __future__ import annotations

import signal
import sys
from pathlib import Path

import yaml
from PyQt6.QtWidgets import QApplication

from smartuibot.core.config import AppConfig, load_config
from smartuibot.core.container import AppContainer
from smartuibot.core.types import ROI
from smartuibot.platform_support.detect import resolve_backend_name
from smartuibot.ui.debug_window import DebugWindow

_DEFAULT_ROI = ROI(monitor=1, x=100, y=100, width=640, height=480)


def load_or_default_roi(state_path: Path) -> ROI:
    if Path(state_path).exists():
        data = yaml.safe_load(Path(state_path).read_text()) or {}
        if "roi" in data:
            return ROI.from_dict(data["roi"])
    return _DEFAULT_ROI


def save_roi(state_path: Path, roi: ROI) -> None:
    Path(state_path).write_text(yaml.safe_dump({"roi": roi.as_dict()}))


def _make_capture_backend(config: AppConfig):  # noqa: ANN202
    name = resolve_backend_name(config.capture.backend)
    if name == "dxcam":
        from smartuibot.vision.capture.mss_backend import MssBackend  # dxcam = later slice

        return MssBackend()  # safe fallback until DxcamBackend lands (S1 follow-up)
    from smartuibot.vision.capture.mss_backend import MssBackend

    return MssBackend()


def _make_detector(config: AppConfig):  # noqa: ANN202
    from smartuibot.vision.detect.yolo import Yolo11Detector

    return Yolo11Detector(
        model_path=config.detection.model,
        device=config.detection.device,
        confidence=config.detection.confidence,
    )


def build_real_container(config_path: Path, state_path: Path) -> AppContainer:
    config = load_config(config_path)
    roi = load_or_default_roi(state_path)
    return AppContainer(
        config=config,
        roi=roi,
        capture_backend=_make_capture_backend(config),
        detector=_make_detector(config),
    )


def main() -> int:
    config_path = Path("configs/default.yaml")
    state_path = Path("configs/state.yaml")
    app = QApplication(sys.argv)
    container = build_real_container(config_path, state_path)
    window = DebugWindow(bus=container.bus,
                         preview_max_width=container.config.ui.preview_max_width)
    window.resize(1100, 700)
    window.show()

    def shutdown(*_a: object) -> None:
        container.stop()
        app.quit()

    signal.signal(signal.SIGINT, shutdown)
    app.aboutToQuit.connect(container.stop)

    # Emergency-stop global hotkey (listener only — never injects input).
    try:
        from pynput import keyboard

        hk = keyboard.GlobalHotKeys(
            {container.config.hotkeys.emergency_stop: shutdown})
        hk.start()
    except Exception:  # noqa: BLE001 - hotkey is best-effort
        pass

    container.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# run.py
from smartuibot.app import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_app_factory.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/app.py run.py tests/unit/test_app_factory.py
git commit -m "feat(app): composition root, run.py, emergency-stop hotkey"
```

---

### Task 24: Interactive controls + ROI overlay wiring

**Files:**
- Create: `src/smartuibot/ui/controls.py`
- Modify: `src/smartuibot/app.py` (replace the `main()` function)
- Test: `tests/unit/test_controls.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_controls.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from smartuibot.core.types import ROI  # noqa: E402
from smartuibot.ui.controls import ControlBar, UiController  # noqa: E402


class _FakeWorker:
    def __init__(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


class _FakeDetection(_FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.conf: float | None = None
        self.reloaded: str | None = None

    def set_confidence(self, v: float) -> None:
        self.conf = v

    def reload_model(self, p: str) -> None:
        self.reloaded = p


class _FakeCapture(_FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.roi: ROI | None = None

    def set_roi(self, roi: ROI) -> None:
        self.roi = roi


class _FakeContainer:
    def __init__(self) -> None:
        self.capture = _FakeCapture()
        self.detection = _FakeDetection()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def _controller(tmp_path):
    saved: list[tuple[Path, ROI]] = []
    c = UiController(
        container=_FakeContainer(),
        state_path=tmp_path / "state.yaml",
        save_roi=lambda p, r: saved.append((p, r)),
    )
    return c, saved


def test_apply_roi_sets_capture_and_persists(tmp_path):
    c, saved = _controller(tmp_path)
    roi = ROI(monitor=1, x=1, y=2, width=30, height=40)
    c.apply_roi(roi)
    assert c.container.capture.roi == roi
    assert saved == [(tmp_path / "state.yaml", roi)]


def test_set_confidence_and_reload_delegate(tmp_path):
    c, _ = _controller(tmp_path)
    c.set_confidence(0.42)
    c.reload_model("custom.pt")
    assert c.container.detection.conf == 0.42
    assert c.container.detection.reloaded == "custom.pt"


def test_toggle_pause_flips_both_workers_and_reports_state(tmp_path):
    c, _ = _controller(tmp_path)
    assert c.toggle_pause() == "paused"
    assert c.container.capture.paused and c.container.detection.paused
    assert c.toggle_pause() == "running"
    assert not c.container.capture.paused and not c.container.detection.paused


def test_request_roi_selection_uses_factory_and_applies(tmp_path):
    c, saved = _controller(tmp_path)
    chosen = ROI(monitor=1, x=0, y=0, width=10, height=10)

    class _FakeOverlay:
        def __init__(self, on_selected):
            on_selected(chosen)  # simulate immediate selection

        def show(self) -> None: ...

    c.set_roi_selector_factory(lambda on_selected: _FakeOverlay(on_selected))
    c.request_roi_selection()
    assert c.container.capture.roi == chosen
    assert saved and saved[-1][1] == chosen


def test_control_bar_slider_sets_confidence(tmp_path):
    app = QApplication.instance() or QApplication([])
    c, _ = _controller(tmp_path)
    bar = ControlBar(controller=c, model_path="yolo11n.pt")
    bar.confidence_slider.setValue(50)
    assert c.container.detection.conf == 0.5
    bar.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_controls.py -q`
Expected: FAIL — `ModuleNotFoundError: smartuibot.ui.controls`

- [ ] **Step 3: Write minimal implementation**

```python
# src/smartuibot/ui/controls.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

from smartuibot.core.types import ROI

RoiSelectorFactory = Callable[[Callable[[ROI], None]], Any]


class UiController:
    """Pure glue between UI actions and the container. No Qt imports used in
    its logic, so every action is unit-testable with a fake container."""

    def __init__(
        self,
        container: Any,
        state_path: Path,
        save_roi: Callable[[Path, ROI], None],
    ) -> None:
        self.container = container
        self._state_path = state_path
        self._save_roi = save_roi
        self._paused = False
        self._roi_factory: RoiSelectorFactory | None = None
        self._overlay: Any | None = None

    def set_roi_selector_factory(self, factory: RoiSelectorFactory) -> None:
        self._roi_factory = factory

    def start(self) -> None:
        self.container.start()

    def stop(self) -> None:
        self.container.stop()

    def toggle_pause(self) -> str:
        self._paused = not self._paused
        for worker in (self.container.capture, self.container.detection):
            worker.pause() if self._paused else worker.resume()
        return "paused" if self._paused else "running"

    def set_confidence(self, value: float) -> None:
        self.container.detection.set_confidence(value)

    def reload_model(self, model_path: str) -> None:
        self.container.detection.reload_model(model_path)

    def apply_roi(self, roi: ROI) -> None:
        self.container.capture.set_roi(roi)
        self._save_roi(self._state_path, roi)

    def request_roi_selection(self) -> None:
        if self._roi_factory is None:
            return
        self._overlay = self._roi_factory(self.apply_roi)
        self._overlay.show()


class ControlBar(QWidget):
    """Buttons + confidence slider wired to a UiController."""

    def __init__(self, controller: UiController, model_path: str) -> None:
        super().__init__()
        self._c = controller
        self._model_path = model_path

        start_btn = QPushButton("Start")
        stop_btn = QPushButton("Stop")
        self.pause_btn = QPushButton("Pause")
        roi_btn = QPushButton("Select ROI")
        reload_btn = QPushButton("Reload model")

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(35)

        start_btn.clicked.connect(self._c.start)
        stop_btn.clicked.connect(self._c.stop)
        self.pause_btn.clicked.connect(self._on_pause)
        roi_btn.clicked.connect(self._c.request_roi_selection)
        reload_btn.clicked.connect(lambda: self._c.reload_model(self._model_path))
        self.confidence_slider.valueChanged.connect(
            lambda v: self._c.set_confidence(v / 100.0))

        layout = QHBoxLayout(self)
        for w in (start_btn, stop_btn, self.pause_btn, roi_btn, reload_btn,
                  QLabel("conf"), self.confidence_slider):
            layout.addWidget(w)

    def _on_pause(self) -> None:
        state = self._c.toggle_pause()
        self.pause_btn.setText("Resume" if state == "paused" else "Pause")
```

Now replace the entire `main()` function in `src/smartuibot/app.py` with this version (imports at the top of `app.py` already include the rest; add the two new imports shown):

```python
def main() -> int:
    from PyQt6.QtWidgets import QVBoxLayout, QWidget as _QWidget

    from smartuibot.ui.controls import ControlBar, UiController
    from smartuibot.ui.roi_selector import ROISelectorOverlay

    config_path = Path("configs/default.yaml")
    state_path = Path("configs/state.yaml")
    app = QApplication(sys.argv)
    container = build_real_container(config_path, state_path)

    controller = UiController(container=container, state_path=state_path,
                              save_roi=save_roi)
    controller.set_roi_selector_factory(
        lambda on_selected: ROISelectorOverlay(
            monitor=container.config.capture.monitor, on_selected=on_selected))

    debug = DebugWindow(bus=container.bus,
                        preview_max_width=container.config.ui.preview_max_width)
    bar = ControlBar(controller=controller,
                     model_path=container.config.detection.model)

    shell = _QWidget()
    shell.setWindowTitle("SmartUIBot")
    shell_layout = QVBoxLayout(shell)
    shell_layout.addWidget(bar)
    shell_layout.addWidget(debug)
    shell.resize(1100, 760)
    shell.show()

    def shutdown(*_a: object) -> None:
        container.stop()
        app.quit()

    signal.signal(signal.SIGINT, shutdown)
    app.aboutToQuit.connect(container.stop)

    try:
        from pynput import keyboard

        hk = keyboard.GlobalHotKeys({container.config.hotkeys.emergency_stop: shutdown})
        hk.start()
    except Exception:  # noqa: BLE001 - hotkey is best-effort
        pass

    container.start()
    if not state_path.exists():
        controller.request_roi_selection()  # first run: ask the user to pick a region
    return app.exec()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_controls.py tests/unit/test_app_factory.py -q`
Expected: PASS (`test_controls.py`: 5 passed; `test_app_factory.py`: 3 passed — `main()` is not exercised by tests so the rewrite is safe)

- [ ] **Step 5: Commit**

```bash
git add src/smartuibot/ui/controls.py src/smartuibot/app.py tests/unit/test_controls.py
git commit -m "feat(ui): runtime controls (pause/conf/reload/ROI) + overlay wiring"
```

---

### Task 25: Full quality gate + docs (README, SETUP)

**Files:**
- Create: `README.md`, `SETUP.md`

- [ ] **Step 1: Run the entire quality gate**

Run:
```bash
.venv/bin/ruff check src tests
.venv/bin/mypy
.venv/bin/pytest -q -m "not model"
```
Expected: ruff clean; mypy `Success: no issues`; all non-model tests PASS, headless (`QT_QPA_PLATFORM=offscreen` is set inside UI tests).

If anything fails, fix with superpowers:systematic-debugging before continuing. Do not weaken assertions or relax mypy.

- [ ] **Step 2: Write `README.md`**

```markdown
# SmartUIBot — Slice A

Read-only real-time CV pipeline: select a screen ROI → capture → YOLO11
detection → live debug window. Cross-platform foundation for the larger
SmartUIBot framework (later slices add decision engine, input, memory).

## Quick start
    python -m pip install -e ".[dev]"
    python run.py

On first run (no `configs/state.yaml`) the ROI selector overlay appears —
drag a rectangle to choose the capture region; it persists across restarts.
The control bar offers Start/Stop, Pause/Resume, a confidence slider, model
hot-reload, and re-select ROI — all at runtime, no restart. Defaults
(model, confidence, target FPS, hotkeys) live in `configs/default.yaml`.

## Architecture
See `docs/superpowers/specs/2026-05-16-smartuibot-slice-a-design.md`.
Threads: capture + detection workers, Qt UI on the main thread, a watchdog
supervisor. A size-1 latest-wins queue ensures inference always runs on the
freshest frame (drop-old backpressure). No mouse/keyboard injection.

## Testing
    pytest -q -m "not model"     # fast, headless, no GPU/screen
    pytest -q -m model           # downloads yolo11n.pt, runs real inference
```

- [ ] **Step 3: Write `SETUP.md`**

```markdown
# Setup

## Python
Python 3.12. `python -m pip install -e ".[dev]"`.

## macOS permissions (REQUIRED)
SmartUIBot reads the screen and listens for a global hotkey. Grant both:

- **Screen Recording**: System Settings → Privacy & Security → Screen
  Recording → enable your terminal / IDE. Without this, captured frames are
  black.
- **Accessibility / Input Monitoring**: required for the emergency-stop
  hotkey listener. SmartUIBot never injects input in Slice A; it only
  listens for the stop key.

Restart the terminal after granting permissions.

## Windows notes
- `mss` works out of the box. The faster `dxcam` backend (Desktop
  Duplication) is a Slice S1 follow-up; `backend: auto` currently falls
  back to `mss` everywhere.
- For GPU inference install a CUDA build of PyTorch, then set
  `detection.device: cuda` in `configs/default.yaml`. On CPU expect lower
  inference FPS than capture FPS — this is expected and shown separately.

## Emergency stop
Default `<ctrl>+<alt>+q` (configurable in `configs/default.yaml`) cleanly
stops all services.
```

- [ ] **Step 4: Re-run the gate to confirm docs didn't break anything**

Run: `.venv/bin/pytest -q -m "not model"`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add README.md SETUP.md
git commit -m "docs: README + SETUP (macOS permissions, Windows notes)"
```

---

## Acceptance Verification (run after Task 25)

Map to spec §11. Use superpowers:verification-before-completion before claiming done.

1. `python run.py` on first run shows the ROI overlay; the picked ROI persists
   in `configs/state.yaml`; YOLO boxes, labels+confidence, and the detections
   table render. ROI logic covered by `tests/unit/test_roi_selector.py` +
   `tests/unit/test_controls.py`; full visual run *(manual, needs screen + weights)*.
2. Capture FPS and detection FPS shown separately and update live —
   `tests/unit/test_debug_window.py`; visual *(manual)*.
3. Confidence change, model hot-reload, pause/resume, re-select ROI at runtime
   (no restart) — covered by `tests/unit/test_controls.py` +
   `tests/unit/test_detection_service.py`; visual *(manual)*.
4. Emergency-stop hotkey cleanly shuts down. *(manual)*
5. Watchdog restarts a crashed worker — `tests/unit/test_watchdog.py`.
6. `ruff`, `mypy --strict`, `pytest -m "not model"` all green headless — Task 25.
7. No input injection anywhere — verified by code review: only `pynput`
   `GlobalHotKeys` *listener* is used; no controller/injection import exists.

> Items marked *(manual)* require a real screen and downloaded weights and are
> validated by the implementer on macOS; they are intentionally not in
> headless CI (documented in the spec).

---

## Notes for the Executor

- **TDD is mandatory** (superpowers:test-driven-development): never write
  implementation before its failing test in the same task.
- **Commit every task.** Never batch.
- **Never weaken a test** to make it pass — debug the code
  (superpowers:systematic-debugging).
- The concrete `dxcam` backend, decision engine, input injection, RAG,
  training pipeline, ONNX/TensorRT, OCR/minimap/replay are **out of scope**
  (later slices). Do not add them. The capture/detector abstractions are
  delivered; only the Windows `dxcam` adapter is deferred (cannot be
  run-tested on the macOS dev box — `backend: auto` falls back to `mss`).
- **Object tracking IDs are intentionally not wired.** The spec marks
  ByteTrack IDs as *optional* for Slice A; the `detection.tracking` config
  key is reserved for the S2 follow-up. `Detection.track_id` stays `None`.
  Do not add tracking unless a later slice requests it.
- `tests/fakes/` keeps the whole pipeline runnable headless — keep fakes in
  sync with the real `CaptureBackend`/`Detector` protocols.
