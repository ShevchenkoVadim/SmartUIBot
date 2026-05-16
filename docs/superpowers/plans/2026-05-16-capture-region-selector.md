# Capture-Region Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the capture ROI selectable by a macOS-screenshot-style fullscreen mouse drag on the configured monitor, with correct Retina (DPR) pixel mapping and Esc-to-cancel.

**Architecture:** A pure `selection_to_roi()` converts two logical drag-corner points + the screen's `devicePixelRatio` into a physical-pixel `ROI` (offsets within the monitor — `MssBackend.grab()` adds the monitor origin). `ROISelectorOverlay` resolves the target `QScreen` from the configured `Monitor`, sizes itself to that screen, shown fullscreen, and emits the ROI on release / cancels on Esc or sub-minimum drag. `app.py` resolves the configured `Monitor` via a new read-only `CaptureService.list_monitors()` pass-through.

**Tech Stack:** Python 3.12, PyQt6, mss, pytest. Gates: `ruff check .` (E,F,I,UP,B; line-length 100), `mypy` (strict, `smartuibot` package), `pytest`. Use the venv: `.venv/bin/python -m <tool>`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/smartuibot/ui/roi_selector.py` | pure `selection_to_roi`; `_resolve_screen`; `ROISelectorOverlay` (fullscreen drag, Esc) | rewrite |
| `src/smartuibot/ui/controls.py` | `request_roi_selection` shows overlay | 1-line: `show()` → `showFullScreen()` |
| `src/smartuibot/app.py` | overlay factory resolves configured `Monitor` | replace lambda with resolver |
| `src/smartuibot/vision/capture/service.py` | read-only `list_monitors()` pass-through | add method + import |
| `tests/unit/test_roi_selector.py` | unit tests for `selection_to_roi` | replace `rect_to_roi` tests |
| `tests/unit/test_capture_service.py` | service pass-through test | add one test |
| `tests/unit/test_controls.py` | `_FakeOverlay` matches new `showFullScreen` contract | rename `show` → `showFullScreen` |

---

### Task 1: Pure `selection_to_roi` (additive — `rect_to_roi` stays until Task 3)

**Files:**
- Modify: `src/smartuibot/ui/roi_selector.py`
- Test: `tests/unit/test_roi_selector.py` (replace entire file contents)

- [ ] **Step 1: Replace the test file with failing tests**

Overwrite `tests/unit/test_roi_selector.py` with:

```python
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint  # noqa: E402

from smartuibot.core.types import ROI  # noqa: E402
from smartuibot.ui.roi_selector import selection_to_roi  # noqa: E402


def test_selection_to_roi_dpr1_maps_one_to_one() -> None:
    roi = selection_to_roi(QPoint(20, 10), QPoint(100, 90), 1.0, monitor=1)
    assert roi == ROI(monitor=1, x=20, y=10, width=80, height=80)


def test_selection_to_roi_retina_dpr2_scales_to_physical_pixels() -> None:
    roi = selection_to_roi(QPoint(20, 10), QPoint(100, 90), 2.0, monitor=1)
    assert roi == ROI(monitor=1, x=40, y=20, width=160, height=160)


def test_selection_to_roi_normalizes_drag_direction() -> None:
    a = selection_to_roi(QPoint(100, 90), QPoint(20, 10), 1.0, monitor=2)
    b = selection_to_roi(QPoint(20, 10), QPoint(100, 90), 1.0, monitor=2)
    assert a == b == ROI(monitor=2, x=20, y=10, width=80, height=80)


def test_selection_to_roi_below_minimum_returns_none() -> None:
    assert selection_to_roi(QPoint(5, 5), QPoint(9, 9), 1.0, monitor=1) is None


def test_selection_to_roi_minimum_uses_physical_pixels() -> None:
    # 5x5 logical * dpr 2 = 10x10 physical >= 8 -> valid
    roi = selection_to_roi(QPoint(0, 0), QPoint(5, 5), 2.0, monitor=1)
    assert roi == ROI(monitor=1, x=0, y=0, width=10, height=10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_roi_selector.py -q`
Expected: FAIL — `ImportError: cannot import name 'selection_to_roi'`.

- [ ] **Step 3: Add `selection_to_roi` and the minimum constant**

In `src/smartuibot/ui/roi_selector.py`, add this immediately after the existing imports and **above** `def rect_to_roi` (do not remove `rect_to_roi` yet):

```python
_MIN_SELECTION_PX = 8


def selection_to_roi(
    origin: QPoint,
    current: QPoint,
    device_pixel_ratio: float,
    monitor: int,
) -> ROI | None:
    """Convert a drag (two logical-point corners) into a physical-pixel ROI
    relative to the monitor's origin. Returns None for a sub-minimum drag
    (treated as a cancel)."""
    left = min(origin.x(), current.x())
    top = min(origin.y(), current.y())
    width = abs(origin.x() - current.x())
    height = abs(origin.y() - current.y())
    px = round(left * device_pixel_ratio)
    py = round(top * device_pixel_ratio)
    pw = round(width * device_pixel_ratio)
    ph = round(height * device_pixel_ratio)
    if pw < _MIN_SELECTION_PX or ph < _MIN_SELECTION_PX:
        return None
    return ROI(monitor=monitor, x=px, y=py, width=pw, height=ph)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_roi_selector.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q`
Expected: ruff clean; mypy `Success`; full suite green.

```bash
git add src/smartuibot/ui/roi_selector.py tests/unit/test_roi_selector.py
git commit -m "feat(ui): pure selection_to_roi (DPR-aware, cancel on sub-min drag)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `CaptureService.list_monitors()` pass-through

**Files:**
- Modify: `src/smartuibot/vision/capture/service.py:12,31-33`
- Test: `tests/unit/test_capture_service.py` (append one test)

- [ ] **Step 1: Append the failing test**

Add to the end of `tests/unit/test_capture_service.py`:

```python
def test_list_monitors_delegates_to_backend() -> None:
    bus = EventBus()
    backend = FakeCaptureBackend(width=800, height=600)
    svc = CaptureService(
        backend, bus,
        ROI(monitor=1, x=0, y=0, width=10, height=10), target_fps=120)
    assert svc.list_monitors() == backend.list_monitors()
    assert svc.list_monitors()[0].width == 800
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_capture_service.py::test_list_monitors_delegates_to_backend -q`
Expected: FAIL — `AttributeError: 'CaptureService' object has no attribute 'list_monitors'`.

- [ ] **Step 3: Add the pass-through**

In `src/smartuibot/vision/capture/service.py`, change the backend import line (currently line 12):

```python
from smartuibot.vision.capture.backend import CaptureBackend, Monitor
```

Then add this method directly after `set_roi` (after the current line 33):

```python
    def list_monitors(self) -> list[Monitor]:
        return self._backend.list_monitors()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_capture_service.py -q`
Expected: PASS — all capture-service tests green.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q`
Expected: ruff clean; mypy `Success`; full suite green.

```bash
git add src/smartuibot/vision/capture/service.py tests/unit/test_capture_service.py
git commit -m "feat(capture): CaptureService.list_monitors() pass-through

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Fullscreen overlay rewrite + controls + app wiring

**Files:**
- Modify: `src/smartuibot/ui/roi_selector.py` (full rewrite — final content below)
- Modify: `src/smartuibot/ui/controls.py:62`
- Modify: `src/smartuibot/app.py` (factory)
- Modify: `tests/unit/test_controls.py:98-106` (`_FakeOverlay`)

- [ ] **Step 1: Update the controls test fake to the new show contract**

In `tests/unit/test_controls.py`, inside `test_request_roi_selection_uses_factory_and_applies`, replace the `_FakeOverlay` definition's `show` method with `showFullScreen`:

```python
    class _FakeOverlay:
        def __init__(self, on_selected: object) -> None:
            if callable(on_selected):
                on_selected(chosen)  # simulate immediate selection

        def showFullScreen(self) -> None: ...
```

- [ ] **Step 2: Switch the controller to `showFullScreen()`**

In `src/smartuibot/ui/controls.py`, in `request_roi_selection` (line 62), change:

```python
        self._overlay.show()
```
to:
```python
        self._overlay.showFullScreen()
```

- [ ] **Step 3: Run the controls test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_controls.py::test_request_roi_selection_uses_factory_and_applies -q`
Expected: PASS.

- [ ] **Step 4: Rewrite `roi_selector.py` (removes `rect_to_roi`)**

Replace the **entire** contents of `src/smartuibot/ui/roi_selector.py` with:

```python
# src/smartuibot/ui/roi_selector.py
from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import (
    QColor,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QScreen,
)
from PyQt6.QtWidgets import QWidget

from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import Monitor

_MIN_SELECTION_PX = 8


def selection_to_roi(
    origin: QPoint,
    current: QPoint,
    device_pixel_ratio: float,
    monitor: int,
) -> ROI | None:
    """Convert a drag (two logical-point corners) into a physical-pixel ROI
    relative to the monitor's origin. Returns None for a sub-minimum drag
    (treated as a cancel)."""
    left = min(origin.x(), current.x())
    top = min(origin.y(), current.y())
    width = abs(origin.x() - current.x())
    height = abs(origin.y() - current.y())
    px = round(left * device_pixel_ratio)
    py = round(top * device_pixel_ratio)
    pw = round(width * device_pixel_ratio)
    ph = round(height * device_pixel_ratio)
    if pw < _MIN_SELECTION_PX or ph < _MIN_SELECTION_PX:
        return None
    return ROI(monitor=monitor, x=px, y=py, width=pw, height=ph)


def _resolve_screen(mon: Monitor) -> QScreen | None:
    """Pick the QScreen for the configured monitor. Match by physical size
    (Qt logical geometry * its DPR), then fall back to index, then primary."""
    screens = QGuiApplication.screens()
    if not screens:
        return None
    for s in screens:
        dpr = s.devicePixelRatio()
        if (round(s.geometry().width() * dpr) == mon.width
                and round(s.geometry().height() * dpr) == mon.height):
            return s
    idx = mon.index - 1
    if 0 <= idx < len(screens):
        return screens[idx]
    return QGuiApplication.primaryScreen()


class ROISelectorOverlay(QWidget):
    """Fullscreen translucent overlay on the configured monitor: drag a
    rectangle (macOS-screenshot style), release to confirm, Esc to cancel."""

    def __init__(self, monitor: Monitor,
                 on_selected: Callable[[ROI], None]) -> None:
        super().__init__()
        self._monitor_index = monitor.index
        self._on_selected = on_selected
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.35)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        screen = _resolve_screen(monitor)
        if screen is not None:
            self.setScreen(screen)
            self.setGeometry(screen.geometry())

    def _dpr(self) -> float:
        s = self.screen()
        return s.devicePixelRatio() if s is not None else 1.0

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
            roi = selection_to_roi(
                self._origin, event.position().toPoint(),
                self._dpr(), self._monitor_index)
            if roi is not None:
                self._on_selected(roi)
        self.close()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if event is None:
            return
        if self._origin is None or self._current is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 200, 0), 2))
        painter.drawRect(QRect(self._origin, self._current))
```

- [ ] **Step 5: Update the `app.py` overlay factory**

In `src/smartuibot/app.py`, add `from smartuibot.vision.capture.backend import Monitor` to the imports (top-level import block; `ruff -I` will order it). Then, in `main()`, replace:

```python
    controller.set_roi_selector_factory(
        lambda on_selected: ROISelectorOverlay(
            monitor=container.config.capture.monitor, on_selected=on_selected))
```
with:
```python
    def _make_roi_overlay(
        on_selected: Callable[[ROI], None],
    ) -> ROISelectorOverlay:
        mon_index = container.config.capture.monitor
        monitors = container.capture.list_monitors()
        monitor = next((m for m in monitors if m.index == mon_index), None)
        if monitor is None:
            monitor = monitors[0] if monitors else Monitor(
                index=mon_index, x=0, y=0, width=1920, height=1080)
        return ROISelectorOverlay(monitor=monitor, on_selected=on_selected)

    controller.set_roi_selector_factory(_make_roi_overlay)
```

Ensure `from collections.abc import Callable` is in `app.py`'s top-level imports (add it if absent — `ruff -I` will order it). `ROI` is already imported (`src/smartuibot/app.py:13`).

- [ ] **Step 6: Full gate**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q`
Expected: ruff clean (run `.venv/bin/python -m ruff check . --fix` only for import ordering if it flags I001, then re-run); mypy `Success: no issues found`; full suite green. There must be no remaining references to `rect_to_roi` (`grep -rn rect_to_roi src tests` returns nothing).

- [ ] **Step 7: Commit**

```bash
git add src/smartuibot/ui/roi_selector.py src/smartuibot/ui/controls.py src/smartuibot/app.py tests/unit/test_controls.py
git commit -m "feat(ui): macOS-style fullscreen drag-select for capture ROI

Resolve the configured monitor's QScreen, size the overlay to it,
showFullScreen, DPR-correct logical->physical mapping, Esc cancels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Manual verification (no code — Qt geometry needs a real display)

The pure logic is unit-tested; the overlay's screen/geometry/focus behavior is display-dependent and verified by hand.

- [ ] **Step 1: Grant macOS Screen Recording** to the launching terminal/IDE (per `SETUP.md`), else captured frames are black (unrelated to selection correctness).

- [ ] **Step 2: First-run path** — ensure no `configs/state.yaml`, run `.venv/bin/python run.py`. Expect a translucent dimmed overlay covering the whole configured monitor with a crosshair cursor.

- [ ] **Step 3: Drag-select** a region; release. Expect: overlay closes; `configs/state.yaml` now has `roi:` with `monitor`/`x`/`y`/`width`/`height`. On a Retina display, the saved `width`/`height` are ~2× the on-screen pixels you dragged (physical pixels) — correct.

- [ ] **Step 4: Verify capture matches** — the preview/debug window shows exactly the region you selected (not offset or zoomed).

- [ ] **Step 5: Esc cancels** — click "Select ROI", press Esc. Overlay closes; `configs/state.yaml` ROI is unchanged.

- [ ] **Step 6: Stray-click guard** — click "Select ROI", single-click without dragging. Overlay closes; ROI unchanged (no degenerate region).

---

## Self-Review

**Spec coverage:**
- Pure `selection_to_roi` (DPR, 8px min, cancel) → Task 1. ✓
- Overlay sizing / `_resolve_screen` / `showFullScreen` → Task 3 (steps 2, 4). ✓
- Conversion + persistence (release → `apply_roi` unchanged) → Task 3 step 4 (`mouseReleaseEvent`). ✓
- Esc + sub-minimum cancel → Task 3 step 4 (`keyPressEvent`, `selection_to_roi` None). ✓
- `app.py` resolves Monitor via `list_monitors()` → Task 2 + Task 3 step 5. ✓
- Scope guards (single monitor, no HUD) → honored; no multi-screen code. ✓
- Touched-files list matches spec (incl. `service.py` pass-through that the spec marked "only if needed" — it is needed; `CaptureService` had no `list_monitors`). ✓

**Placeholder scan:** no TBD/TODO; every code step shows full code; every command has expected output. ✓

**Type consistency:** `selection_to_roi(origin, current, device_pixel_ratio, monitor) -> ROI | None` identical in Task 1 and Task 3. `ROISelectorOverlay.__init__(monitor: Monitor, on_selected: Callable[[ROI], None])` matches the `app.py` factory call and `RoiSelectorFactory = Callable[[Callable[[ROI], None]], Any]`. `CaptureService.list_monitors() -> list[Monitor]` matches the `CaptureBackend` Protocol and `app.py` usage. `_FakeOverlay.showFullScreen` matches the `controls.py` call. ✓
