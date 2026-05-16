# Capture-Region Selector — Design

**Date:** 2026-05-16
**Status:** Approved (design), pending implementation plan
**Area:** `src/smartuibot/ui/roi_selector.py`, `src/smartuibot/ui/controls.py`

## Problem

The user must be able to select the screen region SmartUIBot captures by
dragging a rectangle with the mouse, the same way macOS ⌘⇧4 screenshot
selection works. The existing `ROISelectorOverlay` does not do this:

1. **It never sizes itself to a screen.** `controls.py` calls
   `self._overlay.show()`, and `ROISelectorOverlay.__init__` sets only
   `FramelessWindowHint | WindowStaysOnTopHint` + translucent background —
   it never sets geometry. The result is a tiny/empty frameless window, not
   a fullscreen capture surface. There is no usable drag area.
2. **Coordinates are widget-local logical points and ignore Retina
   scaling.** `mouse*Event` uses `event.position()` (logical points relative
   to the mis-sized widget); `rect_to_roi` feeds those straight into the
   ROI. But `MssBackend.grab()` treats `roi.x/roi.y` as **physical-pixel
   offsets relative to the chosen monitor's origin**
   (`region.left = mon["left"] + roi.x`). On a Retina display
   (`devicePixelRatio == 2.0`) the captured region is offset/zoomed; on a
   non-primary monitor it is wrong entirely.
3. **No cancel path.** No Esc; a stray click produces a degenerate 1×1 ROI
   (`rect_to_roi` forces `max(1, ...)`), silently corrupting the saved
   capture region in `configs/state.yaml`.

## Scope

In scope:

- Fullscreen drag-select on a **single configured monitor**
  (`capture.monitor` in `configs/default.yaml`, default `1`).
- Correct logical→physical conversion via the target screen's
  `devicePixelRatio()`.
- **Esc** cancels with no change; a sub-minimum drag (stray click) is
  treated as cancel.

Out of scope (explicit, per design decision):

- Multi-monitor spanning / picking the monitor by where the rectangle
  lands / auto-rewriting `capture.monitor`. Single configured monitor only.
- HUD: live `W×H` readout, dimmed backdrop with cut-out, marching-ants
  border.
- Changes to `ROI`, the capture Protocol, mss backend, or the
  `configs/state.yaml` persistence format.

## Approach

Qt-driven. Resolve the target `QScreen` for the configured monitor, size
the overlay to that screen, show it truly fullscreen, and convert the drag
rectangle to physical pixels using that screen's `devicePixelRatio()`. ROI
values remain *offsets within the monitor* — `MssBackend.grab()` already
adds the monitor's absolute origin, so that math is never duplicated here.

Rejected alternatives:

- **mss-driven:** drive overlay geometry from `capture.list_monitors()`
  (physical px), converting physical→logical to place the window then back
  again. Rejected: round-trip rounding error; the UI layer would reach
  through the capture Protocol for geometry it does not own.
- **Minimal patch:** `showFullScreen()` + scale by DPR, assume configured
  monitor == Qt primary. Rejected: silently wrong whenever the configured
  monitor is not primary — a symptom fix, not a root-cause fix.

## Design

### 1. Pure, testable core

Replace `rect_to_roi(p1, p2, monitor)` with a pure function:

```
selection_to_roi(origin: QPoint, current: QPoint,
                 device_pixel_ratio: float, monitor: int) -> ROI | None
```

- Normalize the two logical points (works regardless of drag direction).
- Scale by `device_pixel_ratio`, round to int physical px.
- If the resulting region is below a minimum of **8×8 physical px**
  (a stray click, not a drag), return `None` meaning "cancel".
- Otherwise return `ROI(monitor, x, y, width, height)`.

No Qt event/widget dependency in the math itself (it takes plain
`QPoint`/scalars), so it is unit-testable without a display.

### 2. Overlay sizing (the core bug fix)

The app/controller wiring (`app.py`) resolves the configured monitor's
physical rectangle from the capture backend's `list_monitors()` — a public
method already on the `CaptureBackend` Protocol (`backend.py:23`) — and
passes that rectangle to the overlay factory. (If `CaptureService` does not
already proxy `list_monitors()`, adding a read-only pass-through is a
trivial accessor; the implementation plan will confirm and, if needed,
add it. No Protocol or `state.yaml` format change either way.)

`ROISelectorOverlay` then resolves its target `QScreen`:

1. Match the passed physical rectangle against `QGuiApplication.screens()`
   geometry (left/top/width/height).
2. Fallback: Qt screen at the configured index.
3. Fallback: primary screen.

The overlay sets its geometry to that `QScreen` and is shown via
`showFullScreen()` (changed in `controls.py` from `show()`), giving a drag
surface that covers exactly the configured monitor. Because the widget
covers the screen, widget-local logical coordinates equal screen-local
logical coordinates.

### 3. Conversion + persistence

On mouse release:

1. Build the logical rectangle from widget-local press/release points.
2. Call `selection_to_roi(origin, current, self.screen().devicePixelRatio(),
   configured_monitor)`.
3. If `None` → close without change (see Cancel paths).
4. Else → `controls.apply_roi(roi)` → `capture.set_roi(roi)` + save to
   `configs/state.yaml`. Downstream is unchanged.

### 4. Cancel paths

- `keyPressEvent`: **Esc** closes the overlay without calling
  `_on_selected` — saved ROI unchanged.
- A sub-minimum drag (stray click) yields `None` from `selection_to_roi`
  and is treated identically: close, no change. The app never ends up with
  a garbage 1×1 capture region.

### 5. Scope guards

Single configured monitor only — no multi-screen spanning, no auto-rewrite
of `capture.monitor`, no HUD.

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `selection_to_roi` (pure fn) | logical pts + DPR + monitor → `ROI`/`None` | `ROI`, `QPoint` (value only) |
| `ROISelectorOverlay` (Qt widget) | resolve target screen from passed rect, fullscreen drag, Esc, emit ROI | `selection_to_roi`, `QGuiApplication` |
| `app.py` overlay-factory wiring | resolve configured monitor's physical rect via `list_monitors()`, pass to overlay | capture backend `list_monitors()` |
| `UiController.request_roi_selection` | construct overlay, `showFullScreen()` | overlay factory |
| `UiController.apply_roi` | set ROI on capture + persist | capture service, `save_roi` (unchanged) |

The math is isolated from Qt so it can be understood and tested on its own;
the overlay stays a thin wiring layer.

## Error handling

- Stray click / sub-minimum drag → cancel, no state change.
- Esc → cancel, no state change.
- Screen resolution fallbacks (geometry match → index → primary) ensure the
  overlay always appears even if monitor enumeration is imperfect.
- Screen Recording permission is unrelated to selection correctness:
  selection works without it, but captured frames are black until granted
  (documented in `SETUP.md`); out of scope here.

## Testing

Unit tests for `selection_to_roi`:

- DPR = 1.0: logical rect maps 1:1 to physical ROI.
- DPR = 2.0 (Retina): physical ROI is 2× the logical rect.
- Drag direction independence: bottom-right→top-left equals
  top-left→bottom-right.
- Sub-minimum region (e.g. 3×3 logical) → `None`.
- Off-by-one/rounding at fractional DPR boundaries behaves sanely.

The Qt overlay geometry/`showFullScreen`/keyPress wiring requires a display
and is verified manually; the project already follows this pure-core /
thin-Qt split.

## Touched files

- `src/smartuibot/ui/roi_selector.py` — replace `rect_to_roi` with
  `selection_to_roi`; add screen resolution + `keyPressEvent`; set geometry.
- `src/smartuibot/ui/controls.py` — `showFullScreen()` instead of `show()`.
- `src/smartuibot/app.py` — overlay-factory wiring resolves the configured
  monitor's physical rect via `list_monitors()` and passes it in.
- `src/smartuibot/vision/capture/service.py` — *only if needed*: read-only
  `list_monitors()` pass-through accessor.
- `tests/` — new unit test module for `selection_to_roi`.

No changes to `ROI`, the `CaptureBackend` Protocol, the mss backend, or the
`state.yaml` persistence format.
