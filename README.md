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
