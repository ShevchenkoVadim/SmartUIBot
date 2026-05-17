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

## OCR (optional, off by default)
Text-in-detection-box OCR uses PaddleOCR. Install the extra:
`python -m pip install -e ".[ocr]"`. On Intel x86_64 macOS, paddlepaddle
ships only older CPU wheels and inference is slow — keep `ocr.labels` small.
Enable via `ocr.enabled: true` in `configs/default.yaml`.
