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
