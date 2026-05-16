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
from smartuibot.vision.capture.backend import CaptureBackend
from smartuibot.vision.detect.detector import Detector

_DEFAULT_ROI = ROI(monitor=1, x=100, y=100, width=640, height=480)


def load_or_default_roi(state_path: Path) -> ROI:
    if Path(state_path).exists():
        data = yaml.safe_load(Path(state_path).read_text()) or {}
        if "roi" in data:
            return ROI.from_dict(data["roi"])
    return _DEFAULT_ROI


def save_roi(state_path: Path, roi: ROI) -> None:
    Path(state_path).write_text(yaml.safe_dump({"roi": roi.as_dict()}))


def _make_capture_backend(config: AppConfig) -> CaptureBackend:
    name = resolve_backend_name(config.capture.backend)
    if name == "dxcam":
        from smartuibot.vision.capture.mss_backend import MssBackend  # dxcam = later slice

        return MssBackend()  # safe fallback until DxcamBackend lands (S1 follow-up)
    from smartuibot.vision.capture.mss_backend import MssBackend

    return MssBackend()


def _make_detector(config: AppConfig) -> Detector:
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
    from PyQt6.QtWidgets import QVBoxLayout
    from PyQt6.QtWidgets import QWidget as _QWidget

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


if __name__ == "__main__":
    raise SystemExit(main())
