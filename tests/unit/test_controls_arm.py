# tests/unit/test_controls_arm.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from smartuibot.ui.controls import ControlBar, UiController  # noqa: E402


class _Mode:
    def __init__(self) -> None:
        self.armed = False

    def arm(self) -> bool:
        self.armed = True
        return True

    def disarm(self) -> bool:
        self.armed = False
        return True

    def is_armed(self) -> bool:
        return self.armed


class _Container:
    def __init__(self) -> None:
        self.mode = _Mode()
        self.capture = type("C", (), {"pause": lambda s: None,
                                      "resume": lambda s: None,
                                      "set_roi": lambda s, r: None})()
        self.detection = type("D", (), {"pause": lambda s: None,
                                        "resume": lambda s: None,
                                        "set_confidence": lambda s, v: None,
                                        "reload_model": lambda s, p: None})()

    def start(self) -> None: ...
    def stop(self) -> None: ...


def test_controller_arm_disarm_toggles_mode(tmp_path: Path) -> None:
    c = UiController(container=_Container(), state_path=tmp_path / "s.yaml",
                     save_roi=lambda p, r: None)
    assert c.is_armed() is False
    assert c.toggle_arm() == "armed"
    assert c.container.mode.is_armed() is True
    assert c.toggle_arm() == "disarmed"
    assert c.container.mode.is_armed() is False


def test_control_bar_has_arm_button(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    c = UiController(container=_Container(), state_path=tmp_path / "s.yaml",
                     save_roi=lambda p, r: None)
    bar = ControlBar(controller=c, model_path="yolo11n.pt")
    bar.arm_btn.click()
    assert c.container.mode.is_armed() is True
    assert bar.arm_btn.text() == "Disarm"
    bar.close()
