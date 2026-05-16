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


def _controller(tmp_path: Path) -> tuple[UiController, list[tuple[Path, ROI]]]:
    saved: list[tuple[Path, ROI]] = []
    c = UiController(
        container=_FakeContainer(),
        state_path=tmp_path / "state.yaml",
        save_roi=lambda p, r: saved.append((p, r)),
    )
    return c, saved


def test_apply_roi_sets_capture_and_persists(tmp_path: Path) -> None:
    c, saved = _controller(tmp_path)
    roi = ROI(monitor=1, x=1, y=2, width=30, height=40)
    c.apply_roi(roi)
    assert c.container.capture.roi == roi
    assert saved == [(tmp_path / "state.yaml", roi)]


def test_set_confidence_and_reload_delegate(tmp_path: Path) -> None:
    c, _ = _controller(tmp_path)
    c.set_confidence(0.42)
    c.reload_model("custom.pt")
    assert c.container.detection.conf == 0.42
    assert c.container.detection.reloaded == "custom.pt"


def test_toggle_pause_flips_both_workers_and_reports_state(tmp_path: Path) -> None:
    c, _ = _controller(tmp_path)
    assert c.toggle_pause() == "paused"
    assert c.container.capture.paused and c.container.detection.paused
    assert c.toggle_pause() == "running"
    assert not c.container.capture.paused and not c.container.detection.paused


def test_request_roi_selection_uses_factory_and_applies(tmp_path: Path) -> None:
    c, saved = _controller(tmp_path)
    chosen = ROI(monitor=1, x=0, y=0, width=10, height=10)

    class _FakeOverlay:
        def __init__(self, on_selected: object) -> None:
            if callable(on_selected):
                on_selected(chosen)  # simulate immediate selection

        def show(self) -> None: ...

    c.set_roi_selector_factory(lambda on_selected: _FakeOverlay(on_selected))
    c.request_roi_selection()
    assert c.container.capture.roi == chosen
    assert saved and saved[-1][1] == chosen


def test_control_bar_slider_sets_confidence(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    c, _ = _controller(tmp_path)
    bar = ControlBar(controller=c, model_path="yolo11n.pt")
    bar.confidence_slider.setValue(50)
    assert c.container.detection.conf == 0.5
    bar.close()
