import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from smartuibot.app import build_real_container, load_or_default_roi  # noqa: E402
from smartuibot.core.types import ROI  # noqa: E402


def test_load_or_default_roi_uses_state_when_present(tmp_path: Path) -> None:
    state = tmp_path / "state.yaml"
    state.write_text("roi: {monitor: 1, x: 5, y: 6, width: 100, height: 80}\n")
    assert load_or_default_roi(state) == ROI(monitor=1, x=5, y=6, width=100, height=80)


def test_load_or_default_roi_falls_back_when_missing(tmp_path: Path) -> None:
    roi = load_or_default_roi(tmp_path / "absent.yaml")
    assert isinstance(roi, ROI) and roi.width > 0 and roi.height > 0


def test_build_real_container_constructs_without_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Avoid importing torch/ultralytics: stub the YOLO detector factory.
    import smartuibot.app as app_mod

    class _StubDetector:
        def infer(self, image: object) -> list:  # type: ignore[override]
            return []

        def reload(self, p: object) -> None: ...

    monkeypatch.setattr(app_mod, "_make_detector", lambda cfg: _StubDetector())
    monkeypatch.setattr(app_mod, "_make_capture_backend", lambda cfg, sp: __import__(
        "tests.fakes.capture", fromlist=["FakeCaptureBackend"]).FakeCaptureBackend())
    cfg_path = Path("configs/default.yaml")
    container = build_real_container(cfg_path, state_path=tmp_path / "state.yaml")
    assert container.config.detection.model.endswith(".pt")
