# tests/unit/test_app_factory_slice_b.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

from smartuibot.app import _make_input_backend, build_real_container  # noqa: E402
from smartuibot.input.backend import InputBackend  # noqa: E402


def test_make_input_backend_returns_protocol_impl() -> None:
    from smartuibot.core.config import load_config

    cfg = load_config(Path("configs/default.yaml"))
    be = _make_input_backend(cfg)
    assert isinstance(be, InputBackend)


def test_build_real_container_has_mode_and_action(tmp_path: Path, monkeypatch) -> None:
    import smartuibot.app as app_mod

    class _Stub:
        def infer(self, image: object) -> list:  # noqa: E704
            return []

        def reload(self, p: object) -> None: ...

    monkeypatch.setattr(app_mod, "_make_detector", lambda cfg: _Stub())
    monkeypatch.setattr(app_mod, "_make_capture_backend", lambda cfg: __import__(
        "tests.fakes.capture", fromlist=["FakeCaptureBackend"]).FakeCaptureBackend())
    c = build_real_container(Path("configs/default.yaml"),
                             state_path=tmp_path / "state.yaml")
    assert hasattr(c, "mode") and hasattr(c, "action") and hasattr(c, "decision")
