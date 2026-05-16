from smartuibot.core.types import ROI
from smartuibot.vision.capture.backend import CaptureBackend
from tests.fakes.capture import FakeCaptureBackend


def test_fake_backend_satisfies_protocol() -> None:
    fake: CaptureBackend = FakeCaptureBackend(width=20, height=10)
    assert fake.list_monitors()[0].index == 1


def test_fake_grab_returns_roi_sized_bgr_image() -> None:
    fake = FakeCaptureBackend(width=20, height=10)
    img = fake.grab(ROI(monitor=1, x=0, y=0, width=8, height=6))
    assert img.shape == (6, 8, 3)
    assert img.dtype.name == "uint8"
