import numpy as np
import pytest

from smartuibot.core.types import ROI
from smartuibot.vision.capture.mss_backend import MssBackend


def test_bgra_to_bgr_conversion_shape() -> None:
    # Exercise the pure conversion helper without needing a real screen.
    bgra = np.zeros((4, 5, 4), dtype=np.uint8)
    bgr = MssBackend._to_bgr(bgra)
    assert bgr.shape == (4, 5, 3)


@pytest.mark.skipif(
    __import__("os").environ.get("CI") == "true",
    reason="real screen capture not available in CI",
)
def test_real_grab_returns_requested_size() -> None:
    be = MssBackend()
    mons = be.list_monitors()
    assert mons
    img = be.grab(ROI(monitor=mons[0].index, x=0, y=0, width=16, height=12))
    assert img.shape == (12, 16, 3)
