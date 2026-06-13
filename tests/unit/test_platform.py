from smartuibot.platform_support.detect import (
    current_os,
    is_wayland,
    resolve_backend_name,
)

_X11 = {"XDG_SESSION_TYPE": "x11"}
_WAYLAND = {"XDG_SESSION_TYPE": "wayland"}


def test_current_os_known_value() -> None:
    assert current_os() in {"windows", "macos", "linux"}


def test_auto_resolves_to_mss_off_windows() -> None:
    assert resolve_backend_name("auto", os_name="macos", env=_X11) == "mss"
    assert resolve_backend_name("auto", os_name="linux", env=_X11) == "mss"
    assert resolve_backend_name("auto", os_name="windows", env=_X11) == "dxcam"


def test_auto_resolves_to_wayland_on_wayland_linux() -> None:
    assert resolve_backend_name("auto", os_name="linux", env=_WAYLAND) == "wayland"
    # Only Linux routes to the portal backend; the env flag alone doesn't.
    assert resolve_backend_name("auto", os_name="macos", env=_WAYLAND) == "mss"


def test_explicit_backend_is_respected() -> None:
    assert resolve_backend_name("mss", os_name="windows", env=_WAYLAND) == "mss"
    assert resolve_backend_name("mss", os_name="linux", env=_WAYLAND) == "mss"


def test_is_wayland_detection() -> None:
    assert is_wayland({"XDG_SESSION_TYPE": "wayland"}) is True
    assert is_wayland({"WAYLAND_DISPLAY": "wayland-0"}) is True
    assert is_wayland({"XDG_SESSION_TYPE": "x11"}) is False
    assert is_wayland({}) is False
