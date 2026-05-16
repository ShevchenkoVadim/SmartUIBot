from smartuibot.platform_support.detect import current_os, resolve_backend_name


def test_current_os_known_value() -> None:
    assert current_os() in {"windows", "macos", "linux"}


def test_auto_resolves_to_mss_off_windows() -> None:
    assert resolve_backend_name("auto", os_name="macos") == "mss"
    assert resolve_backend_name("auto", os_name="linux") == "mss"
    assert resolve_backend_name("auto", os_name="windows") == "dxcam"


def test_explicit_backend_is_respected() -> None:
    assert resolve_backend_name("mss", os_name="windows") == "mss"
