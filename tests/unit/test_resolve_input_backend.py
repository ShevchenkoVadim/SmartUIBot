# tests/unit/test_resolve_input_backend.py
from smartuibot.platform_support.detect import resolve_input_backend_name


def test_auto_resolves_by_os() -> None:
    assert resolve_input_backend_name("auto", os_name="windows") == "pydirectinput"
    assert resolve_input_backend_name("auto", os_name="macos") == "pynput"
    assert resolve_input_backend_name("auto", os_name="linux") == "pynput"


def test_explicit_respected() -> None:
    assert resolve_input_backend_name("pynput", os_name="windows") == "pynput"
