# tests/unit/test_pydirectinput_backend.py
import importlib
import sys

import pytest

from smartuibot.input.backend import InputBackend


@pytest.mark.skipif(sys.platform != "win32",
                    reason="pydirectinput is Windows-only; deferred run-test")
def test_pydirectinput_backend_satisfies_protocol() -> None:
    from smartuibot.input.pydirectinput_backend import PyDirectInputBackend

    be: InputBackend = PyDirectInputBackend()
    assert isinstance(be, InputBackend)


def test_module_imports_without_pydirectinput_installed() -> None:
    mod = importlib.import_module("smartuibot.input.pydirectinput_backend")
    assert hasattr(mod, "PyDirectInputBackend")
