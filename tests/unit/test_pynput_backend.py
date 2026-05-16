# tests/unit/test_pynput_backend.py
import importlib
import os

import pytest

from smartuibot.input.backend import InputBackend


def test_module_imports() -> None:
    mod = importlib.import_module("smartuibot.input.pynput_backend")
    assert hasattr(mod, "PynputBackend")


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="real input device not available in CI",
)
def test_pynput_backend_constructs_and_satisfies_protocol() -> None:
    from smartuibot.input.pynput_backend import PynputBackend

    be: InputBackend = PynputBackend()
    assert isinstance(be, InputBackend)
