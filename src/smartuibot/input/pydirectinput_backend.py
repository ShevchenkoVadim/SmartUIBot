# src/smartuibot/input/pydirectinput_backend.py
from __future__ import annotations


class PyDirectInputBackend:
    """Windows input via pydirectinput (game-compatible scancodes).
    pydirectinput is imported lazily so this module imports on any OS;
    constructing it off-Windows raises a clear error."""

    def __init__(self) -> None:
        import pydirectinput

        pydirectinput.FAILSAFE = True
        self._pdi = pydirectinput

    def move_to(self, x: int, y: int) -> None:
        self._pdi.moveTo(x, y)

    def mouse_down(self, button: str) -> None:
        self._pdi.mouseDown(button=button)

    def mouse_up(self, button: str) -> None:
        self._pdi.mouseUp(button=button)

    def click(self, button: str) -> None:
        self._pdi.click(button=button)

    def key_down(self, key: str) -> None:
        self._pdi.keyDown(key)

    def key_up(self, key: str) -> None:
        self._pdi.keyUp(key)

    def type_text(self, text: str) -> None:
        self._pdi.write(text)
