# src/smartuibot/input/pynput_backend.py
from __future__ import annotations

from typing import Any


class PynputBackend:
    """Real mouse/keyboard via pynput (macOS/Linux + universal).
    pynput controllers are created lazily so the module imports headless."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller as KeyboardController
        from pynput.mouse import Button
        from pynput.mouse import Controller as MouseController

        self._mouse = MouseController()
        self._kbd = KeyboardController()
        self._buttons: dict[str, Any] = {
            "left": Button.left, "right": Button.right, "middle": Button.middle,
        }

    def _btn(self, button: str) -> Any:
        return self._buttons.get(button, self._buttons["left"])

    def move_to(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def mouse_down(self, button: str) -> None:
        self._mouse.press(self._btn(button))

    def mouse_up(self, button: str) -> None:
        self._mouse.release(self._btn(button))

    def click(self, button: str) -> None:
        self._mouse.click(self._btn(button), 1)

    def key_down(self, key: str) -> None:
        self._kbd.press(key)

    def key_up(self, key: str) -> None:
        self._kbd.release(key)

    def type_text(self, text: str) -> None:
        self._kbd.type(text)
