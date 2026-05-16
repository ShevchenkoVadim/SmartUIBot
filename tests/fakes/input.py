# tests/fakes/input.py
from __future__ import annotations

from typing import Any


class RecordingInputBackend:
    """Records calls instead of touching the OS; lets the loop run headless."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def move_to(self, x: int, y: int) -> None:
        self.calls.append(("move_to", (x, y)))

    def mouse_down(self, button: str) -> None:
        self.calls.append(("mouse_down", (button,)))

    def mouse_up(self, button: str) -> None:
        self.calls.append(("mouse_up", (button,)))

    def click(self, button: str) -> None:
        self.calls.append(("click", (button,)))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", (key,)))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", (key,)))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", (text,)))
