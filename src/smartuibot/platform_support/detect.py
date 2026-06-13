# src/smartuibot/platform_support/detect.py
from __future__ import annotations

import os
import sys


def current_os() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def is_wayland(env: dict[str, str] | None = None) -> bool:
    """True on a Wayland session. `mss` (X11/XGetImage) cannot grab pixels
    there, so we route Linux auto-capture to the PipeWire portal backend."""
    env = env if env is not None else dict(os.environ)
    if env.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        return True
    return bool(env.get("WAYLAND_DISPLAY"))


def resolve_backend_name(
    configured: str, os_name: str | None = None, env: dict[str, str] | None = None
) -> str:
    os_name = os_name or current_os()
    if configured != "auto":
        return configured
    if os_name == "windows":
        return "dxcam"
    if os_name == "linux" and is_wayland(env):
        return "wayland"
    return "mss"


def resolve_input_backend_name(configured: str, os_name: str | None = None) -> str:
    os_name = os_name or current_os()
    if configured != "auto":
        return configured
    return "pydirectinput" if os_name == "windows" else "pynput"
