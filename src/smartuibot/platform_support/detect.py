# src/smartuibot/platform_support/detect.py
from __future__ import annotations

import sys


def current_os() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def resolve_backend_name(configured: str, os_name: str | None = None) -> str:
    os_name = os_name or current_os()
    if configured != "auto":
        return configured
    return "dxcam" if os_name == "windows" else "mss"
