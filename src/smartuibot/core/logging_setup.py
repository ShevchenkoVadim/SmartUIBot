# src/smartuibot/core/logging_setup.py
from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import LogRecord

_RESET = "\033[0m"
_COLORS = {"DEBUG": "\033[37m", "INFO": "\033[36m",
           "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[41m"}


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        time_str = self.formatTime(record)
        base = f"{time_str} [{record.levelname}] {record.name}: {record.getMessage()}"
        return f"{color}{base}{_RESET}"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


class _BusHandler(logging.Handler):
    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        self._bus.publish(LogRecord(
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            ts=record.created,
        ))


def setup_logging(level: str, log_dir: Path, bus: EventBus) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("smartuibot")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(_ColorFormatter())
    root.addHandler(console)

    fname = log_dir / f"smartuibot-{time.strftime('%Y%m%d')}.log"
    fileh = RotatingFileHandler(fname, maxBytes=5_000_000, backupCount=5)
    fileh.setFormatter(_JsonFormatter())
    root.addHandler(fileh)

    root.addHandler(_BusHandler(bus))
