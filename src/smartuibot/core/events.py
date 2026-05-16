# src/smartuibot/core/events.py
from __future__ import annotations

from dataclasses import dataclass

from smartuibot.core.types import Detection, Frame


@dataclass(frozen=True, slots=True)
class Event:
    pass


@dataclass(frozen=True, slots=True)
class FrameCaptured(Event):
    frame: Frame


@dataclass(frozen=True, slots=True)
class DetectionsReady(Event):
    frame: Frame
    detections: tuple[Detection, ...]


@dataclass(frozen=True, slots=True)
class FpsTick(Event):
    name: str
    fps: float


@dataclass(frozen=True, slots=True)
class ServiceError(Event):
    service: str
    error: str
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class LogRecord(Event):
    level: str
    logger: str
    message: str
    ts: float


@dataclass(frozen=True, slots=True)
class StateChanged(Event):
    service: str
    state: str
