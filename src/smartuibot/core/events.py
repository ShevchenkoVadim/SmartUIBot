# src/smartuibot/core/events.py
from __future__ import annotations

from dataclasses import dataclass

from smartuibot.core.types import ROI, ActionStep, Detection, Frame


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
class DetectionsEnriched(Event):
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


@dataclass(frozen=True, slots=True)
class ActionRequested(Event):
    behavior_name: str
    steps: tuple[ActionStep, ...]
    roi: ROI
    priority: float


@dataclass(frozen=True, slots=True)
class ActionStarted(Event):
    behavior_name: str


@dataclass(frozen=True, slots=True)
class ActionCompleted(Event):
    behavior_name: str


@dataclass(frozen=True, slots=True)
class ActionAborted(Event):
    behavior_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ModeChanged(Event):
    mode: str
