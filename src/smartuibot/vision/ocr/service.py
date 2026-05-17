# src/smartuibot/vision/ocr/service.py
from __future__ import annotations

import logging
from dataclasses import replace

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsEnriched, DetectionsReady, FpsTick
from smartuibot.core.fps import FpsMeter
from smartuibot.core.latest_queue import LatestQueue
from smartuibot.core.service import Service
from smartuibot.core.types import Detection, Image
from smartuibot.vision.ocr.engine import OcrEngine

_log = logging.getLogger("smartuibot.ocr")


def _crop(img: Image, d: Detection) -> Image | None:
    h, w = img.shape[:2]
    x1 = max(0, min(d.x1, w))
    x2 = max(0, min(d.x2, w))
    y1 = max(0, min(d.y1, h))
    y2 = max(0, min(d.y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


class OcrService(Service):
    """Crops configured-label YOLO boxes, OCRs them, and republishes the
    detection stream as DetectionsEnriched. Pure pass-through when disabled,
    engine is None, or no detection matches the configured labels."""

    def __init__(
        self,
        engine: OcrEngine | None,
        bus: EventBus,
        labels: frozenset[str],
        min_confidence: float,
        enabled: bool,
    ) -> None:
        super().__init__(name="ocr", bus=bus)
        self._engine = engine
        self._labels = labels
        self._min_confidence = min_confidence
        self._enabled = enabled
        self._queue: LatestQueue[DetectionsReady] = LatestQueue()
        self._fps = FpsMeter(window=30)
        self._warned = False
        bus.subscribe(DetectionsReady, self._on_detections)

    def _on_detections(self, event: DetectionsReady) -> None:
        self._queue.put(event)

    def run_once(self) -> None:
        ev = self._queue.get(timeout=0.1)
        if ev is None:
            return
        dets = ev.detections
        active = (
            self._enabled
            and self._engine is not None
            and any(d.label in self._labels for d in dets)
        )
        if active:
            assert self._engine is not None
            out: list[Detection] = []
            for d in dets:
                if d.label not in self._labels:
                    out.append(d)
                    continue
                crop = _crop(ev.frame.image, d)
                if crop is None:
                    out.append(d)
                    continue
                try:
                    text, conf = self._engine.recognize(crop)
                except Exception:  # noqa: BLE001 - one bad crop must not fail the frame
                    if not self._warned:
                        _log.warning("OCR recognize failed; leaving text empty",
                                     exc_info=True)
                        self._warned = True
                    out.append(d)
                    continue
                if text and conf >= self._min_confidence:
                    out.append(replace(d, text=text, text_confidence=conf))
                else:
                    out.append(d)
            dets = tuple(out)
        self._bus.publish(DetectionsEnriched(frame=ev.frame, detections=dets))
        self._fps.tick()
        self._bus.publish(FpsTick(name="ocr", fps=self._fps.fps))
