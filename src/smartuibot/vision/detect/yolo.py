# src/smartuibot/vision/detect/yolo.py
from __future__ import annotations

from typing import Any

import numpy as np

from smartuibot.core.types import Detection


def _results_to_detections(results: list[Any], conf_threshold: float) -> list[Detection]:
    out: list[Detection] = []
    for res in results:
        names = res.names
        for box in res.boxes:
            conf = float(np.asarray(box.conf[0]))
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in np.asarray(box.xyxy[0]))
            cls_id = int(np.asarray(box.cls[0]))
            out.append(Detection(
                label=str(names[cls_id]),
                confidence=conf,
                class_id=cls_id,
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            ))
    return out


class Yolo11Detector:
    def __init__(self, model_path: str, device: str = "auto",
                 confidence: float = 0.35) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        self._confidence = confidence
        self._device = self._resolve_device(device)
        self._model = YOLO(model_path)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def infer(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            image, device=self._device, conf=self._confidence, verbose=False)
        return _results_to_detections(results, self._confidence)

    def reload(self, model_path: str) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        self._model = YOLO(model_path)
