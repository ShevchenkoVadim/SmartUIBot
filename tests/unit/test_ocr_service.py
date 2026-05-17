import time

import numpy as np

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import DetectionsEnriched, DetectionsReady
from smartuibot.core.types import ROI, Detection, Frame
from smartuibot.vision.ocr.service import OcrService
from tests.fakes.ocr import FakeOcrEngine

_ROI = ROI(monitor=1, x=0, y=0, width=20, height=20)


def _frame() -> Frame:
    return Frame(image=np.zeros((20, 20, 3), dtype=np.uint8),
                 timestamp=time.monotonic(), seq=1, roi=_ROI)


def _det(label: str, box: tuple[int, int, int, int]) -> Detection:
    return Detection(label=label, confidence=0.9, class_id=0,
                     x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def _run(svc: OcrService, bus: EventBus, ev: DetectionsReady
         ) -> DetectionsEnriched:
    out: list[DetectionsEnriched] = []
    bus.subscribe(DetectionsEnriched, out.append)
    svc.start()
    bus.publish(ev)
    time.sleep(0.2)
    svc.stop()
    assert out, "OcrService did not publish DetectionsEnriched"
    return out[-1]


def test_enriches_configured_label_only() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),
                                     _det("enemy", (2, 2, 8, 8))))
    res = _run(svc, bus, ev)
    by_label = {d.label: d for d in res.detections}
    assert by_label["button"].text == "Close"
    assert by_label["button"].text_confidence == 0.9
    assert by_label["enemy"].text is None  # not a configured label


def test_pass_through_when_disabled() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=False)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None


def test_pass_through_when_engine_none() -> None:
    bus = EventBus()
    svc = OcrService(engine=None, bus=bus, labels=frozenset({"button"}),
                     min_confidence=0.5, enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None


def test_min_confidence_gates_text() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.3), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None  # 0.3 < 0.5


def test_degenerate_crop_is_skipped() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (5, 5, 5, 5)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None


def test_recognize_exception_leaves_text_none_and_still_publishes() -> None:
    class _Boom:
        def recognize(self, image: object) -> tuple[str, float]:
            raise RuntimeError("ocr exploded")

    bus = EventBus()
    svc = OcrService(engine=_Boom(), bus=bus, labels=frozenset({"button"}),
                     min_confidence=0.5, enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("button", (1, 1, 9, 9)),))
    res = _run(svc, bus, ev)
    assert res.detections[0].text is None


def test_pass_through_when_no_matching_labels() -> None:
    bus = EventBus()
    svc = OcrService(engine=FakeOcrEngine("Close", 0.9), bus=bus,
                     labels=frozenset({"button"}), min_confidence=0.5,
                     enabled=True)
    ev = DetectionsReady(frame=_frame(),
                         detections=(_det("enemy", (1, 1, 9, 9)),
                                     _det("ally", (2, 2, 8, 8))))
    res = _run(svc, bus, ev)
    assert all(d.text is None for d in res.detections)
    assert [d.label for d in res.detections] == ["enemy", "ally"]
