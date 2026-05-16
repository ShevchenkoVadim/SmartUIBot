# tests/unit/test_event_bus.py
import threading

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import FpsTick, ServiceError


def test_publish_delivers_to_matching_subscribers() -> None:
    bus = EventBus()
    seen: list[float] = []
    bus.subscribe(FpsTick, lambda e: seen.append(e.fps))
    bus.publish(FpsTick(name="c", fps=10.0))
    bus.publish(ServiceError(service="x", error="y"))  # no subscriber, must not raise
    assert seen == [10.0]


def test_subscriber_exception_is_isolated() -> None:
    bus = EventBus()
    seen: list[float] = []
    bus.subscribe(FpsTick, lambda e: (_ for _ in ()).throw(RuntimeError("bad")))
    bus.subscribe(FpsTick, lambda e: seen.append(e.fps))
    bus.publish(FpsTick(name="c", fps=5.0))  # must not raise
    assert seen == [5.0]


def test_publish_is_thread_safe() -> None:
    bus = EventBus()
    count = [0]
    lock = threading.Lock()

    def handler(_e: FpsTick) -> None:
        with lock:
            count[0] += 1

    bus.subscribe(FpsTick, handler)
    threads = [threading.Thread(target=bus.publish, args=(FpsTick(name="c", fps=1.0),))
               for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count[0] == 50
