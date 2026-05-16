import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import ServiceError
from smartuibot.core.service import Service


class _Counter(Service):
    def __init__(self, bus: EventBus) -> None:
        super().__init__(name="counter", bus=bus)
        self.n = 0

    def run_once(self) -> None:
        self.n += 1
        time.sleep(0.01)


class _Boom(Service):
    def __init__(self, bus: EventBus) -> None:
        super().__init__(name="boom", bus=bus)

    def run_once(self) -> None:
        raise RuntimeError("explode")


def test_service_runs_loop_and_heartbeats() -> None:
    bus = EventBus()
    s = _Counter(bus)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert s.n > 1
    assert s.last_heartbeat > 0.0
    assert not s.is_alive


def test_service_exception_published_and_loop_stops() -> None:
    bus = EventBus()
    errs: list[ServiceError] = []
    bus.subscribe(ServiceError, errs.append)
    s = _Boom(bus)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert any(e.service == "boom" and "explode" in e.error for e in errs)
