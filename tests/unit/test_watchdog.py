import time

from smartuibot.core.event_bus import EventBus
from smartuibot.core.service import Service
from smartuibot.core.watchdog import Watchdog


class _Flaky(Service):
    def __init__(self, bus: EventBus) -> None:
        super().__init__(name="flaky", bus=bus)
        self.starts = 0
        self._fail_once = True

    def start(self) -> None:  # count restarts
        self.starts += 1
        super().start()

    def run_once(self) -> None:
        if self._fail_once:
            self._fail_once = False
            raise RuntimeError("first run dies")
        time.sleep(0.01)


def test_watchdog_restarts_crashed_service() -> None:
    bus = EventBus()
    s = _Flaky(bus)
    wd = Watchdog([s], bus=bus, check_interval=0.02, base_backoff=0.01)
    s.start()
    wd.start()
    time.sleep(0.3)
    wd.stop()
    s.stop()
    assert s.starts >= 2  # restarted at least once
    assert s.is_alive or s.starts >= 2
