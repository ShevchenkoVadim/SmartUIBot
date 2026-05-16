# tests/unit/test_latest_queue.py
import threading
import time

from smartuibot.core.latest_queue import LatestQueue


def test_put_overwrites_old_value() -> None:
    q: LatestQueue[int] = LatestQueue()
    q.put(1)
    q.put(2)
    q.put(3)
    assert q.get(timeout=0.1) == 3


def test_get_blocks_until_value_then_times_out() -> None:
    q: LatestQueue[int] = LatestQueue()
    assert q.get(timeout=0.05) is None
    result: list[int | None] = []

    def consumer() -> None:
        result.append(q.get(timeout=1.0))

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)
    q.put(99)
    t.join(timeout=1.0)
    assert result == [99]


def test_clear_discards_pending() -> None:
    q: LatestQueue[int] = LatestQueue()
    q.put(5)
    q.clear()
    assert q.get(timeout=0.05) is None
