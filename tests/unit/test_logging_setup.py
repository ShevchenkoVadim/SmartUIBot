import logging
from pathlib import Path

from smartuibot.core.event_bus import EventBus
from smartuibot.core.events import LogRecord
from smartuibot.core.logging_setup import setup_logging


def test_log_messages_are_published_as_events(tmp_path: Path) -> None:
    bus = EventBus()
    seen: list[LogRecord] = []
    bus.subscribe(LogRecord, seen.append)
    setup_logging(level="INFO", log_dir=tmp_path, bus=bus)
    logging.getLogger("smartuibot.test").info("hello")
    assert any(r.message == "hello" and r.level == "INFO" for r in seen)


def test_rotating_file_is_written(tmp_path: Path) -> None:
    bus = EventBus()
    setup_logging(level="DEBUG", log_dir=tmp_path, bus=bus)
    logging.getLogger("smartuibot.test").warning("disk-me")
    files = list(tmp_path.glob("*.log"))
    assert files and "disk-me" in files[0].read_text()
