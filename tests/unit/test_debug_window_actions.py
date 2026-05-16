# tests/unit/test_debug_window_actions.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from smartuibot.core.event_bus import EventBus  # noqa: E402
from smartuibot.core.events import (  # noqa: E402
    ActionCompleted,
    ActionStarted,
    ModeChanged,
)
from smartuibot.ui.debug_window import DebugWindow  # noqa: E402


def test_debug_window_tracks_mode_and_action_timeline() -> None:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    bus = EventBus()
    win = DebugWindow(bus=bus, preview_max_width=320)
    bus.publish(ModeChanged(mode="armed"))
    bus.publish(ActionStarted(behavior_name="attack"))
    bus.publish(ActionCompleted(behavior_name="attack"))
    win._drain()
    assert win.mode_text() == "armed"
    assert win.action_count() == 2  # started + completed entries
    win.close()
