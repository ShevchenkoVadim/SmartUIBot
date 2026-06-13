# src/smartuibot/vision/capture/wayland_backend.py
"""Wayland screen capture via the xdg-desktop-portal ScreenCast interface.

`mss` only knows the X11 `XGetImage` path, which fails on Wayland. Here we
negotiate a ScreenCast session over D-Bus (a one-time consent dialog; a
persisted restore token suppresses it on later runs), open the PipeWire
remote, and pull frames through a GStreamer `pipewiresrc -> appsink` pipeline.
The newest full-monitor frame is cached; `grab(roi)` crops it.

Heavy GI/GStreamer imports are done lazily in __init__ so importing this module
(e.g. during collection of headless tests) never requires a display or PyGObject.
"""
from __future__ import annotations

import logging
import secrets
import threading
from pathlib import Path
from typing import Any

import numpy as np

from smartuibot.core.types import ROI, Image
from smartuibot.vision.capture.backend import Monitor

_log = logging.getLogger("smartuibot.capture.wayland")

_PORTAL_BUS = "org.freedesktop.portal.Desktop"
_PORTAL_PATH = "/org/freedesktop/portal/desktop"
_SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
_REQUEST_IFACE = "org.freedesktop.portal.Request"

# ScreenCast option enums (see the portal spec).
_SOURCE_MONITOR = 1
_CURSOR_HIDDEN = 1
_PERSIST_PERSISTENT = 2


class WaylandPortalError(RuntimeError):
    """Raised when the ScreenCast handshake fails or the user cancels it."""


class WaylandPipeWireBackend:
    """CaptureBackend that streams the screen over the PipeWire ScreenCast
    portal. Conforms to the `CaptureBackend` Protocol (`list_monitors`/`grab`)."""

    def __init__(
        self,
        restore_token_path: Path | str | None = None,
        startup_timeout: float = 60.0,
    ) -> None:
        # Lazy, display-only dependencies.
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gio, GLib, Gst

        self._gio = Gio
        self._glib = GLib
        self._gst = Gst
        if not Gst.is_initialized():
            Gst.init(None)

        self._token_path = Path(restore_token_path) if restore_token_path else None
        self._restore_token: str | None = self._read_token()

        self._frame_lock = threading.Lock()
        self._frame: Image | None = None  # newest full-stream BGR frame
        self._width = 0
        self._height = 0

        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._pipeline: Any = None
        self._loop: Any = None
        self._conn: Any = None
        self._session_handle: str | None = None

        # Drive the whole async portal handshake on a private GLib main loop.
        self._thread = threading.Thread(
            target=self._run_loop, name="wayland-capture", daemon=True
        )
        self._thread.start()

        if not self._ready.wait(timeout=startup_timeout):
            raise WaylandPortalError("timed out waiting for ScreenCast stream")
        if self._error is not None:
            raise WaylandPortalError(str(self._error)) from self._error

    # ---- CaptureBackend Protocol -------------------------------------------

    def list_monitors(self) -> list[Monitor]:
        with self._frame_lock:
            return [Monitor(index=1, x=0, y=0, width=self._width, height=self._height)]

    def grab(self, roi: ROI) -> Image:
        with self._frame_lock:
            frame = self._frame
        if frame is None:
            raise WaylandPortalError("no frame available yet")
        h, w = frame.shape[:2]
        x0 = max(0, min(roi.x, w))
        y0 = max(0, min(roi.y, h))
        x1 = max(x0, min(roi.x + roi.width, w))
        y1 = max(y0, min(roi.y + roi.height, h))
        return np.ascontiguousarray(frame[y0:y1, x0:x1])

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        if self._loop is not None:
            self._loop.quit()

    # ---- GLib main loop / handshake ----------------------------------------

    def _run_loop(self) -> None:
        GLib = self._glib
        ctx = GLib.MainContext.new()
        ctx.push_thread_default()
        self._loop = GLib.MainLoop.new(ctx, False)
        try:
            self._conn = self._gio.bus_get_sync(self._gio.BusType.SESSION, None)
            self._create_session()
            self._loop.run()
        except BaseException as exc:  # noqa: BLE001 - surfaced to constructor
            self._fail(exc)
        finally:
            ctx.pop_thread_default()

    def _fail(self, exc: BaseException) -> None:
        self._error = exc
        self._ready.set()
        if self._loop is not None:
            self._loop.quit()

    def _new_token(self) -> str:
        return "sui_" + secrets.token_hex(8)

    def _request_path(self, token: str) -> str:
        sender = self._conn.get_unique_name()[1:].replace(".", "_")
        return f"{_PORTAL_PATH}/request/{sender}/{token}"

    def _call_request(self, method: str, body: Any, on_response: Any) -> None:
        """Invoke a portal method that answers asynchronously via a Request's
        Response signal. We subscribe to the predicted request path *before*
        calling to avoid missing an early reply."""
        token = self._new_token()
        path = self._request_path(token)
        sub_id = 0

        def _handler(_conn, _sender, _path, _iface, _signal, params):  # type: ignore[no-untyped-def]
            self._conn.signal_unsubscribe(sub_id)
            code = params.get_child_value(0).get_uint32()
            results = params.get_child_value(1)
            on_response(code, results)

        sub_id = self._conn.signal_subscribe(
            _PORTAL_BUS, _REQUEST_IFACE, "Response", path, None,
            self._gio.DBusSignalFlags.NONE, _handler,
        )
        # Inject our handle_token so the portal uses the predicted request path.
        options = body[-1]
        options["handle_token"] = self._glib.Variant("s", token)
        self._conn.call(
            _PORTAL_BUS, _PORTAL_PATH, _SCREENCAST_IFACE, method,
            self._glib.Variant(self._signature(method), body),
            None, self._gio.DBusCallFlags.NONE, -1, None, self._on_call_error,
        )

    def _signature(self, method: str) -> str:
        return {
            "CreateSession": "(a{sv})",
            "SelectSources": "(oa{sv})",
            "Start": "(osa{sv})",
        }[method]

    def _on_call_error(self, conn: Any, res: Any) -> None:
        try:
            conn.call_finish(res)
        except Exception as exc:  # noqa: BLE001 - boundary
            self._fail(exc)

    # ---- handshake steps ----------------------------------------------------

    def _create_session(self) -> None:
        opts = {"session_handle_token": self._glib.Variant("s", self._new_token())}
        self._call_request("CreateSession", (opts,), self._on_session_created)

    def _on_session_created(self, code: int, results: Any) -> None:
        if code != 0:
            return self._fail(WaylandPortalError(f"CreateSession failed (code {code})"))
        self._session_handle = _lookup(results, "session_handle")
        self._select_sources()

    def _select_sources(self) -> None:
        V = self._glib.Variant
        opts: dict[str, Any] = {
            "types": V("u", _SOURCE_MONITOR),
            "multiple": V("b", False),
            "cursor_mode": V("u", _CURSOR_HIDDEN),
            "persist_mode": V("u", _PERSIST_PERSISTENT),
        }
        if self._restore_token:
            opts["restore_token"] = V("s", self._restore_token)
        self._call_request(
            "SelectSources", (self._session_handle, opts), self._on_sources_selected
        )

    def _on_sources_selected(self, code: int, _results: Any) -> None:
        if code != 0:
            return self._fail(WaylandPortalError(f"SelectSources failed (code {code})"))
        self._start()

    def _start(self) -> None:
        self._call_request(
            "Start", (self._session_handle, "", {}), self._on_started
        )

    def _on_started(self, code: int, results: Any) -> None:
        if code != 0:
            return self._fail(WaylandPortalError(f"Start cancelled or failed (code {code})"))
        token = _lookup(results, "restore_token")
        if token:
            self._restore_token = token
            self._write_token(token)
        streams = _lookup(results, "streams")
        if not streams:
            return self._fail(WaylandPortalError("portal returned no streams"))
        node_id, props = streams[0]
        size = props.get("size")
        if size:
            self._width, self._height = int(size[0]), int(size[1])
        self._open_remote(int(node_id))

    def _open_remote(self, node_id: int) -> None:
        # OpenPipeWireRemote answers directly (fd in the reply's fd list), no Request.
        ret, fd_list = self._conn.call_with_unix_fd_list_sync(
            _PORTAL_BUS, _PORTAL_PATH, _SCREENCAST_IFACE, "OpenPipeWireRemote",
            self._glib.Variant("(oa{sv})", (self._session_handle, {})),
            self._glib.VariantType("(h)"), self._gio.DBusCallFlags.NONE, -1,
            None, None,
        )
        fd = fd_list.get(ret.get_child_value(0).get_handle())
        self._build_pipeline(fd, node_id)

    # ---- GStreamer ----------------------------------------------------------

    def _build_pipeline(self, fd: int, node_id: int) -> None:
        Gst = self._gst
        desc = (
            f"pipewiresrc fd={fd} path={node_id} ! videoconvert ! "
            "video/x-raw,format=BGRx ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        self._pipeline = Gst.parse_launch(desc)
        sink = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)
        self._pipeline.set_state(Gst.State.PLAYING)

    def _on_sample(self, sink: Any) -> Any:
        Gst = self._gst
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value("width")
        h = caps.get_value("height")
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            # BGRx: 4 bytes/pixel, row stride is always 4-aligned -> no padding math.
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 4)
            bgr = np.ascontiguousarray(arr[:, :, :3])
        finally:
            buf.unmap(mapinfo)
        with self._frame_lock:
            self._frame = bgr
            self._height, self._width = h, w
        if not self._ready.is_set():
            self._ready.set()
        return Gst.FlowReturn.OK

    # ---- restore token persistence -----------------------------------------

    def _read_token(self) -> str | None:
        if self._token_path and self._token_path.exists():
            return self._token_path.read_text().strip() or None
        return None

    def _write_token(self, token: str) -> None:
        if self._token_path:
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(token)


def _lookup(variant: Any, key: str) -> Any:
    """Read `key` from an a{sv} GLib.Variant, unwrapping to a native value."""
    value = variant.lookup_value(key, None)
    return None if value is None else value.unpack()
