"""Session control shared by all capture backends."""

from __future__ import annotations

import time
from pathlib import Path

from .config import Config, Conversion
from .session import SessionBuffer, save_session


class Recorder:
    """Receives events and hotkey presses from a backend; owns the current session."""

    def __init__(
        self,
        cfg: Config,
        conversion: Conversion,
        backend: str,
        duration_s: float | None = None,
        log=print,
        chunk_s: float | None = None,
        stop_file: Path | None = None,
        watch_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.conversion = conversion
        self.backend = backend
        self.duration_ns = int(duration_s * 1e9) if duration_s else None
        self.log = log
        self.buf: SessionBuffer | None = None
        self.quit_requested = False
        self.saved: list[Path] = []
        # Continuous mode: save a new file every chunk_s, and whenever watch_dir
        # (the KovaaK's stats folder) changes, so each finished run is on disk at once.
        self.chunk_ns = int(chunk_s * 1e9) if chunk_s else None
        self.stop_file = Path(stop_file) if stop_file else None
        self.watch_dir = Path(watch_dir) if watch_dir else None
        self._watch_mtime = self._dir_mtime()
        self._next_poll = 0

    @property
    def active(self) -> bool:
        return self.buf is not None

    def start(self, t_ns: int | None = None):
        if self.active:
            return
        self.buf = SessionBuffer()
        self.buf.start(t_ns if t_ns is not None else time.perf_counter_ns())
        if not self.chunk_ns:
            self.log(f"[rec] session started ({self.conversion.profile})")

    def stop(self, t_ns: int | None = None) -> Path | None:
        if not self.active:
            return None
        buf, self.buf = self.buf, None
        end = t_ns if t_ns is not None else time.perf_counter_ns()
        duration = end - buf.t0_ns
        if len(buf) == 0:
            if not self.chunk_ns:
                self.log("[rec] session stopped, no events captured, nothing saved")
            return None
        path = save_session(
            buf, self.cfg.output_dir, self.conversion, self.cfg.output_format,
            self.backend, duration,
        )
        rate = len(buf) / (duration / 1e9) if duration else 0
        self.log(f"[rec] saved {len(buf)} events, {duration / 1e9:.1f}s, ~{rate:.0f} Hz -> {path}")
        self.saved.append(path)
        return path

    def toggle(self, t_ns: int | None = None):
        if self.active:
            self.stop(t_ns)
        else:
            self.start(t_ns)

    def request_quit(self):
        self.quit_requested = True

    def on_event(self, t_ns: int, dx: int, dy: int, buttons: int):
        if self.buf is not None:
            self.buf.add(t_ns, dx, dy, buttons)

    def should_exit(self, now_ns: int | None = None) -> bool:
        """Called periodically by the backend loop; handles --duration and continuous mode."""
        now = now_ns if now_ns is not None else time.perf_counter_ns()
        if self.duration_ns and self.active and now - self.buf.t0_ns >= self.duration_ns:
            self.stop(now)
            self.quit_requested = True
        if now >= self._next_poll and (self.stop_file or self.watch_dir or self.chunk_ns):
            self._next_poll = now + 1_000_000_000
            if self.stop_file and self.stop_file.exists():
                self.quit_requested = True
            mtime = self._dir_mtime()
            if mtime != self._watch_mtime:
                self._watch_mtime = mtime
                self.rotate(now)
            elif self.chunk_ns and self.active and now - self.buf.t0_ns >= self.chunk_ns:
                self.rotate(now)
        return self.quit_requested

    def rotate(self, t_ns: int):
        """Save the current file and keep recording into a new one."""
        if self.active and len(self.buf):
            self.stop(t_ns)
            self.start(t_ns)

    def _dir_mtime(self):
        try:
            return self.watch_dir.stat().st_mtime_ns if self.watch_dir else None
        except OSError:
            return None

    def shutdown(self):
        if self.active:
            self.stop()
