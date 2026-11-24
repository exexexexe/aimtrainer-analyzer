"""Local web dashboard: per-run stats and findings, served from your own machine."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .config import Config
from .feedback import findings, mouse_usual, scenario_overview, stats_context
from .kovaaks import Run, RunStore
from .runs import analyze_run_mouse, index_sessions
from .segment import Params

WEB_DIR = Path(__file__).parent / "web"
CACHE_VERSION = 2


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


class App:
    def __init__(self, cfg: Config, stats_dir: Path, params: Params):
        self.cfg = cfg
        self.params = params
        self.store = RunStore(stats_dir)
        self.sessions_dir = cfg.output_dir
        self.cache_dir = cfg.path.parent / "reports" / "cache"
        self.lock = threading.RLock()
        self._mouse: dict[str, tuple[str, dict | None]] = {}
        self.sessions = []
        self.sessions_sig = ""
        self.recorder: subprocess.Popen | None = None
        self.refresh()

    # ---------- data ----------

    def refresh(self) -> list[Run]:
        with self.lock:
            runs = self.store.refresh()
            self.sessions = index_sessions(self.sessions_dir) if self.sessions_dir.exists() else []
            self.sessions_sig = f"{len(self.sessions)}:{self.sessions[-1].path.name if self.sessions else ''}"
            return runs

    def has_recording(self, run: Run) -> bool:
        return any(s.start <= run.end and s.end >= run.start for s in self.sessions)

    def mouse(self, run: Run) -> dict | None:
        with self.lock:
            hit = self._mouse.get(run.id)
            if hit and (hit[1] is not None or hit[0] == self.sessions_sig):
                return hit[1]
            disk = self.cache_dir / f"{run.id}.json"
            if disk.exists():
                data = json.loads(disk.read_text(encoding="utf-8"))
                if data.get("version") == CACHE_VERSION:
                    self._mouse[run.id] = (self.sessions_sig, data["mouse"])
                    return data["mouse"]
            m = None
            if self.has_recording(run):
                try:
                    m = _clean(analyze_run_mouse(run, self.sessions, self.params))
                except Exception as e:  # keep the dashboard up; show the reason on the run
                    m = None
                    print(f"[dashboard] analysis failed for {run.scenario} {run.end}: {e!r}")
            self._mouse[run.id] = (self.sessions_sig, m)
            if m and m["coverage"] > 0.95:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                disk.write_text(json.dumps({"version": CACHE_VERSION, "mouse": m}), encoding="utf-8")
            return m

    def runs_payload(self) -> dict:
        runs = self.refresh()
        best, recent, rows = {}, {}, []
        for r in runs:
            prev_best = best.get(r.scenario)
            hist = recent.setdefault(r.scenario, [])
            usual = statistics.median(hist[-10:]) if len(hist) >= 3 else None
            row = r.summary()
            row["is_pb"] = prev_best is not None and r.score > prev_best
            row["vs_usual"] = (r.score / usual - 1) if usual else None
            row["has_recording"] = self.has_recording(r)
            rows.append(row)
            best[r.scenario] = max(r.score, prev_best if prev_best is not None else r.score)
            hist.append(r.score)
        rows = rows[-400:][::-1]
        return {
            "sig": f"{len(runs)}:{runs[-1].id if runs else ''}:{self.sessions_sig}",
            "recording": self.recorder is not None and self.recorder.poll() is None,
            "runs": rows,
        }

    def run_payload(self, rid: str) -> dict | None:
        self.refresh()
        run = self.store.get(rid)
        if run is None:
            return None
        history = [r for r in self.store.runs() if r.scenario == run.scenario]
        ctx = stats_context(run, history)
        mouse = self.mouse(run)
        prev = [r for r in history if r.end < run.end]
        prev_mouse = [self.mouse(r) for r in prev[-10:] if self.has_recording(r)]
        usual = mouse_usual(prev_mouse)
        return {
            "run": run.summary(),
            "context": ctx,
            "history": [{"id": r.id, "end": r.end.isoformat(), "score": r.score, "accuracy": r.accuracy}
                        for r in history],
            "mouse": mouse,
            "usual": usual,
            "findings": findings(run, ctx, mouse, usual),
        }

    def scenarios_payload(self) -> list[dict]:
        return scenario_overview(self.refresh())

    # ---------- recorder subprocess ----------

    def start_recorder(self, profile: str | None):
        self.stop_file = self.cfg.path.parent / "reports" / ".stop-recorder"
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "aimlab_analyzer", "--config", str(self.cfg.path), "record",
               "--continuous", "--stop-file", str(self.stop_file), "--watch-dir", str(self.store.dir)]
        if profile:
            cmd += ["--profile", profile]
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
        self.recorder = subprocess.Popen(cmd, env=env)

    def stop_recorder(self):
        if self.recorder is None or self.recorder.poll() is not None:
            return
        self.stop_file.touch()
        try:
            self.recorder.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.recorder.terminate()


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_clean(obj)).encode("utf-8"), "application/json")

        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            try:
                if path in ("/", "/index.html"):
                    self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/runs":
                    self._json(app.runs_payload())
                elif path.startswith("/api/run/"):
                    data = app.run_payload(path.rsplit("/", 1)[-1])
                    self._json(data if data else {"error": "run not found"}, 200 if data else 404)
                elif path == "/api/scenarios":
                    self._json(app.scenarios_payload())
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:
                self._json({"error": repr(e)}, 500)

    return Handler


def serve(cfg: Config, stats_dir: Path, params: Params, port: int, record: bool, profile: str | None,
          open_browser: bool = True):
    app = App(cfg, stats_dir, params)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://localhost:{port}/"
    print(f"dashboard: {url}  ({len(app.store.runs())} KovaaK's runs from {stats_dir})")
    if record:
        app.start_recorder(profile)
        print("recording mouse in the background (one file per KovaaK's run); Ctrl+C here stops everything")
    else:
        print("not recording mouse input (add --record to analyze movement)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("stopping...")
        server.server_close()
        app.stop_recorder()
