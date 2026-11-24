"""Synthetic aim movements with known ground truth, for tests and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kinematics import Trajectory


def min_jerk_increments(vec, n: int) -> np.ndarray:
    """Per-sample displacement of a minimum-jerk move covering vec in n samples."""
    tau = np.linspace(0, 1, n + 1)
    shape = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return np.diff(shape)[:, None] * np.asarray(vec, float)[None, :]


class Builder:
    """Compose holds, moves and clicks on a 1 kHz grid (times in ms)."""

    def __init__(self, dt_ms: float = 1.0, seed: int = 0):
        self.dt = dt_ms / 1000
        self.inc = np.zeros((0, 2))
        self.cursor = 0
        self.clicks: list[int] = []
        self.rng = np.random.default_rng(seed)

    def _n(self, ms):
        return int(round(ms / 1000 / self.dt))

    def _add(self, start: int, inc: np.ndarray):
        end = start + len(inc)
        if end > len(self.inc):
            self.inc = np.vstack([self.inc, np.zeros((end - len(self.inc), 2))])
        self.inc[start:end] += inc

    def hold(self, ms, jitter_deg: float = 0.0):
        n = self._n(ms)
        if jitter_deg > 0:
            noise = self.rng.normal(0, jitter_deg, (n + 1, 2))
            noise[0] = noise[-1] = 0
            self._add(self.cursor, np.diff(noise, axis=0))
        else:
            self._add(self.cursor, np.zeros((n, 2)))
        self.cursor += n
        return self

    def move(self, vec, ms, overlap_ms: float = 0.0):
        """Minimum-jerk move; overlap_ms starts it before the previous move ends."""
        start = max(0, self.cursor - self._n(overlap_ms))
        inc = min_jerk_increments(vec, self._n(ms))
        self._add(start, inc)
        self.cursor = max(self.cursor, start + len(inc))
        return self

    def flick(self, vec, ms=150, primary_gain=1.0, correction_ms=100, overlap_ms=0.0, settle_ms=80):
        """Primary move to primary_gain * vec, a correction to vec if needed, a hold, a click."""
        vec = np.asarray(vec, float)
        self.move(vec * primary_gain, ms)
        if abs(primary_gain - 1) > 1e-9:
            self.move(vec * (1 - primary_gain), correction_ms, overlap_ms)
        self.hold(settle_ms)
        return self.click()

    def track(self, ms, speed_deg_s=40.0, freq_hz=0.7, noise_deg=0.0):
        """Smooth side-to-side tracking with optional high-frequency position noise."""
        n = self._n(ms)
        t = np.arange(n) * self.dt
        vx = speed_deg_s * np.sin(2 * np.pi * freq_hz * t + np.pi / 2)
        vy = 0.3 * speed_deg_s * np.sin(2 * np.pi * 0.37 * t)
        inc = np.column_stack([vx, vy]) * self.dt
        if noise_deg > 0:
            noise = self.rng.normal(0, noise_deg, (n + 1, 2))
            noise[0] = noise[-1] = 0
            inc = inc + np.diff(noise, axis=0)
        self._add(self.cursor, inc)
        self.cursor += n
        return self

    def click(self):
        self.clicks.append(self.cursor)
        return self

    def trajectory(self) -> Trajectory:
        inc = self.inc[: max(self.cursor, len(self.inc))]
        pos = np.vstack([np.zeros(2), np.cumsum(inc, axis=0)])
        return Trajectory(dt=self.dt, pos=pos, clicks=np.array(self.clicks, dtype=np.int64))

    def session_frame(self, deg_per_count: float, release_ms: float = 40.0) -> pd.DataFrame:
        """Raw logger-style frame (t_ns, dx, dy, buttons) with count quantization."""
        traj = self.trajectory()
        counts = np.round(traj.pos / deg_per_count).astype(np.int64)
        d = np.diff(counts, axis=0)  # motion during bin i
        n = len(d)
        held = np.zeros(n, bool)
        for c in self.clicks:
            held[max(c - 1, 0): c - 1 + self._n(release_ms)] = True
        change = np.diff(np.concatenate([[False], held])) != 0
        keep = (d[:, 0] != 0) | (d[:, 1] != 0) | change
        i = np.flatnonzero(keep)
        step = int(round(self.dt * 1e9))
        return pd.DataFrame({
            "t_ns": i * step + step // 2,
            "dx": d[i, 0], "dy": d[i, 1],
            "buttons": held[i].astype(np.uint8),
        })
