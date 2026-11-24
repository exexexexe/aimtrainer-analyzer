"""Uniform-grid trajectories and signal helpers (smoothing, runs, spectra).

Everything downstream works on a Trajectory: view angle in degrees on a uniform
time grid plus click indices. Mouse sessions and (later) demo files both convert to
this, so the same segmentation and metrics apply to both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .session import BTN_LEFT


@dataclass
class Trajectory:
    dt: float            # seconds per sample
    pos: np.ndarray      # (N, 2) cumulative view angle [yaw, pitch] in degrees
    clicks: np.ndarray   # sample indices of left-button presses

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) * self.dt


def from_session(df: pd.DataFrame, dt_ms: float = 1.0) -> Trajectory:
    """Bin a loaded session (with yaw_deg / pitch_deg columns) onto a uniform grid."""
    step = int(dt_ms * 1e6)
    idx = (df["t_ns"].to_numpy() // step).astype(np.int64)
    n = int(idx.max()) + 2 if len(idx) else 1
    dyaw = np.bincount(idx, weights=df["yaw_deg"].to_numpy(), minlength=n)
    dpitch = np.bincount(idx, weights=df["pitch_deg"].to_numpy(), minlength=n)
    # pos[i] = angle after all motion in bins < i, so pos[0] is the start point.
    pos = np.zeros((n, 2))
    pos[1:, 0] = np.cumsum(dyaw)[:-1]
    pos[1:, 1] = np.cumsum(dpitch)[:-1]

    left = (df["buttons"].to_numpy() & BTN_LEFT) > 0
    pressed = left & ~np.concatenate([[False], left[:-1]])
    clicks = np.unique(idx[pressed] + 1)  # first sample after the press bin
    return Trajectory(dt=dt_ms / 1000.0, pos=pos, clicks=clicks[clicks < n])


def gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 with reflected edges."""
    x = np.asarray(x, dtype=float)
    if sigma_samples <= 0 or len(x) < 2:
        return x.copy()
    half = int(np.ceil(4 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_samples) ** 2)
    k /= k.sum()
    xp = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


def gaussian_sigma_for_cutoff(fc_hz: float, dt: float) -> float:
    """Sigma in samples for a Gaussian low-pass whose gain is 0.5 at fc_hz."""
    return 0.1874 / fc_hz / dt


def velocity(pos: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(pos, dt, axis=0)


def runs(mask: np.ndarray) -> np.ndarray:
    """(start, end) pairs, end exclusive, for each run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)])


def fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Set interior runs of False no longer than max_gap to True."""
    out = np.asarray(mask, bool).copy()
    r = runs(out)
    for (s0, e0), (s1, _) in zip(r[:-1], r[1:]):
        if s1 - e0 <= max_gap:
            out[e0:s1] = True
    return out


def split_at_valleys(
    speed: np.ndarray, s: int, e: int, valley_frac: float, min_prominence: float = 0.0
) -> list[tuple[int, int]]:
    """Split one above-threshold run into submovements at speed minima.

    A local minimum splits the run when it is below valley_frac times the peak on
    both sides of it (left side since the last split, right side up to the next
    local minimum) and both peaks rise at least min_prominence above it.
    """
    seg = speed[s:e]
    if len(seg) < 3:
        return [(s, e)]
    interior = np.flatnonzero((seg[1:-1] < seg[:-2]) & (seg[1:-1] <= seg[2:])) + 1
    pieces, start = [], 0
    for i, m in enumerate(interior):
        nxt = interior[i + 1] if i + 1 < len(interior) else len(seg)
        left_peak = seg[start:m].max()
        right_peak = seg[m:nxt].max()
        lower_peak = min(left_peak, right_peak)
        if seg[m] < valley_frac * lower_peak and lower_peak - seg[m] >= min_prominence:
            pieces.append((s + start, s + m))
            start = m
    pieces.append((s + start, e))
    return pieces


def sparc(speed: np.ndarray, fs: float, fc: float = 10.0, amp_th: float = 0.05, pad_level: int = 4) -> float:
    """Spectral arc length (Balasubramanian et al. 2015). Closer to 0 is smoother."""
    if len(speed) < 4 or not np.any(speed):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + pad_level))
    f = np.arange(nfft) * fs / nfft
    mag = np.abs(np.fft.fft(speed, nfft))
    mag /= mag.max()
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    above = np.flatnonzero(mag >= amp_th)
    f, mag = f[above[0]:above[-1] + 1], mag[above[0]:above[-1] + 1]
    if len(f) < 2:
        return 0.0
    df = np.diff(f) / (f[-1] - f[0])
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(mag) ** 2)))


def hf_power_ratio(vel: np.ndarray, dt: float, fc: float, f_max: float = 30.0) -> float:
    """Fraction of velocity power in (fc, f_max] relative to (0, f_max].

    Capping at f_max keeps count-quantization noise (white, spread up to the
    sample rate) mostly out of both bands, so the ratio does not depend on sens.
    """
    v = np.asarray(vel, float).reshape(len(vel), -1)
    if len(v) < 8:
        return float("nan")
    v = (v - v.mean(axis=0)) * np.hanning(len(v))[:, None]
    p = (np.abs(np.fft.rfft(v, axis=0)) ** 2).sum(axis=1)
    f = np.fft.rfftfreq(len(v), dt)
    total = p[(f > 0) & (f <= f_max)].sum()
    return float(p[(f > fc) & (f <= f_max)].sum() / total) if total > 0 else 0.0


def savgol_half_window(fc_hz: float, dt: float, order: int = 3) -> int:
    """Half window for a Savitzky-Golay smoother with -3 dB point near fc_hz."""
    return max(order, int(round(((order + 1) / (2 * fc_hz * dt) + 4.6) / 3.2)))


def savgol_smooth(x: np.ndarray, half: int, order: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0 (reflected edges).

    Unlike a Gaussian it passes slow polynomial-like motion (tracking) unchanged,
    so the residual contains only genuinely fast components.
    """
    x = np.asarray(x, dtype=float)
    half = min(half, (len(x) - 1) // 2)
    if half <= order // 2:
        return x.copy()
    j = np.arange(-half, half + 1)
    coeffs = np.linalg.pinv(np.vander(j, order + 1, increasing=True))[0]
    xp = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    return np.apply_along_axis(lambda c: np.convolve(c, coeffs[::-1], mode="valid"), 0, xp)


def band_jitter_rms(pos: np.ndarray, dt: float, f_lo: float, f_hi: float = 30.0) -> float:
    """RMS (degrees) of position content between roughly f_lo and f_hi Hz."""
    if len(pos) < 8:
        return float("nan")
    h_lo = savgol_half_window(f_lo, dt)
    resid = savgol_smooth(pos, savgol_half_window(f_hi, dt)) - savgol_smooth(pos, h_lo)
    resid = resid.reshape(len(pos), -1)
    if len(resid) > 4 * h_lo:
        resid = resid[h_lo:-h_lo]  # drop edges affected by reflection padding
    return float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
