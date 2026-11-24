"""Segment a trajectory into flicks, tracking segments and holds.

Flicks are anchored on clicks: target positions are unknown, so the click position
is taken as the target and the flick is the movement episode that settles into it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields

import numpy as np

from .kinematics import Trajectory, fill_gaps, gaussian_smooth, runs, split_at_valleys, velocity


@dataclass
class Params:
    smooth_ms: float = 6.0             # Gaussian sigma for position before differentiating
    move_threshold: float = 5.0        # deg/s; above this the crosshair is "moving"
    onset_floor: float = 1.0           # deg/s; onset is traced back to here (true start)
    rest_ms: float = 60.0              # a pause this long separates two movements
    max_flick_window_ms: float = 1500.0  # look back at most this far from a click
    max_click_delay_ms: float = 600.0  # settle -> click longer than this = not a flick
    min_peak_velocity: float = 20.0    # deg/s; slower episodes are micro-adjustments
    min_amplitude: float = 0.5         # deg
    sub_frac: float = 0.1              # submovement threshold as a fraction of peak speed
    valley_frac: float = 0.85          # speed dip that splits blended submovements...
    min_prominence: float = 4.0        # ...if both neighbouring peaks rise this far above it (deg/s)
    min_sub_ms: float = 15.0           # shorter corrective blips are ignored
    min_track_ms: float = 500.0
    track_gap_ms: float = 100.0
    hold_threshold: float = 4.0        # deg/s
    min_hold_ms: float = 150.0
    max_hold_ms: float = 2000.0        # longer still periods are treated as idle
    jitter_band_hz: float = 8.0        # jitter = motion between this and 30 Hz
    buckets: tuple = (5.0, 15.0)       # flick amplitude edges (deg): small | medium | large

    @classmethod
    def from_config(cls, raw: dict | None) -> "Params":
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown analysis setting(s): {sorted(unknown)}")
        if "buckets" in raw:
            raw["buckets"] = tuple(float(b) for b in raw["buckets"])
        return cls(**raw)


BUCKET_NAMES = ("small", "medium", "large")


def bucket_of(amplitude: float, edges) -> str:
    lo, hi = edges
    return "small" if amplitude < lo else "medium" if amplitude < hi else "large"


@dataclass
class Kinematics:
    traj: Trajectory
    spos: np.ndarray    # smoothed position (deg)
    vel: np.ndarray     # deg/s
    speed: np.ndarray   # deg/s


def compute_kinematics(traj: Trajectory, p: Params) -> Kinematics:
    spos = gaussian_smooth(traj.pos, p.smooth_ms / 1000 / traj.dt)
    vel = velocity(spos, traj.dt)
    return Kinematics(traj, spos, vel, np.hypot(vel[:, 0], vel[:, 1]))


@dataclass
class FlickSpan:
    onset: int
    peak: int
    primary_end: int
    settle: int          # speed drops below move_threshold (or the click, if still moving)
    click: int
    submovements: list   # [(start, end)] in the episode, primary included
    primary: int         # index into submovements


def detect_flicks(kin: Kinematics, p: Params) -> tuple[list[FlickSpan], Counter]:
    dt = kin.traj.dt
    to_n = lambda ms: max(1, int(round(ms / 1000 / dt)))
    speed = kin.speed
    flicks, skipped = [], Counter()
    prev = -1
    for k in kin.traj.clicks:
        lo = max(prev + 1, k - to_n(p.max_flick_window_ms), 0)
        prev = k
        r = runs(speed[lo:k + 1] > p.move_threshold) + lo if k > lo else np.empty((0, 2), int)
        if not len(r):
            skipped["no movement before click"] += 1
            continue

        # Walk back from the last movement until a pause of rest_ms: that is the episode.
        first = len(r) - 1
        while first > 0 and r[first, 0] - r[first - 1, 1] < to_n(p.rest_ms):
            first -= 1
        onset, settle = int(r[first, 0]), int(min(r[-1, 1], k))
        # The threshold crossing is late on a slow start; trace back down the ramp.
        while onset > lo and p.onset_floor < speed[onset - 1] < speed[onset]:
            onset -= 1
        if settle <= onset:
            skipped["no movement before click"] += 1
            continue
        if k - settle > to_n(p.max_click_delay_ms):
            skipped["click long after movement"] += 1
            continue

        peak = onset + int(np.argmax(speed[onset:settle]))
        vmax = speed[peak]
        if vmax < p.min_peak_velocity:
            skipped["slow small adjust (follow-up shot)"] += 1
            continue
        if np.linalg.norm(kin.spos[k] - kin.spos[onset]) < p.min_amplitude:
            skipped["too small"] += 1
            continue

        thr = max(p.move_threshold, p.sub_frac * vmax)
        subs = []
        for s, e in runs(speed[onset:settle] > thr) + onset:
            for a, b in split_at_valleys(speed, int(s), int(e), p.valley_frac, p.min_prominence):
                if b - a >= to_n(p.min_sub_ms) or a <= peak < b:
                    subs.append((a, b))
        primary = next(i for i, (a, b) in enumerate(subs) if a <= peak < b)
        flicks.append(FlickSpan(onset, peak, subs[primary][1], settle, int(k), subs, primary))
    return flicks, skipped


def detect_tracking(kin: Kinematics, flicks: list[FlickSpan], p: Params) -> list[tuple[int, int]]:
    """Sustained movement that is not part of a click-anchored flick."""
    dt = kin.traj.dt
    m = fill_gaps(kin.speed > p.move_threshold, int(round(p.track_gap_ms / 1000 / dt)))
    for f in flicks:
        m[f.onset:f.click + 1] = False
    min_n = int(round(p.min_track_ms / 1000 / dt))
    return [(int(s), int(e)) for s, e in runs(m) if e - s >= min_n]


def detect_holds(kin: Kinematics, p: Params) -> list[tuple[int, int]]:
    """Still periods (crosshair held on a spot), excluding long idle stretches."""
    dt = kin.traj.dt
    lo = int(round(p.min_hold_ms / 1000 / dt))
    hi = int(round(p.max_hold_ms / 1000 / dt))
    return [(int(s), int(e)) for s, e in runs(kin.speed < p.hold_threshold) if lo <= e - s <= hi]
