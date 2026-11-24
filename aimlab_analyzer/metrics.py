"""Per-flick, per-tracking-segment and per-hold metrics, all in degrees."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .kinematics import Trajectory, band_jitter_rms, hf_power_ratio, sparc
from .segment import (
    FlickSpan, Kinematics, Params, bucket_of, compute_kinematics,
    detect_flicks, detect_holds, detect_tracking,
)


def flick_metrics(kin: Kinematics, f: FlickSpan, p: Params) -> dict:
    dt_ms = kin.traj.dt * 1000
    P = kin.spos
    start, target = P[f.onset], P[f.click]
    d = target - start
    amp = float(np.hypot(*d))
    u = d / amp
    normal = np.array([-u[1], u[0]])
    prim = P[f.primary_end] - start
    along = (P[f.onset:f.click + 1] - start) @ u
    path = np.sum(np.hypot(*np.diff(P[f.onset:f.click + 1], axis=0).T))
    speed_click = float(kin.speed[f.click])
    return {
        "onset_s": f.onset * kin.traj.dt,
        "click_s": f.click * kin.traj.dt,
        "amplitude_deg": amp,
        "bucket": bucket_of(amp, p.buckets),
        "direction_deg": float(np.degrees(np.arctan2(-d[1], d[0]))),  # 0 = right, 90 = up
        "time_to_target_ms": (f.click - f.onset) * dt_ms,
        "movement_time_ms": (f.settle - f.onset) * dt_ms,
        "peak_velocity": float(kin.speed[f.peak]),
        "time_to_peak_ms": (f.peak - f.onset) * dt_ms,
        # where in the primary submovement the peak falls (0.5 = symmetric bell)
        "peak_time_frac": (f.peak - f.onset) / max(1, f.primary_end - f.onset),
        # primary submovement endpoint along the target direction: <0 under, >0 over
        "primary_error_pct": float((prim @ u / amp - 1) * 100),
        "primary_offaxis_pct": float(prim @ normal / amp * 100),
        "max_overshoot_pct": float(max(0.0, (along.max() / amp - 1) * 100)),
        "n_corrections": len(f.submovements) - f.primary - 1,
        "correction_time_ms": (f.settle - f.primary_end) * dt_ms,
        "click_delay_ms": (f.click - f.settle) * dt_ms,
        "speed_at_click": speed_click,
        "clicked_moving": speed_click > p.move_threshold,
        "path_efficiency": float(amp / path) if path > 0 else np.nan,
    }


def tracking_metrics(kin: Kinematics, s: int, e: int, p: Params, target_vel: np.ndarray | None = None) -> dict:
    """target_vel: optional (e - s, 2) target angular velocity in deg/s."""
    dt = kin.traj.dt
    sp = kin.speed[s:e]
    raw_vel = np.gradient(kin.traj.pos[s:e], dt, axis=0)
    win = int(round(1.0 / dt))
    sparcs = [sparc(sp[i:i + win], 1 / dt) for i in range(0, len(sp), win) if len(sp[i:i + win]) >= win // 2]
    out = {
        "start_s": s * dt,
        "duration_ms": (e - s) * dt * 1000,
        "mean_speed": float(sp.mean()),
        "speed_cv": float(sp.std() / sp.mean()) if sp.mean() > 0 else np.nan,
        "sparc": float(np.nanmean(sparcs)) if sparcs else np.nan,
        "jitter_hf_ratio": hf_power_ratio(raw_vel, dt, p.jitter_band_hz),
        "jitter_rms_deg": band_jitter_rms(kin.traj.pos[s:e], dt, p.jitter_band_hz),
    }
    if target_vel is not None:
        err = kin.vel[s:e] - target_vel
        tspeed = np.hypot(*np.asarray(target_vel).T)
        out["velocity_rmse"] = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        out["speed_gain"] = float(sp.mean() / tspeed.mean()) if tspeed.mean() > 0 else np.nan
    return out


def hold_metrics(kin: Kinematics, s: int, e: int, p: Params) -> dict:
    dt = kin.traj.dt
    P = kin.traj.pos[s:e]
    clicks = kin.traj.clicks
    tail = int(round(0.05 / dt))
    return {
        "start_s": s * dt,
        "duration_ms": (e - s) * dt * 1000,
        "drift_rms_deg": float(np.sqrt(np.mean(np.sum((P - P.mean(axis=0)) ** 2, axis=1)))),
        "jitter_rms_deg": band_jitter_rms(P, dt, p.jitter_band_hz),
        "before_click": bool(np.any((clicks >= s) & (clicks <= e + tail))),
    }


@dataclass
class Analysis:
    kin: Kinematics
    params: Params
    spans: list[FlickSpan]
    tracking_spans: list[tuple[int, int]]
    hold_spans: list[tuple[int, int]]
    flicks: pd.DataFrame
    tracking: pd.DataFrame
    holds: pd.DataFrame
    skipped: Counter


def analyze(traj: Trajectory, p: Params | None = None) -> Analysis:
    p = p or Params()
    kin = compute_kinematics(traj, p)
    spans, skipped = detect_flicks(kin, p)
    track = detect_tracking(kin, spans, p)
    holds = detect_holds(kin, p)
    return Analysis(
        kin, p, spans, track, holds,
        flicks=pd.DataFrame([flick_metrics(kin, f, p) for f in spans]),
        tracking=pd.DataFrame([tracking_metrics(kin, s, e, p) for s, e in track]),
        holds=pd.DataFrame([hold_metrics(kin, s, e, p) for s, e in holds]),
        skipped=skipped,
    )
