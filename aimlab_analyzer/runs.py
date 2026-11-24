"""Join mouse recordings with KovaaK's runs: slice, align, label hits/misses, analyze."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Conversion
from .kinematics import band_jitter_rms, from_session
from .kovaaks import Run, run_conversion
from .metrics import analyze, tracking_metrics
from .segment import Params
from .session import meta_path
from .units import add_angles

KILL_MATCH_MS = 40.0


@dataclass
class SessionInfo:
    path: Path
    start: datetime   # naive local time, same clock as KovaaK's timestamps
    duration_s: float
    conversion: dict

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


def index_sessions(sessions_dir: Path) -> list[SessionInfo]:
    out = []
    for m in Path(sessions_dir).glob("*.meta.json"):
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = m.name[: -len(".meta.json")]
        data = next((p for p in (m.with_name(stem + ".parquet"), m.with_name(stem + ".csv")) if p.exists()), None)
        if data is None:
            continue
        start = datetime.fromisoformat(meta["started_at"]).replace(tzinfo=None)
        out.append(SessionInfo(data, start, float(meta["duration_s"]), meta["conversion"]))
    return sorted(out, key=lambda s: s.start)


@lru_cache(maxsize=4)
def _read_raw(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def load_window(sessions: list[SessionInfo], start: datetime, end: datetime, pad_s: float = 1.0):
    """Raw events from every recording overlapping [start - pad, end + pad].

    Returns (frame with t_ns relative to the window start, covered fraction of the run,
    the conversion of the first recording) or (None, 0, None).
    """
    w0, w1 = start - timedelta(seconds=pad_s), end + timedelta(seconds=pad_s)
    parts, covered, conv = [], 0.0, None
    for s in sessions:
        if s.end < w0 or s.start > w1:
            continue
        df = _read_raw(str(s.path), s.path.stat().st_mtime)
        offset_ns = int((s.start - w0).total_seconds() * 1e9)
        t = df["t_ns"].to_numpy() + offset_ns
        keep = (t >= 0) & (t <= int((w1 - w0).total_seconds() * 1e9))
        part = df.loc[keep, ["dx", "dy", "buttons"]].copy()
        part.insert(0, "t_ns", t[keep])
        parts.append(part)
        overlap = (min(s.end, end) - max(s.start, start)).total_seconds()
        covered += max(0.0, overlap)
        conv = conv or s.conversion
    if not parts:
        return None, 0.0, None
    df = pd.concat(parts, ignore_index=True).sort_values("t_ns", kind="stable").reset_index(drop=True)
    run_s = (end - start).total_seconds()
    return df, min(1.0, covered / run_s) if run_s > 0 else 0.0, conv


def estimate_offset_ms(click_ms: np.ndarray, kill_ms: np.ndarray, search_ms: float = 1000.0) -> tuple[float, int]:
    """Clock offset (kill log time - click time) that lines most kills up with a click."""
    if len(click_ms) == 0 or len(kill_ms) == 0:
        return 0.0, 0
    diffs = (kill_ms[:, None] - click_ms[None, :]).ravel()
    diffs = diffs[np.abs(diffs) <= search_ms]
    if len(diffs) == 0:
        return 0.0, 0
    hist, edges = np.histogram(diffs, bins=np.arange(-search_ms, search_ms + 5, 5))
    mode = edges[np.argmax(hist)] + 2.5
    near = diffs[np.abs(diffs - mode) <= 15]
    offset = float(np.median(near))
    matched = int(np.sum(np.min(np.abs(kill_ms[:, None] - offset - click_ms[None, :]), axis=1) <= KILL_MATCH_MS))
    return offset, matched


def label_clicks(click_ms: np.ndarray, kill_ms: np.ndarray, kill_hits: np.ndarray) -> np.ndarray:
    """'kill', 'miss' or 'unknown' per click (times already on the same clock).

    A kill is matched to the nearest unused click within KILL_MATCH_MS before/after it.
    Other clicks leading up to a kill that took a single hit must have missed. With
    multi-hit targets we can't tell which non-lethal shots hit, so they stay unknown.
    """
    labels = np.full(len(click_ms), "unknown", dtype=object)
    used = np.zeros(len(click_ms), bool)
    kill_click = np.full(len(kill_ms), -1)
    for i, k in enumerate(kill_ms):
        d = np.abs(click_ms - k)
        d[used] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= KILL_MATCH_MS:
            used[j] = True
            kill_click[i] = j
            labels[j] = "kill"
    one_hit = len(kill_hits) > 0 and bool(np.all(kill_hits <= 1))
    prev_t = -np.inf
    for i, k in enumerate(kill_ms):
        between = (click_ms > prev_t) & (click_ms <= k + KILL_MATCH_MS) & ~used
        if kill_hits[i] <= 1:
            labels[between] = "miss"
        prev_t = click_ms[kill_click[i]] if kill_click[i] >= 0 else k
    if one_hit:  # shots after the last kill never killed anything
        labels[(click_ms > prev_t) & ~used] = "miss"
    return labels


def classify_misses(kin, clicks: np.ndarray, labels: np.ndarray, spans, p: Params) -> pd.DataFrame:
    """For each missed click: which way the next correction went, and was it fired moving.

    The movement from the miss to the next click shows where the target really was:
    continuing forward = undershoot, coming back = overshoot.
    """
    flick_dir = {f.click: kin.spos[f.click] - kin.spos[f.onset] for f in spans}
    rows = []
    n_next = int(round(1.5 / kin.traj.dt))
    for i, c in enumerate(clicks):
        if labels[i] != "miss":
            continue
        nxt = clicks[i + 1] if i + 1 < len(clicks) and clicks[i + 1] - c <= n_next else None
        approach = flick_dir.get(c)
        if approach is None and i > 0:
            approach = kin.spos[c] - kin.spos[clicks[i - 1]]
        moving = bool(kin.speed[c] > p.move_threshold)
        off_x = off_y = np.nan
        if nxt is None:
            kind, dist = "no follow-up", np.nan
        else:
            v = kin.spos[nxt] - kin.spos[c]
            dist = float(np.hypot(*v))
            # where the miss landed relative to the target, screen axes, when the next
            # kill is plausibly the same target (soon after, and close by)
            amp = np.hypot(*approach) if approach is not None else 0.0
            limit = SAME_TARGET_FRAC * amp if amp > 0.5 else 1.0
            same = (nxt - c) * kin.traj.dt <= SAME_TARGET_MAX_S and 0.05 <= dist <= limit
            if labels[i + 1] == "kill" and same:
                off_x, off_y = float(-v[0]), float(v[1])
            if dist < 0.05:
                kind = "re-clicked same spot"
            elif approach is None or np.hypot(*approach) < 0.1:
                kind = "off target"
            else:
                cos = float(v @ approach / (dist * np.hypot(*approach)))
                kind = "undershoot" if cos > 0.5 else "overshoot" if cos < -0.5 else "sideways"
        rows.append({"t_s": c * kin.traj.dt, "kind": kind, "moving": moving,
                     "speed_at_click": float(kin.speed[c]), "miss_distance_deg": dist,
                     "offset_x": off_x, "offset_y": off_y})
    return pd.DataFrame(rows, columns=["t_s", "kind", "moving", "speed_at_click", "miss_distance_deg",
                                       "offset_x", "offset_y"])


def _med(s) -> float | None:
    s = pd.Series(s, dtype=float).dropna()
    return float(s.median()) if len(s) else None


def _mean(s) -> float | None:
    s = pd.Series(s, dtype=float).dropna()
    return float(s.mean()) if len(s) else None


SECTORS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
PULL_WINDOW_S = 0.02      # movement this soon after the press is mechanical, not a new aim
PULL_CLEAR_S = 0.12       # ...as long as the next flick's peak is at least this far away
PULL_SETTLED_DEG = 0.03   # crosshair must be this still in the window before the press
PULL_FLAG_DEG = 0.03      # pulls bigger than this are counted as noticeable
SAME_TARGET_MAX_S = 0.6   # a miss and the next kill count as the same target within this time...
SAME_TARGET_FRAC = 0.4    # ...and within this fraction of the flick distance (or 1 deg if unknown)


def reaction_times(spans, clicks: np.ndarray, labels: np.ndarray, dt: float, max_ms: float = 1000) -> list:
    """Onset of each flick minus the previous kill click (when the next target appears).

    Only defined when the click right before the flick was a kill; after a miss the
    target was already on screen.
    """
    out = []
    for f in spans:
        j = int(np.searchsorted(clicks, f.onset)) - 1
        rt = None
        if j >= 0 and labels[j] == "kill":
            ms = (f.onset - clicks[j]) * dt * 1000
            rt = ms if ms <= max_ms else None
        out.append(rt)
    return out


def click_pull(pos: np.ndarray, clicks: np.ndarray, dt: float, flick_peaks=()) -> pd.DataFrame:
    """How far the crosshair moves in the 20 ms after a press, for presses made while still.

    Pressing the button with a tense grip drags the mouse; a relaxed click barely moves it.
    Clicks followed quickly by the next flick are skipped so its start isn't counted.
    """
    n = int(round(PULL_WINDOW_S / dt))
    clear = int(round(PULL_CLEAR_S / dt))
    peaks = np.sort(np.asarray(list(flick_peaks), dtype=np.int64))
    rows = []
    for c in clicks:
        if c - n < 0 or c + n >= len(pos):
            continue
        j = np.searchsorted(peaks, c)
        if j < len(peaks) and peaks[j] - c < clear:
            continue
        pre = np.max(np.hypot(*(pos[c - n:c + 1] - pos[c]).T))
        if pre > PULL_SETTLED_DEG:
            continue
        post = np.max(np.hypot(*(pos[c:c + n + 1] - pos[c]).T))
        rows.append({"t_s": c * dt, "pull_deg": float(post), "pre_deg": float(pre)})
    return pd.DataFrame(rows, columns=["t_s", "pull_deg", "pre_deg"])


def velocity_spectrum(pos: np.ndarray, segments, dt: float, nper: int = 512, f_max: float = 30.0):
    """Welch power spectrum of crosshair velocity during the given segments (deg^2/s^2 per Hz)."""
    win = np.hanning(nper)
    acc, count = None, 0
    for s, e in segments:
        v = np.gradient(pos[s:e], dt, axis=0)
        for i in range(0, len(v) - nper + 1, nper // 2):
            seg = (v[i:i + nper] - v[i:i + nper].mean(axis=0)) * win[:, None]
            p = (np.abs(np.fft.rfft(seg, axis=0)) ** 2).sum(axis=1)
            acc = p if acc is None else acc + p
            count += 1
    if not count:
        return None
    f = np.fft.rfftfreq(nper, dt)
    psd = acc / count * dt / (win ** 2).sum()
    sel = (f >= 1) & (f <= f_max)
    f, psd = f[sel], psd[sel]
    # tremor: bump in 6-15 Hz above a power-law trend fitted to the rest of 2-30 Hz
    band = (f >= 6) & (f <= 15)
    rest = (f >= 2) & ~band
    tremor_hz, tremor_ratio = None, None
    if rest.sum() > 3 and band.any() and np.all(psd[rest] > 0):
        k, b = np.polyfit(np.log(f[rest]), np.log(psd[rest]), 1)
        ratio = psd[band] / np.exp(b + k * np.log(f[band]))
        i = int(np.argmax(ratio))
        tremor_hz, tremor_ratio = float(f[band][i]), float(ratio[i])
    return {"f": np.round(f, 2).tolist(), "power": psd.tolist(),
            "tremor_hz": tremor_hz, "tremor_ratio": tremor_ratio}


def tension_timeline(run_s: float, pad_s: float, holds: pd.DataFrame, pulls: pd.DataFrame,
                     track_jitter: list, window_s: float = 10.0) -> dict:
    """Shake measures per window over the run, to see tension or fatigue building up."""
    edges = np.arange(0, run_s + 1e-9, window_s)
    if len(edges) < 2 or edges[-1] < run_s - 1:
        edges = np.append(edges, run_s)
    out = {"window_s": window_s, "t": [], "hold_jitter": [], "click_pull": [], "track_jitter": []}
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = a + pad_s, b + pad_s
        h = holds[(holds.start_s >= lo) & (holds.start_s < hi)] if len(holds) else holds
        pl = pulls[(pulls.t_s >= lo) & (pulls.t_s < hi)] if len(pulls) else pulls
        tj = [j for t, j in track_jitter if lo <= t < hi]
        out["t"].append(float((a + b) / 2))
        out["hold_jitter"].append(_med(h.jitter_rms_deg) if len(h) >= 3 else None)
        out["click_pull"].append(_med(pl.pull_deg) if len(pl) >= 5 else None)
        out["track_jitter"].append(_med(tj) if tj else None)

    def change(series):
        v = [x for x in series if x is not None]
        if len(v) < 4:
            return None
        k = max(1, len(v) // 3)
        first, last = np.median(v[:k]), np.median(v[-k:])
        return float(last / first - 1) if first > 0 else None

    out["change"] = {k: change(out[k]) for k in ("hold_jitter", "click_pull", "track_jitter")}
    return out


def analyze_run_mouse(run: Run, sessions: list[SessionInfo], params: Params, fallback: Conversion | None = None):
    """Mouse metrics for one run, or None if no recording covers it."""
    raw, coverage, rec_conv = load_window(sessions, run.start, run.end)
    if raw is None or coverage < 0.5:
        return None
    conv = run_conversion(run) or fallback
    if conv is None:
        if rec_conv is None:
            return None
        conv = Conversion(**{k: rec_conv[k] for k in
                             ("profile", "game", "dpi", "yaw_deg_per_count", "pitch_deg_per_count", "method")})
    df = add_angles(raw, conv.yaw_deg_per_count, conv.pitch_deg_per_count, conv.dpi)
    traj = from_session(df)
    a = analyze(traj, params)
    kin, dt = a.kin, traj.dt
    dt_ms = dt * 1000
    pad_s = 1.0  # load_window pad
    in_run = lambda t_s: (t_s >= pad_s) & (t_s <= pad_s + run.duration_s)

    clicks = traj.clicks
    click_ms = clicks * dt_ms
    kill_ms = run.kill_times * 1000 + pad_s * 1000
    offset, matched = (0.0, 0)
    labels = np.full(len(click_ms), "unknown", dtype=object)
    if run.kind == "click":
        offset, matched = estimate_offset_ms(click_ms, kill_ms)
        if matched < 0.3 * max(1, len(kill_ms)):
            offset = 0.0
        labels = label_clicks(click_ms, kill_ms - offset, run.kill_hits)

    # ---- flicks ----
    flicks = a.flicks.copy()
    if len(flicks):
        click_to_label = dict(zip(clicks.tolist(), labels))
        flicks["outcome"] = [click_to_label.get(f.click, "unknown") for f in a.spans]
        flicks["reaction_ms"] = reaction_times(a.spans, clicks, labels, dt)
        flicks["ballistic_ms"] = [(f.primary_end - f.onset) * dt_ms for f in a.spans]
        flicks["sector"] = [SECTORS[int(((d + 22.5) % 360) // 45)] for d in flicks.direction_deg]
    kill_flicks = flicks[flicks.outcome == "kill"] if len(flicks) else flicks
    use = kill_flicks if len(kill_flicks) >= 5 else flicks

    def bucket_stats(g):
        return {
            "n": int(len(g)),
            "amplitude_deg": _med(g.get("amplitude_deg", [])),
            "time_to_target_ms": _med(g.get("time_to_target_ms", [])),
            "reaction_ms": _med(g.get("reaction_ms", [])),
            "peak_velocity": _med(g.get("peak_velocity", [])),
            "peak_time_frac": _med(g.get("peak_time_frac", [])),
            "primary_error_pct": _med(g.get("primary_error_pct", [])),
            "offaxis_abs_pct": _med(np.abs(g.primary_offaxis_pct)) if len(g) else None,
            "undershoot_share": float((g.primary_error_pct < -5).mean()) if len(g) else None,
            "overshoot_share": float((g.primary_error_pct > 5).mean()) if len(g) else None,
            "corrections": float(g.n_corrections.mean()) if len(g) else None,
            "click_delay_ms": _med(g.get("click_delay_ms", [])),
            "clicked_moving": float(g.clicked_moving.mean()) if len(g) else None,
            "path_efficiency": _med(g.get("path_efficiency", [])),
        }

    buckets = {b: bucket_stats(use[use.bucket == b] if len(use) else use) for b in ("small", "medium", "large")}
    directions = []
    for name in SECTORS:
        g = use[use.sector == name] if len(use) else use
        directions.append({"name": name, "n": int(len(g)),
                           "along": _med(g.get("primary_error_pct", [])),
                           "perp": _med(g.get("primary_offaxis_pct", []))})

    phases = {
        "reaction_ms": _med(use.get("reaction_ms", [])),
        "ballistic_ms": _med(use.get("ballistic_ms", [])),
        "correction_ms": _med(use.get("correction_time_ms", [])),
        "verify_ms": _med(use.get("click_delay_ms", [])),
    }

    # ---- misses ----
    misses = classify_misses(kin, clicks, labels, a.spans, params)
    placed = misses.dropna(subset=["offset_x"]) if len(misses) else misses
    miss_bias = None
    if len(placed) >= 3:
        mx, my = float(placed.offset_x.mean()), float(placed.offset_y.mean())
        mean_len = float(np.hypot(placed.offset_x, placed.offset_y).mean())
        miss_bias = {"n": int(len(placed)), "mean_x": mx, "mean_y": my,
                     "median_dist": _med(np.hypot(placed.offset_x, placed.offset_y)),
                     "consistency": float(np.hypot(mx, my) / mean_len) if mean_len > 0 else None}
    hits_moving = None
    kill_clicks = clicks[labels == "kill"]
    if len(kill_clicks):
        hits_moving = float(np.mean(kin.speed[kill_clicks] > params.move_threshold))

    # ---- tension ----
    pulls = click_pull(traj.pos, clicks, dt, [f.peak for f in a.spans])
    pulls = pulls[in_run(pulls.t_s)] if len(pulls) else pulls
    mm_per_deg = 25.4 / (conv.yaw_deg_per_count * conv.dpi)
    holds = a.holds[in_run(a.holds.start_s)] if len(a.holds) else a.holds
    # clip tracking segments to the run itself (movement often starts before the run does)
    r0, r1 = int(round(pad_s / dt)), int(round((pad_s + run.duration_s) / dt))
    min_track = int(round(params.min_track_ms / 1000 / dt))
    track_spans = [(max(s, r0), min(e, r1)) for s, e in a.tracking_spans if min(e, r1) - max(s, r0) >= min_track]
    tracking = pd.DataFrame([tracking_metrics(kin, s, e, params) for s, e in track_spans],
                            columns=a.tracking.columns if len(a.tracking.columns) else None)
    win = int(round(2.0 / dt))
    track_jitter = []  # (t_s, jitter) per 2 s piece of tracking
    for s, e in track_spans:
        for i in range(s, e - win // 2, win):
            track_jitter.append((i * dt, band_jitter_rms(traj.pos[i:min(e, i + win)], dt, params.jitter_band_hz)))
    timeline = tension_timeline(run.duration_s, pad_s, holds, pulls, track_jitter,
                                window_s=10.0 if run.duration_s >= 40 else 5.0)
    spectrum = None
    for nper in (1024, 512):  # ~1 Hz resolution when segments are long enough
        spectrum = velocity_spectrum(traj.pos, track_spans, dt, nper=nper) if track_spans else None
        if spectrum:
            break

    # ---- per-item points for charts (capped) ----
    flick_points = [] if not len(use) else [
        {"amp": r.amplitude_deg, "bucket": r.bucket, "dir": r.direction_deg, "along": r.primary_error_pct,
         "perp": r.primary_offaxis_pct, "ttt": r.time_to_target_ms, "pv": r.peak_velocity,
         "rt": r.reaction_ms, "corr": int(r.n_corrections), "out": r.outcome}
        for r in use.head(500).itertuples()
    ]
    miss_points = [{"x": r.offset_x, "y": r.offset_y, "kind": r.kind, "moving": bool(r.moving)}
                   for r in placed.head(500).itertuples()]

    return {
        "coverage": coverage,
        "conversion": conv.to_dict(),
        "clock_offset_ms": offset,
        "kills_matched": matched,
        "clicks": int(len(click_ms)),
        "labels": {k: int(v) for k, v in zip(*np.unique(labels, return_counts=True))} if len(labels) else {},
        "flicks_used": "kills only" if use is kill_flicks and len(kill_flicks) else "all clicks",
        "n_flicks": int(len(use)),
        "buckets": buckets,
        "flicks_all": bucket_stats(use),
        "directions": directions,
        "phases": phases,
        "misses": {
            "n": int(len(misses)),
            "by_kind": misses.kind.value_counts().to_dict(),
            "moving_share": float(misses.moving.mean()) if len(misses) else None,
            "hits_moving_share": hits_moving,
            "median_distance_deg": _med(misses.miss_distance_deg),
        },
        "miss_bias": miss_bias,
        "click_pull": {
            "n": int(len(pulls)),
            "median_deg": _med(pulls.pull_deg),
            "median_mm": (_med(pulls.pull_deg) or 0) * mm_per_deg if len(pulls) else None,
            "share_over": float((pulls.pull_deg >= PULL_FLAG_DEG).mean()) if len(pulls) else None,
            "threshold_deg": PULL_FLAG_DEG,
            "counts_per_deg": 1 / conv.yaw_deg_per_count,
        },
        "tracking": {
            "seconds": float(tracking.duration_ms.sum() / 1000) if len(tracking) else 0.0,
            "mean_speed": _med(tracking.get("mean_speed", [])),
            "speed_cv": _med(tracking.get("speed_cv", [])),
            "sparc": _med(tracking.get("sparc", [])),
            "jitter_hf_ratio": _med(tracking.get("jitter_hf_ratio", [])),
            "jitter_rms_deg": _med(tracking.get("jitter_rms_deg", [])),
        },
        "holds": {
            "n": int(len(holds)),
            "jitter_rms_deg": _med(holds.get("jitter_rms_deg", [])),
            "drift_rms_deg": _med(holds.get("drift_rms_deg", [])),
        },
        "tension": timeline,
        "spectrum": spectrum,
        "flick_points": flick_points,
        "miss_points": miss_points,
        "speed_profile": _median_profile(a, use.index if len(use) else []),
    }


def _median_profile(a, rows, max_ms: int = 600, step_ms: int = 5):
    """Median speed curve (deg/s) of the flicks used, sampled every step_ms from onset."""
    spans = [a.spans[i] for i in rows]
    if len(spans) < 3:
        return None
    dt_ms = a.kin.traj.dt * 1000
    n = int(max_ms / dt_ms)
    stack = np.full((len(spans), n), np.nan)
    for i, f in enumerate(spans):
        sp = a.kin.speed[f.onset:f.click + 1][:n]
        stack[i, :len(sp)] = sp
    enough = np.sum(~np.isnan(stack), axis=0) >= len(spans) / 2
    idx = np.arange(0, n, int(step_ms / dt_ms))
    idx = idx[enough[idx]]
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack[:, idx], axis=0)
    return {"t_ms": (idx * dt_ms).tolist(), "speed": np.round(med, 1).tolist()}
