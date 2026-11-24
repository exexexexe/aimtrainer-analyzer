"""Session summary report: markdown, per-item CSVs and matplotlib plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import Analysis  # noqa: E402
from .segment import BUCKET_NAMES  # noqa: E402

# Reference palette (dataviz skill), light surface. Categorical slots 1-3 = buckets.
BUCKET_COLORS = {"small": "#2a78d6", "medium": "#eb6834", "large": "#1baf7a"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def bucket_labels(edges) -> dict[str, str]:
    lo, hi = edges
    return {"small": f"small (<{lo:g}°)", "medium": f"medium ({lo:g}–{hi:g}°)", "large": f"large (≥{hi:g}°)"}


# ---------- tables ----------

FLICK_ROWS = [
    # (label, column, aggregation, format)
    ("flicks", None, "count", "{:.0f}"),
    ("amplitude (deg)", "amplitude_deg", "median", "{:.1f}"),
    ("time to target (ms)", "time_to_target_ms", "median", "{:.0f}"),
    ("peak velocity (deg/s)", "peak_velocity", "median", "{:.0f}"),
    ("time to peak (ms)", "time_to_peak_ms", "median", "{:.0f}"),
    ("peak position in primary (0.5 = symmetric)", "peak_time_frac", "median", "{:.2f}"),
    ("primary endpoint error (%)", "primary_error_pct", "median_iqr", "{:+.1f}"),
    ("overshoots (primary > +5%)", "primary_error_pct", "share_over", "{:.0%}"),
    ("undershoots (primary < -5%)", "primary_error_pct", "share_under", "{:.0%}"),
    ("corrective submovements", "n_corrections", "mean", "{:.2f}"),
    ("correction time (ms)", "correction_time_ms", "median", "{:.0f}"),
    ("settle to click (ms)", "click_delay_ms", "median", "{:.0f}"),
    ("clicked while moving", "clicked_moving", "mean", "{:.0%}"),
    ("path efficiency", "path_efficiency", "median", "{:.3f}"),
]


def _agg(s: pd.Series, how: str, fmt: str) -> str:
    if s.empty:
        return "–"
    if how == "count":
        return fmt.format(len(s))
    if how == "median":
        return fmt.format(s.median())
    if how == "mean":
        return fmt.format(s.astype(float).mean())
    if how == "median_iqr":
        q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
        return f"{fmt.format(q2)} [{fmt.format(q1)}, {fmt.format(q3)}]"
    if how == "share_over":
        return fmt.format((s > 5).mean())
    if how == "share_under":
        return fmt.format((s < -5).mean())
    raise ValueError(how)


def flick_summary(flicks: pd.DataFrame, edges) -> pd.DataFrame:
    labels = bucket_labels(edges)
    groups = {labels[b]: flicks[flicks.bucket == b] if len(flicks) else flicks for b in BUCKET_NAMES}
    groups["all"] = flicks
    rows = {}
    for label, col, how, fmt in FLICK_ROWS:
        rows[label] = {g: _agg(df[col] if col else df.get("amplitude_deg", pd.Series(dtype=float)), how, fmt)
                       for g, df in groups.items()}
    return pd.DataFrame(rows).T


def _md_table(df: pd.DataFrame, index_name: str = "") -> str:
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


# ---------- plots ----------

def _empty(ax, text="no flicks"):
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def plot_velocity_profiles(a: Analysis, path: Path, max_ms: float = 600):
    dt_ms = a.kin.traj.dt * 1000
    labels = bucket_labels(a.params.buckets)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    grid = np.arange(0, max_ms, dt_ms)
    for ax, b in zip(axes, BUCKET_NAMES):
        ax.set_title(labels[b], loc="left")
        spans = [f for f, bucket in zip(a.spans, a.flicks.get("bucket", [])) if bucket == b]
        if not spans:
            _empty(ax)
            continue
        stack = np.full((len(spans), len(grid)), np.nan)
        for i, f in enumerate(spans):
            sp = a.kin.speed[f.onset:f.click + 1][: len(grid)]
            stack[i, : len(sp)] = sp
            ax.plot(grid[: len(sp)], sp, color=BUCKET_COLORS[b], lw=0.8, alpha=0.25)
        enough = np.sum(~np.isnan(stack), axis=0) >= max(2, len(spans) // 2)
        if enough.any():
            ax.plot(grid[enough], np.nanmedian(stack[:, enough], axis=0), color=INK, lw=2, label="median")
            ax.legend(loc="upper right")
        ax.set_xlabel("ms from onset")
    axes[0].set_ylabel("angular speed (deg/s)")
    fig.suptitle("Flick velocity profiles (onset to click)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overshoot(a: Analysis, path: Path):
    labels = bucket_labels(a.params.buckets)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    bins = np.arange(-60, 62.5, 5)
    for ax, b in zip(axes, BUCKET_NAMES):
        ax.set_title(labels[b], loc="left")
        vals = a.flicks.primary_error_pct[a.flicks.bucket == b] if len(a.flicks) else []
        if len(vals) == 0:
            _empty(ax)
            continue
        ax.hist(np.clip(vals, -60, 60), bins=bins, color=BUCKET_COLORS[b], edgecolor=SURFACE, linewidth=1)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.axvline(0, color=AXIS, lw=1)
        med = float(np.median(vals))
        ax.axvline(med, color=INK, lw=2)
        ax.annotate(f"median {med:+.1f}%", (med, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", va="top", color=INK, fontsize=8)
        ax.set_xlabel("primary endpoint error (%)  ← under | over →")
    axes[0].set_ylabel("flicks")
    fig.suptitle("Where the first (ballistic) movement lands", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_path_overlays(a: Analysis, path: Path):
    labels = bucket_labels(a.params.buckets)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, b in zip(axes, BUCKET_NAMES):
        ax.set_title(labels[b], loc="left")
        spans = [f for f, bucket in zip(a.spans, a.flicks.get("bucket", [])) if bucket == b]
        if not spans:
            _empty(ax)
            continue
        for f in spans:
            P = a.kin.spos[f.onset:f.click + 1] - a.kin.spos[f.onset]
            P = P * [1, -1]  # screen up = positive
            d = P[-1]
            amp = np.hypot(*d)
            c, s = d / amp
            rot = np.array([[c, s], [-s, c]])
            Q = P @ rot.T / amp
            ax.plot(Q[:, 0], Q[:, 1], color=BUCKET_COLORS[b], lw=0.8, alpha=0.35)
        ax.plot([0], [0], "o", color=INK_2, ms=5)
        ax.plot([1], [0], "o", mfc=SURFACE, mec=INK, mew=2, ms=10)
        ax.set_xlabel("progress toward click point (1 = target)")
        ax.set_xlim(-0.15, 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("off-axis (fraction of distance)")
    fig.suptitle("Flick paths, rotated and scaled so every target sits at (1, 0)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_timeline(a: Analysis, path: Path):
    t = a.kin.traj.t
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for f, b in zip(a.spans, a.flicks.get("bucket", [])):
        ax.axvspan(t[f.onset], t[f.click], color=BUCKET_COLORS[b], alpha=0.18, lw=0)
    for s, e in a.tracking_spans:
        ax.axvspan(t[s], t[e - 1], color=NEUTRAL, alpha=0.15, lw=0, hatch="///")
    ax.plot(t, a.kin.speed, color=INK_2, lw=0.6)
    for c in a.kin.traj.clicks:
        ax.axvline(t[c], color=INK, lw=0.5, alpha=0.5, ymax=0.06)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BUCKET_COLORS[b], alpha=0.35) for b in BUCKET_NAMES]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NEUTRAL, alpha=0.3, hatch="///"))
    labels = list(bucket_labels(a.params.buckets).values()) + ["tracking"]
    ax.legend(handles, labels, ncol=4, loc="upper right")
    ax.set_xlabel("session time (s)   ticks at bottom = clicks")
    ax.set_ylabel("angular speed (deg/s)")
    ax.set_title("Session timeline: detected flicks and tracking", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- report ----------

def write_report(a: Analysis, meta: dict, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    a.flicks.to_csv(out_dir / "flicks.csv", index=False)
    a.tracking.to_csv(out_dir / "tracking.csv", index=False)
    a.holds.to_csv(out_dir / "holds.csv", index=False)

    plots = {
        "timeline.png": plot_timeline,
        "velocity_profiles.png": plot_velocity_profiles,
        "overshoot.png": plot_overshoot,
        "paths.png": plot_path_overlays,
    }
    for fname, fn in plots.items():
        fn(a, out_dir / fname)

    conv = meta["conversion"]
    n_clicks = len(a.kin.traj.clicks)
    lines = [
        f"# Session report: {name}",
        "",
        f"- Recorded {meta['started_at']}, {meta['duration_s']:.1f} s, profile `{conv['profile']}` "
        f"({conv['cm_per_360']:.1f} cm/360, {conv['yaw_deg_per_count']:.5f} deg/count)",
        f"- {n_clicks} clicks -> {len(a.flicks)} flicks detected"
        + (", skipped: " + ", ".join(f"{v} {k}" for k, v in a.skipped.items()) if a.skipped else ""),
        f"- {len(a.tracking)} tracking segments, {len(a.holds)} holds",
        "",
        "Targets are not known, so each flick's target is assumed to be where you clicked. "
        "Missed shots therefore count as if the click point was the target.",
        "",
        "![timeline](timeline.png)",
        "",
        "## Flicks",
        "",
    ]
    if len(a.flicks):
        lines += [
            _md_table(flick_summary(a.flicks, a.params.buckets), "metric (median unless noted)"),
            "",
            "Primary endpoint error is where the first, ballistic part of the flick ends, as a percentage "
            "of the flick distance (−20% = stopped 20% short, then corrected). Shown as median [25th, 75th].",
            "",
            "![velocity profiles](velocity_profiles.png)",
            "",
            "![overshoot](overshoot.png)",
            "",
            "![paths](paths.png)",
            "",
        ]
    else:
        lines += ["No flicks detected (no clicks preceded by a clear movement).", ""]

    lines += ["## Tracking", ""]
    if len(a.tracking):
        t = a.tracking
        lines += [
            f"| | median |", "|---|---|",
            f"| segments | {len(t)} (total {t.duration_ms.sum() / 1000:.1f} s) |",
            f"| mean speed (deg/s) | {t.mean_speed.median():.1f} |",
            f"| speed variability (CV) | {t.speed_cv.median():.2f} |",
            f"| smoothness, SPARC per 1 s (closer to 0 = smoother) | {t.sparc.median():.2f} |",
            f"| jitter power share {a.params.jitter_band_hz:g}–30 Hz | {t.jitter_hf_ratio.median():.1%} |",
            f"| jitter RMS (deg) | {t.jitter_rms_deg.median():.4f} |",
            "",
        ]
    else:
        lines += ["No sustained tracking segments (>= "
                  f"{a.params.min_track_ms:.0f} ms of continuous movement outside flicks).", ""]

    lines += ["## Holds", ""]
    if len(a.holds):
        h = a.holds
        pre = h[h.before_click]
        lines += [
            "| | all holds | holds right before a click |", "|---|---|---|",
            f"| count | {len(h)} | {len(pre)} |",
            f"| duration (ms) | {h.duration_ms.median():.0f} | {pre.duration_ms.median() if len(pre) else float('nan'):.0f} |",
            f"| jitter RMS (deg) | {h.jitter_rms_deg.median():.4f} | "
            f"{pre.jitter_rms_deg.median() if len(pre) else float('nan'):.4f} |",
            f"| drift RMS (deg) | {h.drift_rms_deg.median():.4f} | "
            f"{pre.drift_rms_deg.median() if len(pre) else float('nan'):.4f} |",
            "",
            f"For scale: one mouse count is {conv['yaw_deg_per_count']:.4f} deg at this sens.",
            "",
        ]
    else:
        lines += ["No holds detected.", ""]

    lines += ["Per-item data: `flicks.csv`, `tracking.csv`, `holds.csv`.", ""]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
