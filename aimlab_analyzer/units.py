"""Unit conversion for logged sessions.

Sign convention (screen space, same for every game):
  dx > 0, yaw_deg > 0     -> view turns right
  dy > 0, pitch_deg > 0   -> view turns down
Source/CS2 store yaw counter-clockwise (turning right decreases yaw), so the demo
parser will flip its sign to match this convention.
"""

from __future__ import annotations

import pandas as pd

CM_PER_INCH = 2.54


def counts_to_degrees(counts, deg_per_count: float):
    return counts * deg_per_count


def counts_to_cm(counts, dpi: float):
    return counts / dpi * CM_PER_INCH


def add_angles(
    df: pd.DataFrame, yaw_deg_per_count: float, pitch_deg_per_count: float, dpi: float
) -> pd.DataFrame:
    """Return a copy of a raw session frame with time in seconds, angles and distances.

    Adds: t_s, yaw_deg, pitch_deg (per-event rotation), yaw_cum, pitch_cum
    (cumulative view angle relative to session start), x_cm, y_cm (cumulative mouse
    position on the pad).
    """
    out = df.copy()
    out["t_s"] = out["t_ns"] / 1e9
    out["yaw_deg"] = counts_to_degrees(out["dx"], yaw_deg_per_count)
    out["pitch_deg"] = counts_to_degrees(out["dy"], pitch_deg_per_count)
    out["yaw_cum"] = out["yaw_deg"].cumsum()
    out["pitch_cum"] = out["pitch_deg"].cumsum()
    out["x_cm"] = counts_to_cm(out["dx"].cumsum(), dpi)
    out["y_cm"] = counts_to_cm(out["dy"].cumsum(), dpi)
    return out
