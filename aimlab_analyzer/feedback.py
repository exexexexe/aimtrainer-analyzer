"""Turn a run's stats and mouse metrics into plain findings, compared with your history."""

from __future__ import annotations

import numpy as np

from .kovaaks import Run

HISTORY_RUNS = 10   # "your usual" = median of this many previous runs of the scenario
MIN_HISTORY = 3

LEVEL_ORDER = {"bad": 0, "warn": 1, "good": 2, "info": 3}


def _f(level, title, detail=""):
    return {"level": level, "title": title, "detail": detail}


def _median(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(v)) if v else None


def stats_context(run: Run, history: list[Run]) -> dict:
    prev = [r for r in history if r.end < run.end]
    recent = prev[-HISTORY_RUNS:]
    scores = [r.score for r in prev]
    pb = max(scores) if scores else None
    return {
        "n_prev": len(prev),
        "pb_before": pb,
        "is_pb": pb is not None and run.score > pb,
        "usual_score": _median([r.score for r in recent]),
        "usual_accuracy": _median([r.accuracy for r in recent]),
        "usual_ttk": _median([r.avg_ttk for r in recent]),
        "percentile": float(np.mean([s < run.score for s in scores])) if scores else None,
    }


USUAL_KEYS = [
    ("flicks_all", "time_to_target_ms"), ("flicks_all", "primary_error_pct"),
    ("flicks_all", "corrections"), ("flicks_all", "click_delay_ms"),
    ("flicks_all", "clicked_moving"), ("flicks_all", "path_efficiency"),
    ("flicks_all", "offaxis_abs_pct"), ("phases", "correction_ms"), ("phases", "verify_ms"),
    ("misses", "moving_share"), ("click_pull", "median_deg"), ("holds", "jitter_rms_deg"),
    ("tracking", "jitter_rms_deg"), ("tracking", "jitter_hf_ratio"), ("tracking", "speed_cv"),
]


def mouse_usual(prev_mouse: list[dict]) -> dict | None:
    """Median of each mouse metric over previous recorded runs of the scenario."""
    prev_mouse = [m for m in prev_mouse if m][-HISTORY_RUNS:]
    if len(prev_mouse) < 2:
        return None
    out = {"n": len(prev_mouse)}
    for group, key in USUAL_KEYS:
        out.setdefault(group, {})[key] = _median([(m.get(group) or {}).get(key) for m in prev_mouse])
    return out


def _pct(x):
    return f"{x:.0%}"


def _side_median(directions, names):
    rows = [d for d in directions if d["name"] in names and d["n"] and d["along"] is not None]
    n = sum(d["n"] for d in rows)
    if n == 0:
        return None, 0
    return sum(d["along"] * d["n"] for d in rows) / n, n


def _flick_word(pe):
    return "underflick" if pe < 0 else "overflick"


def findings(run: Run, ctx: dict, mouse: dict | None, usual: dict | None) -> list[dict]:
    """Mechanical findings about how the hand moved. Score/accuracy live in the header."""
    if mouse is None:
        return [_f("info", "No mouse recording for this run",
                   "Start the dashboard with --record to get movement analysis for every run.")]
    out = []
    if run.kind == "click":
        _click_findings(out, mouse)
    else:
        _tracking_findings(out, mouse)
    _tension_findings(out, mouse, run.kind)
    if usual:
        _usual_findings(out, mouse, usual, run.kind)
    if not any(f["level"] in ("bad", "warn") for f in out):
        out.append(_f("good", "No mechanical problem stood out in this run"))
    return _sorted(out)


def _click_findings(out, m):
    fa = m["flicks_all"]
    n = fa.get("n") or 0
    pe = fa.get("primary_error_pct")

    # under/overflick, overall and per distance
    if n >= 8 and pe is not None and (pe <= -6 or pe >= 5):
        share = fa["undershoot_share"] if pe < 0 else fa["overshoot_share"]
        out.append(_f("warn", f"You {_flick_word(pe)}: first movement lands {abs(pe):.0f}% {'short' if pe < 0 else 'past'}",
                      f"Median of {n} flicks; {_pct(share)} land more than 5% {'short' if pe < 0 else 'past'}. "
                      f"{fa['corrections']:.1f} corrective movements per flick follow. "
                      + ("Commit to the full distance in one motion." if pe < 0
                         else "Start decelerating earlier or lower flick speed.")))
    for b, v in m["buckets"].items():
        bpe = v["primary_error_pct"]
        if v["n"] >= 5 and bpe is not None and pe is not None and abs(bpe - pe) >= 6 and (bpe <= -8 or bpe >= 6):
            out.append(_f("warn", f"{b.capitalize()} flicks: {_flick_word(bpe)} by {abs(bpe):.0f}%",
                          f"Your other flick distances differ (overall {pe:+.0f}%), so this is distance-specific "
                          f"({v['n']} flicks, avg {v['amplitude_deg']:.1f}°)."))

    # direction asymmetry
    dirs = m.get("directions") or []
    for (a_name, a_set), (b_name, b_set) in [
        (("left", {"left", "up-left", "down-left"}), ("right", {"right", "up-right", "down-right"})),
        (("up", {"up", "up-left", "up-right"}), ("down", {"down", "down-left", "down-right"})),
        (("horizontally", {"left", "right"}), ("vertically", {"up", "down"})),
    ]:
        a, na = _side_median(dirs, a_set)
        b, nb = _side_median(dirs, b_set)
        if na >= 4 and nb >= 4 and abs(a - b) >= 7:
            worse_name, worse, other_name, other = (a_name, a, b_name, b) if abs(a) > abs(b) else (b_name, b, a_name, a)
            out.append(_f("warn", f"Direction bias: you {_flick_word(worse)} {worse_name} ({worse:+.0f}%) more than {other_name} ({other:+.0f}%)",
                          f"Weighted median first-movement error by direction ({na}+{nb} flicks). "
                          "Asymmetries like this usually come from wrist vs arm range or "
                          "mouse grip angle on that side."))

    off = fa.get("offaxis_abs_pct")
    if n >= 8 and off is not None and off >= 8:
        out.append(_f("warn", f"Flicks curve off-line by {off:.0f}% of the distance",
                      "The first movement ends this far to the side of the straight line to the target "
                      f"(median). Path straightness {fa['path_efficiency']:.2f}."))

    # miss offset
    mb = m.get("miss_bias")
    if mb and mb["n"] >= 5:
        mx, my = mb["mean_x"], mb["mean_y"]
        where = " and ".join(w for w in [
            f"{abs(mx):.2f}° {'right' if mx > 0 else 'left'}" if abs(mx) >= 0.05 else "",
            f"{abs(my):.2f}° {'high' if my > 0 else 'low'}" if abs(my) >= 0.05 else ""] if w)
        if mb["consistency"] is not None and mb["consistency"] >= 0.4 and where:
            out.append(_f("warn", f"Misses lean {where}",
                          f"Average offset of {mb['n']} misses from where the target was; "
                          f"consistency {mb['consistency']:.2f} (1 = always the same side). "
                          "A steady lean is often crosshair placement or a grip that rotates the mouse."))
        else:
            out.append(_f("info", "Miss offsets are scattered, no consistent side",
                          f"{mb['n']} misses, median {mb['median_dist']:.2f}° from the target, "
                          f"consistency {mb['consistency']:.2f}. Misses come from timing/precision rather than a bias."))

    mi = m["misses"]
    kinds = {k: v for k, v in mi["by_kind"].items() if k in ("undershoot", "overshoot", "sideways")}
    if mi["n"] >= 5 and kinds:
        top, n_top = max(kinds.items(), key=lambda kv: kv[1])
        if n_top / mi["n"] >= 0.45:
            out.append(_f("warn", f"Missed shots are mostly {top}s ({n_top} of {mi['n']})",
                          {"undershoot": "You clicked before reaching the target and kept going the same way after.",
                           "overshoot": "You clicked past the target and came back after.",
                           "sideways": "You clicked beside the target's line and corrected sideways."}[top]))
    if mi["moving_share"] is not None and mi["n"] >= 5:
        hm = mi["hits_moving_share"] or 0.0
        if mi["moving_share"] >= 0.3 and mi["moving_share"] >= 1.5 * hm:
            out.append(_f("bad", f"{_pct(mi['moving_share'])} of misses were fired while the crosshair was still moving",
                          f"Only {_pct(hm)} of hits were. Those shots go out before the flick has settled."))

    ph = m.get("phases") or {}
    if ph.get("correction_ms") and ph.get("ballistic_ms") and n >= 8 and ph["correction_ms"] >= 0.5 * ph["ballistic_ms"]:
        out.append(_f("warn", f"Long correction phase: {ph['correction_ms']:.0f} ms after the flick",
                      f"The main flick takes {ph['ballistic_ms']:.0f} ms, then {ph['correction_ms']:.0f} ms of corrections. "
                      "Most of the time to target is spent fixing the first movement."))
    if ph.get("verify_ms") is not None and n >= 8 and ph["verify_ms"] >= 120:
        out.append(_f("warn", f"You wait {ph['verify_ms']:.0f} ms after stopping before clicking",
                      "That is confirmation time; it shrinks as the first movement becomes more reliable."))


def _tracking_findings(out, m):
    tr = m["tracking"]
    if tr["seconds"] < 3:
        out.append(_f("info", "Little continuous tracking detected",
                      f"{tr['seconds']:.1f} s of sustained movement, so tracking metrics are unreliable."))
        return
    if tr.get("jitter_hf_ratio") is not None and tr["jitter_hf_ratio"] >= 0.05:
        out.append(_f("warn", f"Shake while tracking: {tr['jitter_hf_ratio']:.0%} of movement is 8–30 Hz",
                      f"Jitter size {tr['jitter_rms_deg']:.3f}°. Often grip tension: relax the hand and let the arm drive."))
    if tr.get("speed_cv") is not None and tr["speed_cv"] >= 0.8:
        out.append(_f("warn", f"Stop-and-go tracking (speed variability {tr['speed_cv']:.2f})",
                      "Your crosshair speed swings a lot instead of matching the target smoothly; "
                      "often reactive catching-up rather than following."))
    sp = m.get("spectrum")
    if sp and sp.get("tremor_ratio") and sp["tremor_ratio"] >= 1.5:
        lvl = "warn" if sp["tremor_ratio"] >= 2 else "info"
        out.append(_f(lvl, f"Tremor peak at {sp['tremor_hz']:.0f} Hz ({sp['tremor_ratio']:.1f}× background)",
                      "Physiological tremor sits around 8–12 Hz and gets stronger with grip tension, "
                      "caffeine and fatigue."))


def _tension_findings(out, m, kind):
    cp = m.get("click_pull") or {}
    if kind == "click" and cp.get("n", 0) >= 8 and cp.get("median_deg") is not None:
        counts = cp["median_deg"] * cp.get("counts_per_deg", 0)
        if cp["share_over"] >= 0.3:
            out.append(_f("warn", f"Click pull: the crosshair moves when you press ({_pct(cp['share_over'])} of shots)",
                          f"On {_pct(cp['share_over'])} of shots made while still, the crosshair moved ≥{cp['threshold_deg']:.2f}° "
                          f"within 20 ms of the press (median {counts:.1f} counts, {cp['median_mm']:.3f} mm). "
                          "Usually pressing down/sideways with a tense finger or grip."))
        else:
            out.append(_f("good", f"Clean clicks: median pull {counts:.1f} counts",
                          f"Only {_pct(cp['share_over'])} of still shots moved ≥{cp['threshold_deg']:.2f}° on the press."))
    ch = (m.get("tension") or {}).get("change") or {}
    series = m.get("tension") or {}
    for key, label in (("hold_jitter", "Shake while holding still"), ("click_pull", "Click pull"),
                       ("track_jitter", "Tracking shake")):
        c = ch.get(key)
        windows = sum(v is not None for v in series.get(key, []))
        if c is not None and windows >= 4 and c >= 0.4:
            out.append(_f("warn", f"{label} grew {c:.0%} from the start to the end of the run",
                          "Tension or fatigue building up during the scenario. Check grip pressure mid-run."))
        elif c is not None and windows >= 4 and c <= -0.3:
            out.append(_f("info", f"{label} dropped {abs(c):.0%} over the run",
                          "You loosened up as the run went on; the start was tenser."))


def _usual_findings(out, m, usual, kind):
    if kind == "click":
        fa, fu = m["flicks_all"], usual["flicks_all"]
        _compare(out, fa.get("time_to_target_ms"), fu.get("time_to_target_ms"), rel=0.10,
                 worse="Slower to reach targets than your usual", better="Faster to reach targets than your usual",
                 fmt="{:.0f} ms", higher_is_worse=True)
        _compare(out, fa.get("corrections"), fu.get("corrections"), abs_min=0.25,
                 worse="More corrections than your usual", better="Fewer corrections than your usual",
                 fmt="{:.2f} per flick", higher_is_worse=True)
        _compare(out, (m.get("phases") or {}).get("verify_ms"), usual["phases"].get("verify_ms"), rel=0.30, abs_min=15,
                 worse="Waiting longer before clicking than usual", better="Clicking sooner after stopping than usual",
                 fmt="{:.0f} ms", higher_is_worse=True)
        _compare(out, (m.get("click_pull") or {}).get("median_deg"), usual["click_pull"].get("median_deg"),
                 rel=0.40, worse="More click pull than usual", better="Less click pull than usual",
                 fmt="{:.3f}°", higher_is_worse=True)
    else:
        tr, tu = m["tracking"], usual["tracking"]
        _compare(out, tr.get("jitter_rms_deg"), tu.get("jitter_rms_deg"), rel=0.25,
                 worse="Shakier tracking than your usual", better="Steadier tracking than your usual",
                 fmt="{:.3f}° jitter", higher_is_worse=True)
    _compare(out, (m.get("holds") or {}).get("jitter_rms_deg"), usual["holds"].get("jitter_rms_deg"), rel=0.25,
             worse="More shake while holding still than usual", better="Steadier holds than usual",
             fmt="{:.4f}°", higher_is_worse=True)


def _compare(out, value, usual, *, worse, better, fmt, higher_is_worse, rel=None, abs_min=0.0):
    if value is None or usual is None:
        return
    diff = value - usual
    threshold = max(abs_min, abs(usual) * rel if rel else 0.0)
    if abs(diff) < threshold or threshold == 0:
        return
    bad = (diff > 0) == higher_is_worse
    detail = f"{fmt.format(value)} vs your usual {fmt.format(usual)}."
    out.append(_f("warn" if bad else "good", worse if bad else better, detail))


def _sorted(items):
    return sorted(items, key=lambda f: LEVEL_ORDER[f["level"]])


def scenario_overview(runs: list[Run]) -> list[dict]:
    by = {}
    for r in runs:
        by.setdefault(r.scenario, []).append(r)
    rows = []
    for name, rs in by.items():
        rs.sort(key=lambda r: r.end)
        scores = [r.score for r in rs]
        last5, prev5 = scores[-5:], scores[-10:-5]
        trend = (np.median(last5) / np.median(prev5) - 1) if len(prev5) >= 3 and np.median(prev5) else None
        rows.append({
            "scenario": name, "kind": rs[-1].kind, "plays": len(rs),
            "pb": max(scores), "last": scores[-1], "last_played": rs[-1].end.isoformat(),
            "last_id": rs[-1].id, "trend": None if trend is None else float(trend),
            "accuracy": _median([r.accuracy for r in rs[-5:]]),
        })
    return sorted(rows, key=lambda r: r["last_played"], reverse=True)
