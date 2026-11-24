import time

import numpy as np
import pytest

from aimlab_analyzer.feedback import findings, mouse_usual, stats_context
from aimlab_analyzer.kovaaks import parse_stats, run_conversion
from aimlab_analyzer.metrics import analyze
from aimlab_analyzer.runs import classify_misses, estimate_offset_ms, label_clicks
from aimlab_analyzer.segment import Params
from aimlab_analyzer.synth import Builder

STATS = """Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits,Accuracy,Damage Done,Damage Possible,Efficiency,Cheated,OverShots
1,23:59:59.500,bot,pistol,0.228000s,2,1,0.5,1,2,0.5,0,0
2,00:00:00.250,bot,pistol,0.000000s,1,1,1.0,1,1,1.0,0,0

Weapon,Shots,Hits,Damage Done,Damage Possible,,Sens Scale,Horiz Sens
pistol,3,2,2.0,3.0,

Kills:,2
Avg TTK:,0.114
Total Overshots:,0
Hit Count:,2
Miss Count:,1
Score:,123.5
Scenario:,Test - Scenario
Challenge Start:,23:59:58.000
Sens Scale:,Counter-Strike
Horiz Sens:,0.28008
Vert Sens:,0.28008
DPI:,2800
FOV:,103.0
"""


@pytest.fixture
def stats_file(tmp_path):
    p = tmp_path / "Test - Scenario - Challenge - 2026.11.25-00.00.01 Stats.csv"
    p.write_text(STATS, encoding="utf-8")
    return p


def test_parse_stats_across_midnight(stats_file):
    r = parse_stats(stats_file)
    assert r.scenario == "Test - Scenario"
    assert r.start.isoformat() == "2026-11-24T23:59:58"
    assert r.duration_s == pytest.approx(3.0)
    np.testing.assert_allclose(r.kill_times, [1.5, 2.25])
    assert (r.shots, r.hits, r.kills, r.score) == (3, 2, 2, 123.5)
    assert r.accuracy == pytest.approx(2 / 3)
    assert r.kind == "click"


def test_run_conversion_counter_strike(stats_file):
    c = run_conversion(parse_stats(stats_file))
    assert c.yaw_deg_per_count == pytest.approx(0.28008 * 0.022)
    assert c.cm_per_360 == pytest.approx(53.0, abs=0.01)


def test_estimate_offset_and_labels():
    clicks = np.array([100.0, 400.0, 520.0, 900.0, 1300.0])  # ms, recorder clock
    # kills logged 7 ms after the 2nd, 3rd and 5th clicks on the game clock (+250 ms skew)
    kills = np.array([407.0, 527.0, 1307.0]) + 250
    off, matched = estimate_offset_ms(clicks, kills)
    assert off == pytest.approx(257, abs=3) and matched == 3
    labels = label_clicks(clicks, kills - off, np.array([1, 1, 1]))
    assert labels.tolist() == ["miss", "kill", "kill", "miss", "kill"]


def test_multi_hit_targets_leave_non_lethal_shots_unknown():
    labels = label_clicks(np.array([0.0, 100, 200]), np.array([200.0]), np.array([3]))
    assert labels.tolist() == ["unknown", "unknown", "kill"]


@pytest.mark.parametrize("second_move, expected", [((3, 0), "undershoot"), ((-2, 0), "overshoot"), ((0, 2), "sideways")])
def test_classify_misses(second_move, expected):
    b = Builder().hold(300)
    b.move((10, 0), 150).hold(80).click()          # miss
    b.hold(100).move(second_move, 120).hold(80).click().hold(300)  # correction then kill
    a = analyze(b.trajectory())
    clicks = b.trajectory().clicks
    m = classify_misses(a.kin, clicks, np.array(["miss", "kill"], dtype=object), a.spans, Params())
    assert m.kind.tolist() == [expected]
    assert not m.moving.iloc[0]


def _mouse(**over):
    m = {
        "misses": {"n": 10, "by_kind": {"overshoot": 6, "undershoot": 2, "sideways": 2},
                   "moving_share": 0.5, "hits_moving_share": 0.1, "median_distance_deg": 0.8},
        "flicks_all": {"n": 30, "primary_error_pct": -12.0, "undershoot_share": 0.6, "overshoot_share": 0.1,
                       "corrections": 0.9, "offaxis_abs_pct": 10.0, "path_efficiency": 0.85,
                       "time_to_target_ms": 500, "click_delay_ms": 90, "clicked_moving": 0.2},
        "buckets": {"small": {"n": 2, "primary_error_pct": 0, "corrections": 0.2, "amplitude_deg": 2},
                    "medium": {"n": 12, "primary_error_pct": -14.0, "corrections": 0.9, "amplitude_deg": 8},
                    "large": {"n": 0, "primary_error_pct": None, "corrections": None, "amplitude_deg": None}},
        "directions": [{"name": "left", "n": 6, "along": -20.0, "perp": 0}, {"name": "right", "n": 6, "along": -2.0, "perp": 0},
                       {"name": "up", "n": 5, "along": -10.0, "perp": 0}, {"name": "down", "n": 5, "along": -9.0, "perp": 0}],
        "phases": {"reaction_ms": 150, "ballistic_ms": 200, "correction_ms": 150, "verify_ms": 140},
        "miss_bias": {"n": 8, "mean_x": 0.3, "mean_y": -0.2, "median_dist": 0.5, "consistency": 0.7},
        "click_pull": {"n": 20, "median_deg": 0.04, "median_mm": 0.06, "share_over": 0.5,
                       "threshold_deg": 0.03, "counts_per_deg": 162},
        "tension": {"t": [5, 15, 25, 35, 45, 55], "hold_jitter": [0.01, 0.01, 0.012, 0.015, 0.016, 0.02],
                    "click_pull": [None] * 6, "track_jitter": [None] * 6,
                    "change": {"hold_jitter": 0.8, "click_pull": None, "track_jitter": None}},
        "tracking": {"seconds": 0},
        "holds": {"jitter_rms_deg": 0.012},
    }
    m.update(over)
    return m


def test_findings_are_mechanical(stats_file):
    run = parse_stats(stats_file)
    ctx = stats_context(run, [run])
    usual = {"flicks_all": {"time_to_target_ms": 400, "corrections": 0.5}, "phases": {"verify_ms": 60},
             "click_pull": {"median_deg": 0.02}, "holds": {"jitter_rms_deg": 0.012}, "tracking": {}}
    out = findings(run, ctx, _mouse(), usual)
    titles = [f["title"] for f in out]
    assert any(t.startswith("You underflick: first movement lands 12% short") for t in titles)
    assert not any(t.startswith("Medium flicks") for t in titles)  # -14 vs -12 overall: not distance-specific
    assert any("Direction bias: you underflick left (-20%) more than right" in t for t in titles)
    assert any("curve off-line by 10%" in t for t in titles)
    assert any(t.startswith("Misses lean 0.30° right and 0.20° low") for t in titles)
    assert any("mostly overshoots (6 of 10)" in t for t in titles)
    assert any("fired while the crosshair was still moving" in t for t in titles)
    assert any(t.startswith("Long correction phase") for t in titles)
    assert any(t.startswith("You wait 140 ms") for t in titles)
    assert any(t.startswith("Click pull: the crosshair moves") for t in titles)
    assert any(t.startswith("Shake while holding still grew 80%") for t in titles)
    assert any(t.startswith("Slower to reach targets") for t in titles)
    assert any(t.startswith("More click pull than usual") for t in titles)
    assert not any("Accuracy" in t or "Score" in t or "personal best" in t for t in titles)
    levels = [f["level"] for f in out]
    assert levels == sorted(levels, key=["bad", "warn", "good", "info"].index)


def test_findings_scattered_misses_and_clean_clicks(stats_file):
    run = parse_stats(stats_file)
    m = _mouse(miss_bias={"n": 8, "mean_x": 0.05, "mean_y": 0.0, "median_dist": 0.5, "consistency": 0.1},
               click_pull={"n": 20, "median_deg": 0.01, "median_mm": 0.015, "share_over": 0.1,
                           "threshold_deg": 0.03, "counts_per_deg": 162})
    titles = [f["title"] for f in findings(run, stats_context(run, [run]), m, None)]
    assert "Miss offsets are scattered, no consistent side" in titles
    assert any(t.startswith("Clean clicks") for t in titles)


def test_findings_without_mouse(stats_file):
    run = parse_stats(stats_file)
    out = findings(run, stats_context(run, [run]), None, None)
    assert [f["title"] for f in out] == ["No mouse recording for this run"]


def test_mouse_usual_needs_two_runs():
    assert mouse_usual([_mouse()]) is None
    u = mouse_usual([_mouse(), _mouse(flicks_all={"time_to_target_ms": 300})])
    assert u["flicks_all"]["time_to_target_ms"] == 400
    assert u["click_pull"]["median_deg"] == pytest.approx(0.04)


# ---------- technical measures ----------

def test_reaction_times_only_after_kills():
    from aimlab_analyzer.runs import reaction_times
    from aimlab_analyzer.segment import FlickSpan

    spans = [FlickSpan(onset=o, peak=o + 50, primary_end=o + 100, settle=o + 150, click=o + 200, submovements=[], primary=0)
             for o in (300, 900)]
    clicks = np.array([100, 500, 700, 1100])
    out = reaction_times(spans, clicks, np.array(["kill", "kill", "miss", "kill"], dtype=object), dt=0.001)
    assert out == [200.0, None]   # second flick follows a miss


def test_click_pull_measures_press_movement_only():
    from aimlab_analyzer.runs import click_pull

    pos = np.zeros((2000, 2))
    pos[510:, 0] += 0.06           # crosshair jumps 0.06 deg 10 ms after the press at 500
    pos[1310:, 0] += 0.06          # same at 1300, but the next flick peaks 50 ms later
    out = click_pull(pos, np.array([500, 1300]), 0.001, flick_peaks=[1350])
    assert len(out) == 1
    assert out.pull_deg.iloc[0] == pytest.approx(0.06)


def test_velocity_spectrum_finds_tremor():
    from aimlab_analyzer.runs import velocity_spectrum

    rng = np.random.default_rng(0)
    t = np.arange(6000) * 0.001
    pos = np.column_stack([10 * np.sin(2 * np.pi * 0.5 * t) + 0.01 * np.sin(2 * np.pi * 10 * t)
                           + np.cumsum(rng.normal(0, 0.002, len(t))), np.zeros(len(t))])
    sp = velocity_spectrum(pos, [(0, len(t))], 0.001, nper=1024)
    assert sp["tremor_hz"] == pytest.approx(10, abs=1)
    assert sp["tremor_ratio"] > 2


def test_miss_offset_only_when_next_kill_is_same_target():
    b = Builder().hold(300)
    b.move((10, 0), 150).hold(80).click()                 # miss, 10 deg flick
    b.hold(100).move((1, -0.5), 120).hold(80).click()     # small correction -> same target, kill
    b.hold(200).move((-8, 0), 150).hold(80).click()       # miss
    b.hold(100).move((6, 3), 150).hold(80).click().hold(300)  # big move -> different target
    a = analyze(b.trajectory())
    labels = np.array(["miss", "kill", "miss", "kill"], dtype=object)
    m = classify_misses(a.kin, b.trajectory().clicks, labels, a.spans, Params())
    assert m.offset_x.iloc[0] == pytest.approx(-1, abs=0.05)   # landed 1 deg left of target
    assert m.offset_y.iloc[0] == pytest.approx(-0.5, abs=0.05)  # and 0.5 deg low (pitch down was +)
    assert np.isnan(m.offset_x.iloc[1])


def test_recorder_continuous_rotates_on_new_stats_file(tmp_path):
    from aimlab_analyzer.config import load_config
    from aimlab_analyzer.recorder import Recorder

    (tmp_path / "config.yaml").write_text(
        "mouse: {dpi: 800}\noutput: {dir: out, format: csv}\nprofiles: {k: {sens: 1, yaw: 0.022}}\n")
    cfg = load_config(tmp_path / "config.yaml")
    watch = tmp_path / "stats"
    watch.mkdir()
    stop = tmp_path / "stop"
    rec = Recorder(cfg, cfg.conversion("k"), "test", log=lambda *_: None,
                   chunk_s=1800, stop_file=stop, watch_dir=watch)
    t = time.perf_counter_ns()
    rec.start(t)
    rec.on_event(t + 1_000_000, 5, 0, 0)
    assert not rec.should_exit(t + 2_000_000)
    time.sleep(0.02)
    (watch / "run Stats.csv").write_text("x")       # KovaaK's finishes a run
    rec._next_poll = 0
    assert not rec.should_exit(t + 3_000_000)
    assert len(rec.saved) == 1 and rec.active        # saved and kept recording
    rec.on_event(t + 4_000_000, 1, 1, 1)
    stop.touch()
    rec._next_poll = 0
    assert rec.should_exit(t + 5_000_000)
    rec.shutdown()
    assert len(rec.saved) == 2
