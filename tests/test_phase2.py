import numpy as np
import pytest

from aimlab_analyzer.kinematics import fill_gaps, from_session, runs, sparc, split_at_valleys
from aimlab_analyzer.metrics import analyze
from aimlab_analyzer.segment import Params, bucket_of
from aimlab_analyzer.synth import Builder
from aimlab_analyzer.units import add_angles


def one_flick(**kw):
    b = Builder().hold(300)
    b.flick(**kw)
    b.hold(300)
    return analyze(b.trajectory())


# ---------- helpers ----------

def test_runs_and_fill_gaps():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert runs(m).tolist() == [[1, 3], [5, 6], [7, 10]]
    assert runs(fill_gaps(m, 1)).tolist() == [[1, 3], [5, 10]]
    assert runs(fill_gaps(m, 2)).tolist() == [[1, 10]]


def test_split_at_valleys_separates_two_bumps():
    t = np.linspace(0, 1, 200)
    speed = np.exp(-((t - 0.3) / 0.08) ** 2) + 0.6 * np.exp(-((t - 0.7) / 0.08) ** 2) + 0.05
    pieces = split_at_valleys(speed, 0, 200, 0.6)
    assert len(pieces) == 2
    assert 90 < pieces[0][1] < 120


def test_bucket_edges():
    assert [bucket_of(a, (5, 15)) for a in (1, 5, 14.9, 30)] == ["small", "medium", "medium", "large"]


# ---------- flicks ----------

def test_clean_flick_metrics():
    a = one_flick(vec=(10, 0), ms=150)
    assert len(a.flicks) == 1
    f = a.flicks.iloc[0]
    assert f.amplitude_deg == pytest.approx(10, abs=0.05)
    # minimum-jerk peak = 1.875 * A / T; smoothing shaves a little off
    assert f.peak_velocity == pytest.approx(1.875 * 10 / 0.150, rel=0.05)
    assert f.time_to_peak_ms == pytest.approx(75, abs=6)
    assert f.n_corrections == 0
    assert abs(f.primary_error_pct) < 2
    assert f.path_efficiency > 0.99
    assert f.bucket == "medium"
    assert not f.clicked_moving


@pytest.mark.parametrize("gain", [0.8, 1.15])
def test_under_and_overshoot_with_correction(gain):
    a = one_flick(vec=(20, 0), ms=180, primary_gain=gain, correction_ms=120)
    f = a.flicks.iloc[0]
    assert f.amplitude_deg == pytest.approx(20, abs=0.1)
    assert f.primary_error_pct == pytest.approx((gain - 1) * 100, abs=2.5)
    assert f.n_corrections == 1
    assert f.correction_time_ms > 80
    if gain > 1:
        assert f.max_overshoot_pct == pytest.approx(15, abs=1.5)
        assert f.path_efficiency < 0.8  # went 23 deg and came back 3 to cover 20
    assert f.bucket == "large"


def test_blended_correction_is_split_at_valley():
    # correction starts 40 ms before the primary ends, so speed never drops to rest
    a = one_flick(vec=(12, 0), ms=160, primary_gain=0.75, correction_ms=140, overlap_ms=40)
    f = a.flicks.iloc[0]
    assert f.n_corrections == 1
    assert f.primary_error_pct < -15


def test_diagonal_flick_direction_and_offaxis():
    # pitch negative = up, so (3, -4) is up-right at 53.13 deg
    a = one_flick(vec=(3, -4), ms=140)
    f = a.flicks.iloc[0]
    assert f.amplitude_deg == pytest.approx(5, abs=0.05)
    assert f.direction_deg == pytest.approx(53.13, abs=0.5)
    assert abs(f.primary_offaxis_pct) < 1


def test_curved_path_lowers_efficiency():
    b = Builder().hold(300)
    b.move((5, -5), 120).move((5, 5), 120, overlap_ms=60).hold(80).click().hold(300)
    f = analyze(b.trajectory()).flicks.iloc[0]
    assert f.amplitude_deg == pytest.approx(10, abs=0.1)
    assert 0.6 < f.path_efficiency < 0.95


def test_multiple_flicks_and_skipped_clicks():
    b = Builder().hold(300)
    for vec in [(3, 0), (-8, 2), (25, -5), (-2, -1)]:
        b.flick(vec=vec, ms=150).hold(250)
    b.click().hold(100)                 # double click, no movement
    b.move((0.2, 0), 150).hold(50).click().hold(300)  # tiny slow adjust
    b.hold(900).click().hold(300)       # click long after any movement
    a = analyze(b.trajectory())
    assert a.flicks.bucket.tolist() == ["small", "medium", "large", "small"]
    assert sum(a.skipped.values()) == 3


def test_click_while_moving_is_flagged():
    b = Builder().hold(300).move((10, 0), 150)
    b.cursor -= 30  # click 30 ms before the move ends
    b.click()
    b.cursor += 30
    b.hold(300)
    f = analyze(b.trajectory()).flicks.iloc[0]
    assert f.clicked_moving
    assert f.click_delay_ms == 0


def test_raw_session_roundtrip_with_quantization():
    """Same flicks through the logger format (integer counts at a real sens)."""
    dpc = 0.00616  # 0.28 sens CS2
    b = Builder().hold(300)
    for vec in [(4, 1), (-15, 3), (30, -2)]:
        b.flick(vec=vec, ms=170, primary_gain=0.9, correction_ms=110).hold(300)
    df = add_angles(b.session_frame(dpc), dpc, dpc, 2800)
    a = analyze(from_session(df))
    assert len(a.flicks) == 3
    np.testing.assert_allclose(a.flicks.amplitude_deg, [np.hypot(4, 1), np.hypot(15, 3), np.hypot(30, 2)], atol=0.1)
    np.testing.assert_allclose(a.flicks.primary_error_pct, -10, atol=3)
    assert (a.flicks.n_corrections == 1).all()
    assert (a.flicks.path_efficiency > 0.97).all()


# ---------- tracking and holds ----------

def test_tracking_segments_and_jitter():
    smooth = Builder(seed=1).hold(300).track(3000).hold(300)
    noisy = Builder(seed=1).hold(300).track(3000, noise_deg=0.03).hold(300)
    a_s, a_n = analyze(smooth.trajectory()), analyze(noisy.trajectory())
    assert len(a_s.tracking) == 1 and len(a_n.tracking) == 1
    assert a_s.tracking.duration_ms.iloc[0] == pytest.approx(3000, abs=150)
    assert a_s.flicks.empty
    s, n = a_s.tracking.iloc[0], a_n.tracking.iloc[0]
    assert n.jitter_rms_deg > 5 * s.jitter_rms_deg
    assert n.jitter_hf_ratio > s.jitter_hf_ratio
    assert s.jitter_hf_ratio < 0.01


def test_tracking_velocity_error_with_target():
    from aimlab_analyzer.metrics import tracking_metrics
    from aimlab_analyzer.segment import compute_kinematics

    b = Builder().hold(300).track(2000).hold(300)
    kin = compute_kinematics(b.trajectory(), Params())
    s, e = 400, 2200
    perfect = tracking_metrics(kin, s, e, Params(), target_vel=kin.vel[s:e])
    slow = tracking_metrics(kin, s, e, Params(), target_vel=kin.vel[s:e] * 1.25)
    assert perfect["velocity_rmse"] == pytest.approx(0, abs=1e-9)
    assert perfect["speed_gain"] == pytest.approx(1)
    assert slow["speed_gain"] == pytest.approx(0.8, abs=0.01)
    assert slow["velocity_rmse"] > 1


def test_sparc_prefers_single_bump():
    t = np.linspace(0, 1, 1000)
    one = np.sin(np.pi * t) ** 2
    two = np.abs(np.sin(3 * np.pi * t))
    assert sparc(one, 1000) > sparc(two, 1000)


def test_hold_jitter():
    b = Builder(seed=2).hold(300).hold(800, jitter_deg=0.02).hold(400, jitter_deg=0.002)
    a = analyze(b.trajectory(), Params(hold_threshold=1000, max_hold_ms=5000))
    # one long still segment covering everything; compare halves via metrics directly
    from aimlab_analyzer.metrics import hold_metrics

    shaky = hold_metrics(a.kin, 300, 1100, a.params)
    steady = hold_metrics(a.kin, 1100, 1500, a.params)
    assert shaky["jitter_rms_deg"] > 5 * steady["jitter_rms_deg"]
    assert not shaky["before_click"]
