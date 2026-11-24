import ctypes
import sys

import numpy as np
import pytest

from aimlab_analyzer.capture import update_buttons
from aimlab_analyzer.config import resolve_conversion
from aimlab_analyzer.hotkeys import parse_hotkey
from aimlab_analyzer.recorder import Recorder
from aimlab_analyzer.session import BTN_LEFT, BTN_RIGHT, load_session


def test_cs2_sens_conversion():
    c = resolve_conversion("cs2", {"sens": 1.2, "yaw": 0.022, "pitch": 0.022}, dpi=800)
    assert c.method == "sens*yaw"
    assert c.yaw_deg_per_count == pytest.approx(0.0264)
    # 100 counts at sens 1.2 -> 2.64 degrees
    assert 100 * c.yaw_deg_per_count == pytest.approx(2.64)
    # 800 dpi, 1.2 sens: 360 / 0.0264 = 13636 counts = 17.05 in = 43.30 cm
    assert c.cm_per_360 == pytest.approx(43.295, abs=0.01)


def test_cm_per_360_wins_over_sens():
    c = resolve_conversion(
        "kovaaks", {"sens": 1.2, "yaw": 0.022, "cm_per_360": 30.0, "pitch_ratio": 0.8}, dpi=1600
    )
    assert c.method == "cm_per_360"
    assert c.cm_per_360 == pytest.approx(30.0)
    assert c.pitch_deg_per_count == pytest.approx(c.yaw_deg_per_count * 0.8)


def test_explicit_deg_per_count_and_missing_fields():
    c = resolve_conversion("x", {"deg_per_count": 0.05}, dpi=400)
    assert c.yaw_deg_per_count == 0.05 and c.pitch_deg_per_count == 0.05
    with pytest.raises(ValueError):
        resolve_conversion("x", {"sens": 1.0}, dpi=400)


def test_update_buttons():
    s = update_buttons(0, {BTN_LEFT: True})
    s = update_buttons(s, {BTN_RIGHT: True})
    assert s == BTN_LEFT | BTN_RIGHT
    assert update_buttons(s, {BTN_LEFT: False}) == BTN_RIGHT


def test_parse_hotkey():
    assert parse_hotkey("F8") == (frozenset(), "F8")
    assert parse_hotkey("ctrl+Alt+f9") == (frozenset({"ctrl", "alt"}), "F9")
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+hyper+F1")


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_record_save_load_roundtrip(tmp_path, fmt):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "mouse: {dpi: 800}\n"
        f"output: {{dir: out, format: {fmt}}}\n"
        "profiles: {cs2: {sens: 1.0, yaw: 0.022, pitch: 0.022}}\n"
    )
    from aimlab_analyzer.config import load_config

    cfg = load_config(cfg_path)
    rec = Recorder(cfg, cfg.conversion("cs2"), "test", log=lambda *_: None)
    t0 = 1_000_000_000
    rec.start(t0)
    # 100 events at 1 kHz, 10 counts right each, click half way
    for i in range(100):
        rec.on_event(t0 + (i + 1) * 1_000_000, 10, -2, BTN_LEFT if i >= 50 else 0)
    path = rec.stop(t0 + 101_000_000)

    df, meta = load_session(path)
    assert len(df) == 100 and meta["events"] == 100
    assert df.t_ns.iloc[0] == 1_000_000
    assert df.yaw_cum.iloc[-1] == pytest.approx(1000 * 0.022)
    assert df.pitch_cum.iloc[-1] == pytest.approx(-200 * 0.022)
    assert df.x_cm.iloc[-1] == pytest.approx(1000 / 800 * 2.54)
    assert np.count_nonzero(df.buttons & BTN_LEFT) == 50


def test_empty_session_not_saved(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("mouse: {dpi: 800}\noutput: {dir: out}\nprofiles: {cs2: {sens: 1, yaw: 0.022}}\n")
    from aimlab_analyzer.config import load_config

    cfg = load_config(cfg_path)
    rec = Recorder(cfg, cfg.conversion("cs2"), "test", log=lambda *_: None)
    rec.start(0)
    assert rec.stop(10) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows struct layout")
def test_rawmouse_layout_matches_win32():
    from aimlab_analyzer.capture import windows_raw as w

    assert ctypes.sizeof(w.RAWMOUSE) == 24
    assert w.RAWMOUSE.u.offset == 4
    assert w.RAWMOUSE.lLastX.offset == 12
    assert w.RAWINPUT.mouse.offset == ctypes.sizeof(w.RAWINPUTHEADER)
