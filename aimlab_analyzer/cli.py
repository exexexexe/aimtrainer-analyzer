"""Command line entry point: python -m aimlab_analyzer <command>."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .session import BTN_LEFT, load_session


def cmd_show_config(args):
    cfg = load_config(args.config)
    print(f"config: {cfg.path}")
    print(f"output: {cfg.output_dir} ({cfg.output_format})")
    print(f"hotkeys: toggle={cfg.hotkeys['toggle']} quit={cfg.hotkeys['quit']}")
    for name in (cfg.raw.get("profiles") or {}):
        c = cfg.conversion(name)
        marker = "*" if name == cfg.active_profile else " "
        print(
            f"{marker} {name:<10} {c.yaw_deg_per_count:.5f} deg/count yaw, "
            f"{c.pitch_deg_per_count:.5f} pitch, {c.cm_per_360:.2f} cm/360 "
            f"@ {c.dpi:g} dpi  [{c.method}]"
        )
    return 0


def cmd_record(args):
    from .capture import get_backend
    from .recorder import Recorder

    cfg = load_config(args.config)
    conv = cfg.conversion(args.profile)
    backend = get_backend()
    watch_dir = None
    if args.continuous:
        from .kovaaks import find_stats_dir

        watch_dir = Path(args.watch_dir) if args.watch_dir else find_stats_dir(
            (cfg.raw.get("kovaaks") or {}).get("stats_dir"))
    rec = Recorder(
        cfg, conv, backend.NAME, duration_s=args.duration,
        chunk_s=args.chunk_minutes * 60 if args.continuous else None,
        stop_file=args.stop_file, watch_dir=watch_dir,
    )
    if args.continuous:
        args.autostart = True
        print("continuous mode: recording now; a file is saved after every KovaaK's run"
              + ("" if watch_dir else " (stats folder not found: saving every chunk only)"))

    print(f"profile {conv.profile}: {conv.yaw_deg_per_count:.5f} deg/count, "
          f"{conv.cm_per_360:.2f} cm/360 @ {conv.dpi:g} dpi")
    print(f"backend {backend.NAME}. {cfg.hotkeys['toggle']} = start/stop session, "
          f"{cfg.hotkeys['quit']} = quit, Ctrl+C also quits.")
    if args.autostart:
        rec.start()
    try:
        backend.run(rec, cfg.hotkeys)
    except KeyboardInterrupt:
        pass
    finally:
        rec.shutdown()
    return 0


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def cmd_inspect(args):
    conversion = load_config(args.config).conversion(args.profile) if args.profile else None
    df, meta = load_session(args.file, conversion)
    conv = meta["conversion"]
    dur = meta["duration_s"]
    n = len(df)

    dt_ms = np.diff(df["t_ns"].to_numpy()) / 1e6
    moves = df[(df.dx != 0) | (df.dy != 0)]
    move_dt = np.diff(moves["t_ns"].to_numpy()) / 1e6
    move_dt = move_dt[move_dt < 50]  # ignore pauses between movements

    path_deg = np.hypot(df.yaw_deg, df.pitch_deg).sum()
    path_cm = np.hypot(np.diff(df.x_cm, prepend=0), np.diff(df.y_cm, prepend=0)).sum()
    left = (df.buttons.to_numpy() & BTN_LEFT) > 0
    clicks = int(np.count_nonzero(left[1:] & ~left[:-1]) + (1 if n and left[0] else 0))

    # Angular speed on 5 ms bins, robust to timestamp jitter between single events.
    bins = (df.t_ns // 5_000_000).to_numpy()
    deg = np.hypot(df.yaw_deg, df.pitch_deg).to_numpy()
    binned = np.bincount(bins - bins.min(), weights=deg) / 0.005 if n else np.array([0.0])

    print(f"file:        {args.file}")
    print(f"started:     {meta['started_at']}  backend={meta['backend']}")
    print(f"conversion:  {conv['profile']} ({conv['method']}): {conv['yaw_deg_per_count']:.5f} deg/count, "
          f"{conv['cm_per_360']:.2f} cm/360 @ {conv['dpi']:g} dpi")
    print(f"duration:    {dur:.2f} s, {n} events ({n / dur:.0f}/s average incl. idle)")
    print(f"event gap:   median {_pct(move_dt, 50):.3f} ms, p99 {_pct(move_dt, 99):.3f} ms while moving "
          f"(~{1000 / _pct(move_dt, 50):.0f} Hz effective)" if len(move_dt) else "event gap:   n/a")
    if len(dt_ms) and dt_ms.min() < 0:
        print("WARNING: timestamps go backwards")
    print(f"travel:      {path_deg:.1f} deg of rotation, {path_cm:.1f} cm on the pad")
    print(f"net view:    yaw {df.yaw_cum.iloc[-1]:+.2f} deg, pitch {df.pitch_cum.iloc[-1]:+.2f} deg")
    print(f"peak speed:  {binned.max():.0f} deg/s (5 ms bins)")
    print(f"left clicks: {clicks}")
    print()
    cols = ["t_s", "dx", "dy", "buttons", "yaw_deg", "pitch_deg", "yaw_cum", "pitch_cum"]
    with np.printoptions(suppress=True):
        print(df[cols].head(args.rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.export:
        out = Path(args.export)
        df.to_csv(out, index=False)
        print(f"\nexported converted frame -> {out}")
    return 0


def cmd_analyze(args):
    from .kinematics import from_session
    from .metrics import analyze
    from .report import write_report
    from .segment import Params

    cfg = load_config(args.config)
    params = Params.from_config(cfg.raw.get("analysis"))
    conversion = cfg.conversion(args.profile) if args.profile else None
    for f in args.files:
        path = Path(f)
        df, meta = load_session(path, conversion)
        a = analyze(from_session(df), params)
        name = path.name.split(".")[0]
        out = Path(args.out) / name if args.out else cfg.path.parent / "reports" / name
        report = write_report(a, meta, out, name)
        fl = a.flicks
        print(f"{path.name}: {len(fl)} flicks, {len(a.tracking)} tracking segments, {len(a.holds)} holds")
        if len(fl):
            for b, g in fl.groupby("bucket", sort=False):
                print(f"  {b:<6} n={len(g):<4} amp {g.amplitude_deg.median():5.1f} deg  "
                      f"TTT {g.time_to_target_ms.median():4.0f} ms  "
                      f"primary {g.primary_error_pct.median():+5.1f}%  "
                      f"corr {g.n_corrections.mean():.2f}  eff {g.path_efficiency.median():.3f}")
        print(f"  report -> {report}")
    return 0


def cmd_dashboard(args):
    from .dashboard import serve
    from .kovaaks import find_stats_dir
    from .segment import Params

    cfg = load_config(args.config)
    stats_dir = find_stats_dir((cfg.raw.get("kovaaks") or {}).get("stats_dir"))
    if stats_dir is None:
        print("KovaaK's stats folder not found. Set kovaaks.stats_dir in config.yaml to the folder "
              "...\\steamapps\\common\\FPSAimTrainer\\FPSAimTrainer\\stats")
        return 1
    profile = args.profile or ("kovaaks" if "kovaaks" in (cfg.raw.get("profiles") or {}) else None)
    serve(cfg, stats_dir, Params.from_config(cfg.raw.get("analysis")), args.port,
          record=args.record, profile=profile, open_browser=not args.no_browser)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="aimlab_analyzer")
    p.add_argument("--config", help="path to config.yaml (default: project root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("record", help="log raw mouse input; hotkey starts/stops sessions")
    s.add_argument("--profile", help="config profile (default: active_profile)")
    s.add_argument("--autostart", action="store_true", help="start a session immediately")
    s.add_argument("--duration", type=float, help="stop and quit after N seconds of recording")
    s.add_argument("--continuous", action="store_true",
                   help="record all the time; save a file after each KovaaK's run and every --chunk-minutes")
    s.add_argument("--chunk-minutes", type=float, default=30.0)
    s.add_argument("--watch-dir", help="KovaaK's stats folder to watch (default: auto-detect)")
    s.add_argument("--stop-file", help=argparse.SUPPRESS)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("dashboard", help="local web dashboard of KovaaK's runs and findings")
    s.add_argument("--record", action="store_true", help="also record mouse input in the background")
    s.add_argument("--profile", help="recording profile (default: kovaaks if present)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("inspect", help="summarize a recorded session")
    s.add_argument("file")
    s.add_argument("--profile", help="re-convert using this config profile instead of the saved one")
    s.add_argument("--rows", type=int, default=10, help="rows of converted data to print")
    s.add_argument("--export", help="write the converted frame (with degree columns) to CSV")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("analyze", help="detect flicks/tracking/holds and write a report")
    s.add_argument("files", nargs="+")
    s.add_argument("--profile", help="re-convert using this config profile instead of the saved one")
    s.add_argument("--out", help="output folder (default: reports/ next to config.yaml)")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("show-config", help="print resolved deg/count and cm/360 per profile")
    s.set_defaults(func=cmd_show_config)

    args = p.parse_args(argv)
    return args.func(args)
