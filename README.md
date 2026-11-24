# aimlab-analyzer

Log raw mouse movement while playing CS2 or training in KovaaK's, then turn it into
per-flick and per-track metrics. Later phases compare those metrics with pro players
from CS2 demos. All metrics are in degrees of view rotation, so they don't depend on sens.

| Phase | Status |
|---|---|
| 1. Logger (raw input, sessions, unit conversion) | done |
| 2. Flick/track segmentation and metrics | done |
| 3. KovaaK's dashboard: per-run stats, hit/miss analysis, findings, progress | done |
| 4. CS2: pro baseline from demos, comparison | later |
| 5. Hand/grip webcam layer (optional) | later |

## Quick start: the dashboard

```powershell
C:\Users\slopp\aimlab-analyzer\aim.bat dashboard --record
```

That command opens `http://localhost:8765` in your browser and records your mouse in
the background. Play KovaaK's as usual. When a scenario ends, the dashboard switches to
that run. It focuses on **how your hand moved**; score and accuracy are a single line,
since KovaaK's already shows those.

**Mechanics findings** are plain statements with numbers, for example: you underflick
(first movement lands 12% short), a direction bias (underflicking left more than right,
or horizontally more than vertically), flicks curving off the line, misses leaning to one
side, misses being mostly overshoots, firing while still moving, a long correction phase,
waiting before the click, click pull, shake growing during the run, and a tremor peak.

**Click scenarios**
- **Where the time goes**: the median flick split into kill → next move, the main
  (ballistic) flick, corrections, and the wait before clicking.
- **Click pull**: how far the crosshair moves in the 20 ms after you press, when you
  were holding still, in counts and mm. A tense finger or grip drags the mouse.
- **Flick landing**: where each flick's first movement stops relative to the target, as
  a % of flick distance. Short or past means under- or overflicking; above or below
  means off to the side of the path.
- **Miss offset**: where each miss landed relative to where the target was, in degrees
  on screen, with the average marked. Only misses followed by a nearby kill on the same
  target count.
- **Under / overflick by direction**: in 8 directions.
- **Time to target vs distance**, the **flick speed profile**, and **tension over the
  run** (shake while holding still and click pull, per 10 s window).
- **Flick breakdown** by distance: kill → next move, time to target, peak speed and its
  timing, landing error, under/overflick shares, sideways offset, corrections,
  settle → click, clicked while moving, and path straightness.

**Tracking scenarios**: 8–30 Hz shake, speed variability, smoothness (SPARC), a movement
spectrum with the tremor band (8–12 Hz) marked and the tremor peak found, plus shake
over the run.

KovaaK's kill log tells us which clicks killed, so every other click was a miss. With
multi-hit targets, shots that hit without killing can't be told from misses and are left
out. A run can be opened directly with `http://localhost:8765/#run=<id>`. The address
bar updates as you click runs, so you can bookmark one.

Runs without a mouse recording (including your older history) still show the score
and accuracy parts. Without `--record`, the dashboard is read-only.

Stop with **Ctrl+C** in the dashboard window. This saves the recording; closing the
window with X loses the part of the recording not yet saved. While recording, a file
is saved after every finished KovaaK's run and at least every 30 minutes.

Sens is read per run from KovaaK's own stats, so the 53 and 40 cm/360 profiles are
both handled automatically. Supported scales: Counter-Strike/Source/Apex, Overwatch,
Valorant, cm/360. Runs on other scales (such as The FINALS) show stats only.

The KovaaK's stats folder is found automatically under Steam. If it isn't, set
`kovaaks.stats_dir` in `config.yaml`.

## Setup

Requires Python 3.10+ (tested on 3.14, Windows 10).

```powershell
cd C:\Users\slopp\aimlab-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`.

On Windows you can skip activating the venv: `aim.bat` runs the tool with it from any
folder. For example `C:\Users\slopp\aimlab-analyzer\aim.bat record`, or `.\aim record`
inside the project folder. Any `python -m aimlab_analyzer ...` below can be `.\aim ...`.
`ModuleNotFoundError: No module named 'yaml'` means you used the system Python instead
of the venv.

### Config

Edit `config.yaml`: set your mouse DPI, your CS2 sens, and how KovaaK's converts sens.
Then check what it resolves to:

```powershell
python -m aimlab_analyzer show-config
```
```
* cs2        0.02640 deg/count yaw, 0.02640 pitch, 43.30 cm/360 @ 800 dpi  [sens*yaw]
  kovaaks    0.02640 deg/count yaw, 0.02640 pitch, 43.30 cm/360 @ 800 dpi  [sens*yaw]
```

Conversion rules, in priority order for each profile:

1. `deg_per_count`: used as given.
2. `cm_per_360`: `360 / (cm_per_360 / 2.54 * dpi)`. Easiest for KovaaK's, which shows cm/360 in its settings.
3. `sens` and `yaw`: `sens * yaw`. CS2 uses `m_yaw = 0.022`. This also works for KovaaK's with Sensitivity Scale set to "CSGO".

The cm/360 printed by `show-config` should match KovaaK's. If it doesn't, the config is wrong.

## Phase 1: recording

```powershell
python -m aimlab_analyzer record                    # uses active_profile
python -m aimlab_analyzer record --profile kovaaks
```

- **F8** starts or stops a session. You can do several sessions in one run, and each is saved when you stop it.
- **F10** (or Ctrl+C in the terminal) quits and saves any running session.
- The hotkeys are global and work while the game has focus. The game doesn't see these keys while the logger runs.
- `--autostart` starts recording right away. `--duration 60` stops and exits after 60 s.

Each session writes two files to `sessions/`:

`20261124-165625_cs2.parquet` (or `.csv`) contains one row per mouse report:

| column | meaning |
|---|---|
| `t_ns` | ns since session start (host clock when the event arrived) |
| `dx`, `dy` | raw counts. + = right / + = down (mouse pulled toward you) |
| `buttons` | bitmask: 1 left, 2 right, 4 middle, 8 mouse4, 16 mouse5 |

`20261124-165625_cs2.meta.json` holds the start time, backend, platform, and the
**resolved conversion used for this session** (dpi, deg/count, cm/360). If you change
sens later, old sessions keep their own values.

### Inspecting a session

```powershell
python -m aimlab_analyzer inspect sessions\20261124-165625_cs2.parquet
```
```
duration:    1.72 s, 1637 events (954/s average incl. idle)
event gap:   median 0.993 ms, p99 2.032 ms while moving (~1007 Hz effective)
travel:      145.5 deg of rotation, 17.5 cm on the pad
net view:    yaw -16.32 deg, pitch +21.52 deg
peak speed:  224 deg/s (5 ms bins)
left clicks: 0

   t_s  dx  dy  buttons  yaw_deg  pitch_deg  yaw_cum  pitch_cum
0.0003   0   2        0   0.0000     0.0528   0.0000     0.0528
...
```

Options:
- `--profile NAME` re-converts with a profile from the current config, for example if you recorded with the wrong sens.
- `--export out.csv` writes the converted frame with columns `t_s, yaw_deg, pitch_deg, yaw_cum, pitch_cum, x_cm, y_cm`.

In code: `from aimlab_analyzer.session import load_session; df, meta = load_session(path)`.

Sign convention everywhere: positive yaw = turn right, positive pitch = look down.

## Phase 2: analysis

```powershell
.\aim analyze sessions\20261124-170512_kovaaks.parquet     # one or more files
.\aim analyze (Get-ChildItem sessions\*.parquet).FullName  # everything
```

This writes `reports/<session>/` with `report.md`, the plots `timeline.png`,
`velocity_profiles.png`, `overshoot.png` and `paths.png`, and per-item tables in
`flicks.csv`, `tracking.csv` and `holds.csv`.

### How it segments

The session is binned onto a 1 ms grid in degrees and smoothed with a 6 ms Gaussian
to get angular speed.

- **Flicks** are anchored on clicks. KovaaK's target positions aren't logged, so the
  **click position is treated as the target**. For each click, the flick is the
  movement episode leading into it: the run of motion back to the last ≥60 ms pause.
  Onset is traced back to where the movement really starts. The episode is split into
  submovements at speed dips. The fastest submovement is the *primary* (ballistic)
  one, and every submovement after it is a *correction*.
  A click is not counted as a flick if there was no movement before it, if the
  movement ended more than 600 ms earlier, if the peak speed was under 20°/s (a small
  follow-up adjustment), or if the distance was under 0.5°.
- **Tracking segments** are at least 500 ms of continuous movement (gaps under 100 ms
  are bridged) that isn't part of a flick.
- **Holds** are 150 ms to 2 s with speed under 4°/s.

### Flick metrics (flicks.csv)

| column | meaning |
|---|---|
| `amplitude_deg`, `bucket` | onset-to-click distance; small / medium / large by `analysis.buckets` |
| `time_to_target_ms` | onset to click |
| `peak_velocity`, `time_to_peak_ms` | deg/s, and when it happens |
| `peak_time_frac` | peak position within the primary submovement (0.5 = symmetric bell, lower = front-loaded) |
| `primary_error_pct` | where the primary submovement lands along the target direction: −20 = 20% short, +10 = 10% past |
| `primary_offaxis_pct` | sideways error of the primary, % of distance |
| `max_overshoot_pct` | furthest point past the target at any time |
| `n_corrections`, `correction_time_ms` | submovements after the primary, and time from primary end to settle |
| `click_delay_ms` | settle to click (0 if you clicked while still moving) |
| `clicked_moving`, `speed_at_click` | crosshair speed when you fired |
| `path_efficiency` | straight-line distance / travelled distance (1 = perfectly straight) |

### Tracking and hold metrics

- `sparc`: spectral arc length per 1 s window. It measures smoothness; closer to 0 is smoother.
- `jitter_hf_ratio`: share of velocity power in the 8–30 Hz band, where tremor and shake show up.
- `jitter_rms_deg`: size of that 8–30 Hz motion in degrees.
- `speed_cv`: speed variability.
- Holds add `drift_rms_deg`, the slow wander during a hold, and `before_click`.
- `tracking_metrics(..., target_vel=...)` also reports `velocity_rmse` and `speed_gain`
  when target motion is known. KovaaK's doesn't provide it yet.

Thresholds live under `analysis:` in `config.yaml`, with the full list in
`aimlab_analyzer/segment.py` `Params`. They are all in degrees and ms, so they work
the same at any sens.

## Tests

```powershell
python -m pytest -q
```

## Manual setup and caveats

**Windows**
- No admin rights or drivers needed. Raw Input is a normal user-level API.
- Windows pointer speed and "Enhance pointer precision" don't affect the logged counts.
- Timestamps are taken when the event reaches Python, not on the mouse, so expect about 0.1 to 0.5 ms of jitter. `inspect` prints the effective rate. If it's well below your polling rate (for example with 4 kHz or 8 kHz mice), report it and the loop can be switched to buffered reads.
- If F8 or F10 is already a global hotkey in another app (OBS, Discord, overlays), registration fails with a clear error. Change the keys in `config.yaml`.
- Anti-cheat: the logger only listens to input and never injects any, like mouse-rate testers. That should be fine with VAC and offline KovaaK's. To be safe, don't run it during FACEIT or other third-party anti-cheat matches.
- Inputs from every mouse are combined. Unplug or don't touch a second pointing device while recording.

**Linux**
- Your user needs read access to `/dev/input/event*`: run `sudo usermod -aG input $USER`, then log out and back in. That gives the user access to *all* input devices, keyboards included, so you may prefer running with `sudo` only while recording.
- Works on Wayland and X11 because it reads the kernel devices directly. The Linux backend hasn't been tested on real hardware yet.

**Why not pynput or `mouse`?** They report cursor positions, which games re-center each
frame and Windows can accelerate. They don't give raw counts, so they can't produce
correct degrees and aren't included as a fallback.
