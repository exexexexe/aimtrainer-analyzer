"""Read KovaaK's per-run stats CSVs (FPSAimTrainer/stats/*.csv)."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import Conversion

STEAM_STATS = Path("steamapps/common/FPSAimTrainer/FPSAimTrainer/stats")
FILENAME_RE = re.compile(
    r"^(?P<scenario>.+) - (?P<mode>[^-]+?) - (?P<date>\d{4}\.\d{2}\.\d{2})-(?P<time>\d{2}\.\d{2}\.\d{2}) Stats\.csv$"
)
# Degrees per count at sens 1.0 for KovaaK's game sens scales.
SENS_SCALE_YAW = {
    "counter-strike": 0.022, "csgo": 0.022, "cs:go": 0.022, "source": 0.022,
    "quake/source": 0.022, "apex": 0.022, "apex legends": 0.022,
    "overwatch": 0.0066, "valorant": 0.07,
}
SHOTS_PER_S_TRACKING = 4.0  # faster than this = automatic weapon, i.e. tracking


@dataclass
class Run:
    path: Path
    id: str
    scenario: str
    start: datetime
    end: datetime
    score: float
    kills: int
    shots: int
    hits: int
    avg_ttk: float
    overshots: int
    sens_scale: str
    horiz_sens: float
    vert_sens: float
    dpi: float
    fov: float
    kill_times: np.ndarray = field(repr=False)   # seconds since run start
    kill_shots: np.ndarray = field(repr=False)
    kill_hits: np.ndarray = field(repr=False)

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def accuracy(self) -> float:
        return self.hits / self.shots if self.shots else float("nan")

    @property
    def kind(self) -> str:
        rate = self.shots / self.duration_s if self.duration_s > 0 else 0
        return "tracking" if rate > SHOTS_PER_S_TRACKING else "click"

    def summary(self) -> dict:
        return {
            "id": self.id, "scenario": self.scenario, "kind": self.kind,
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "duration_s": round(self.duration_s, 1), "score": self.score,
            "kills": self.kills, "shots": self.shots, "hits": self.hits,
            "accuracy": self.accuracy, "avg_ttk": self.avg_ttk, "overshots": self.overshots,
            "sens": f"{self.horiz_sens:g} ({self.sens_scale})",
        }


def run_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]


def _float(v, default=float("nan")) -> float:
    try:
        return float(str(v).strip().rstrip("s"))
    except (TypeError, ValueError):
        return default


def _time_on(day: datetime, hms: str) -> datetime:
    t = datetime.strptime(hms.strip(), "%H:%M:%S.%f").time()
    return datetime.combine(day.date(), t)


def parse_stats(path: str | Path) -> Run:
    path = Path(path)
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"not a KovaaK's stats file name: {path.name}")
    end = datetime.strptime(f"{m['date']} {m['time']}", "%Y.%m.%d %H.%M.%S")

    kill_rows, kv = [], {}
    in_kills = True
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if not row or not any(c.strip() for c in row):
                in_kills = False
                continue
            if row[0] == "Kill #":
                continue
            if in_kills:
                kill_rows.append(row)
            elif row[0].endswith(":") and len(row) >= 2:
                kv[row[0][:-1].strip()] = row[1].strip()

    start = _time_on(end, kv["Challenge Start"]) if "Challenge Start" in kv else end
    if start > end:  # started before midnight
        start -= timedelta(days=1)

    times, shots, hits = [], [], []
    for r in kill_rows:
        t = _time_on(start, r[1])
        if t < start - timedelta(minutes=1):
            t += timedelta(days=1)
        times.append((t - start).total_seconds())
        shots.append(int(_float(r[5], 0)))
        hits.append(int(_float(r[6], 0)))

    return Run(
        path=path, id=run_id(path),
        scenario=kv.get("Scenario") or m["scenario"],
        start=start, end=end,
        score=_float(kv.get("Score")),
        kills=int(_float(kv.get("Kills"), len(kill_rows))),
        shots=int(_float(kv.get("Hit Count"), 0) + _float(kv.get("Miss Count"), 0)),
        hits=int(_float(kv.get("Hit Count"), 0)),
        avg_ttk=_float(kv.get("Avg TTK")),
        overshots=int(_float(kv.get("Total Overshots"), 0)),
        sens_scale=kv.get("Sens Scale", ""),
        horiz_sens=_float(kv.get("Horiz Sens")),
        vert_sens=_float(kv.get("Vert Sens")),
        dpi=_float(kv.get("DPI")),
        fov=_float(kv.get("FOV")),
        kill_times=np.array(times), kill_shots=np.array(shots, dtype=int), kill_hits=np.array(hits, dtype=int),
    )


def run_conversion(run: Run) -> Conversion | None:
    """Counts -> degrees from the sens KovaaK's saved with the run, if the scale is known."""
    scale = run.sens_scale.strip().lower()
    if not (run.horiz_sens > 0 and run.dpi > 0):
        return None
    if "cm" in scale:
        yaw = 360.0 / (run.horiz_sens / 2.54 * run.dpi)
    elif scale in SENS_SCALE_YAW:
        yaw = run.horiz_sens * SENS_SCALE_YAW[scale]
    else:
        return None
    ratio = run.vert_sens / run.horiz_sens if run.vert_sens > 0 else 1.0
    if "cm" in scale and run.vert_sens > 0:
        ratio = run.horiz_sens / run.vert_sens  # cm/360: bigger number = slower
    return Conversion(
        profile=f"kovaaks ({run.sens_scale} {run.horiz_sens:g})", game="kovaaks",
        dpi=run.dpi, yaw_deg_per_count=yaw, pitch_deg_per_count=yaw * ratio, method="kovaaks-stats",
    )


def find_stats_dir(configured: str | None = None) -> Path | None:
    if configured and configured != "auto":
        p = Path(configured)
        return p if p.is_dir() else None
    roots = [Path("C:/Program Files (x86)/Steam"), Path("C:/Program Files/Steam"),
             Path.home() / ".local/share/Steam", Path.home() / ".steam/steam"]
    for drive in "CDEFGH":
        roots += [Path(f"{drive}:/SteamLibrary"), Path(f"{drive}:/Steam")]
    for r in roots:
        if (r / STEAM_STATS).is_dir():
            return r / STEAM_STATS
    return None


class RunStore:
    """Parses the stats folder incrementally; call refresh() to pick up new runs."""

    def __init__(self, stats_dir: Path):
        self.dir = Path(stats_dir)
        self._by_path: dict[str, tuple[float, Run]] = {}
        self.errors: dict[str, str] = {}

    def refresh(self) -> list[Run]:
        seen = set()
        for p in self.dir.glob("*Stats.csv"):
            key = str(p)
            seen.add(key)
            mtime = p.stat().st_mtime
            cached = self._by_path.get(key)
            if cached and cached[0] == mtime:
                continue
            try:
                self._by_path[key] = (mtime, parse_stats(p))
                self.errors.pop(key, None)
            except Exception as e:  # half-written or odd files: skip, retry next refresh
                self.errors[key] = str(e)
        for key in set(self._by_path) - seen:
            del self._by_path[key]
        return self.runs()

    def runs(self) -> list[Run]:
        return sorted((r for _, r in self._by_path.values()), key=lambda r: r.end)

    def get(self, rid: str) -> Run | None:
        return next((r for _, r in self._by_path.values() if r.id == rid), None)
