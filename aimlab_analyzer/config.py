"""Config loading and counts-to-degrees resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class Conversion:
    """Resolved conversion for one game profile."""

    profile: str
    game: str
    dpi: float
    yaw_deg_per_count: float
    pitch_deg_per_count: float
    method: str  # which rule produced yaw_deg_per_count

    @property
    def cm_per_360(self) -> float:
        counts_per_360 = 360.0 / self.yaw_deg_per_count
        return counts_per_360 / self.dpi * 2.54

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cm_per_360"] = self.cm_per_360
        return d


def resolve_conversion(profile_name: str, profile: dict, dpi: float) -> Conversion:
    if not dpi or dpi <= 0:
        raise ValueError("mouse.dpi must be a positive number")

    if profile.get("deg_per_count"):
        yaw = float(profile["deg_per_count"])
        method = "deg_per_count"
    elif profile.get("cm_per_360"):
        counts_per_360 = float(profile["cm_per_360"]) / 2.54 * dpi
        yaw = 360.0 / counts_per_360
        method = "cm_per_360"
    elif profile.get("sens") is not None and profile.get("yaw") is not None:
        yaw = float(profile["sens"]) * float(profile["yaw"])
        method = "sens*yaw"
    else:
        raise ValueError(
            f"profile '{profile_name}' needs one of: deg_per_count, cm_per_360, or sens + yaw"
        )

    if method == "sens*yaw" and profile.get("pitch") is not None:
        pitch = float(profile["sens"]) * float(profile["pitch"])
    else:
        pitch = yaw * float(profile.get("pitch_ratio", 1.0))

    return Conversion(
        profile=profile_name,
        game=str(profile.get("game", profile_name)),
        dpi=float(dpi),
        yaw_deg_per_count=yaw,
        pitch_deg_per_count=pitch,
        method=method,
    )


@dataclass
class Config:
    path: Path
    raw: dict

    @property
    def dpi(self) -> float:
        return float(self.raw["mouse"]["dpi"])

    @property
    def active_profile(self) -> str:
        return self.raw.get("active_profile", "cs2")

    @property
    def hotkeys(self) -> dict:
        return {"toggle": "F8", "quit": "F10", **(self.raw.get("hotkeys") or {})}

    @property
    def output_dir(self) -> Path:
        d = Path((self.raw.get("output") or {}).get("dir", "sessions"))
        return d if d.is_absolute() else self.path.parent / d

    @property
    def output_format(self) -> str:
        fmt = (self.raw.get("output") or {}).get("format", "parquet").lower()
        if fmt not in ("parquet", "csv"):
            raise ValueError("output.format must be 'parquet' or 'csv'")
        return fmt

    def conversion(self, profile_name: str | None = None) -> Conversion:
        name = profile_name or self.active_profile
        profiles = self.raw.get("profiles") or {}
        if name not in profiles:
            raise KeyError(f"profile '{name}' not in config (have: {', '.join(profiles)})")
        return resolve_conversion(name, profiles[name], self.dpi)


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(path=p.resolve(), raw=raw)
