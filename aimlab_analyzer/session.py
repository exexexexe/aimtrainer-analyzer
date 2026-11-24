"""In-memory session buffer, saving, and loading."""

from __future__ import annotations

import json
import platform
import time
from array import array
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Conversion
from .units import add_angles

COLUMNS = ["t_ns", "dx", "dy", "buttons"]

# Button bitmask stored in the `buttons` column.
BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, BTN_X1, BTN_X2 = 1, 2, 4, 8, 16


class SessionBuffer:
    """Collects events for one recording. t_ns is relative to the first start()."""

    def __init__(self):
        self.t_ns = array("q")
        self.dx = array("q")
        self.dy = array("q")
        self.buttons = array("q")
        self.t0_ns: int | None = None
        self.started_at: datetime | None = None

    def start(self, t0_ns: int):
        self.t0_ns = t0_ns
        self.started_at = datetime.now().astimezone()

    def add(self, t_ns: int, dx: int, dy: int, buttons: int):
        self.t_ns.append(t_ns - self.t0_ns)
        self.dx.append(dx)
        self.dy.append(dy)
        self.buttons.append(buttons)

    def __len__(self):
        return len(self.t_ns)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t_ns": pd.array(self.t_ns, dtype="int64"),
                "dx": pd.array(self.dx, dtype="int32"),
                "dy": pd.array(self.dy, dtype="int32"),
                "buttons": pd.array(self.buttons, dtype="uint8"),
            }
        )


def save_session(
    buf: SessionBuffer,
    out_dir: Path,
    conversion: Conversion,
    fmt: str,
    backend: str,
    duration_ns: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = buf.started_at.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_{conversion.profile}"
    data_path = out_dir / f"{stem}.{fmt}"

    df = buf.to_frame()
    if fmt == "parquet":
        df.to_parquet(data_path, index=False)
    else:
        df.to_csv(data_path, index=False)

    meta = {
        "started_at": buf.started_at.isoformat(),
        "duration_s": duration_ns / 1e9,
        "events": len(df),
        "backend": backend,
        "platform": platform.platform(),
        "conversion": conversion.to_dict(),
        "columns": {
            "t_ns": "nanoseconds since session start (host clock at event receipt)",
            "dx": "raw mouse counts, + = right",
            "dy": "raw mouse counts, + = down (toward you)",
            "buttons": "bitmask: 1 left, 2 right, 4 middle, 8 mouse4, 16 mouse5",
        },
        "saved_at_unix": time.time(),
    }
    meta_path(data_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return data_path


def meta_path(data_path: Path) -> Path:
    return data_path.with_suffix(".meta.json")


def load_session(path: str | Path, conversion: Conversion | None = None):
    """Load a session file. Returns (frame with angle columns, meta dict).

    Uses the conversion saved with the session unless one is passed in.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    meta = json.loads(meta_path(path).read_text(encoding="utf-8"))
    conv = conversion.to_dict() if conversion else meta["conversion"]
    df = add_angles(df, conv["yaw_deg_per_count"], conv["pitch_deg_per_count"], conv["dpi"])
    return df, {**meta, "conversion": conv}
