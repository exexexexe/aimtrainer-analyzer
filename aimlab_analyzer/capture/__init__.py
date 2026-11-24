"""Capture backends. Each exposes NAME and run(recorder, hotkeys)."""

from __future__ import annotations

import sys


def get_backend():
    if sys.platform == "win32":
        from . import windows_raw as backend
    elif sys.platform.startswith("linux"):
        from . import linux_evdev as backend
    else:
        raise RuntimeError(f"no capture backend for platform {sys.platform}")
    return backend


def update_buttons(state: int, transitions: dict[int, bool]) -> int:
    """Apply {bit: pressed} transitions to a button bitmask."""
    for bit, pressed in transitions.items():
        state = state | bit if pressed else state & ~bit
    return state
