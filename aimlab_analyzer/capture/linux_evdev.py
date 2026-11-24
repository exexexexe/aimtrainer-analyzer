"""Linux evdev mouse capture. Needs read access to /dev/input/event* (see README).

Uses kernel event timestamps, so timing does not depend on Python scheduling.
"""

from __future__ import annotations

import select
import time

try:
    import evdev
    from evdev import ecodes
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Linux capture needs the 'evdev' package: pip install evdev") from e

from ..hotkeys import parse_hotkey
from ..session import BTN_LEFT, BTN_MIDDLE, BTN_RIGHT, BTN_X1, BTN_X2
from . import update_buttons

NAME = "linux-evdev"

BUTTON_BITS = {
    ecodes.BTN_LEFT: BTN_LEFT, ecodes.BTN_RIGHT: BTN_RIGHT, ecodes.BTN_MIDDLE: BTN_MIDDLE,
    ecodes.BTN_SIDE: BTN_X1, ecodes.BTN_EXTRA: BTN_X2,
}
MOD_KEYS = {
    "ctrl": {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL},
    "alt": {ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT},
    "shift": {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT},
    "win": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
}


def key_code(key: str) -> int:
    name = "KEY_" + key.upper()
    if not hasattr(ecodes, name):
        raise ValueError(f"unsupported hotkey key '{key}'")
    return getattr(ecodes, name)


def find_devices():
    mice, keyboards = [], []
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        caps = dev.capabilities()
        rel = caps.get(ecodes.EV_REL, [])
        keys = caps.get(ecodes.EV_KEY, [])
        if ecodes.REL_X in rel and ecodes.REL_Y in rel:
            mice.append(dev)
        elif ecodes.KEY_A in keys and ecodes.KEY_F8 in keys:
            keyboards.append(dev)
        else:
            dev.close()
    if not mice:
        raise RuntimeError(
            "no readable mouse found in /dev/input. Add yourself to the 'input' group "
            "(sudo usermod -aG input $USER, then log out and in) or run with sudo."
        )
    if not keyboards:
        raise RuntimeError("no readable keyboard found in /dev/input for hotkeys (same fix as above)")
    return mice, keyboards


def run(recorder, hotkeys: dict):
    combos = {}
    for name in ("toggle", "quit"):
        mods, key = parse_hotkey(hotkeys[name])
        combos[key_code(key)] = (name, mods)

    mice, keyboards = find_devices()
    recorder.log("[evdev] mice: " + ", ".join(d.name for d in mice))
    devices = {d.fd: d for d in mice + keyboards}
    mouse_fds = {d.fd for d in mice}

    # evdev timestamps are CLOCK_REALTIME; map them onto perf_counter_ns so they
    # share a clock with Recorder.start().
    offset_ns = time.perf_counter_ns() - time.time_ns()

    buttons = 0
    pending = {}  # fd -> [dx, dy, button_changed]
    held = set()

    try:
        while not recorder.should_exit():
            ready, _, _ = select.select(list(devices), [], [], 0.05)
            for fd in ready:
                dev = devices[fd]
                for ev in dev.read():
                    if fd in mouse_fds:
                        acc = pending.setdefault(fd, [0, 0, False])
                        if ev.type == ecodes.EV_REL:
                            if ev.code == ecodes.REL_X:
                                acc[0] += ev.value
                            elif ev.code == ecodes.REL_Y:
                                acc[1] += ev.value
                        elif ev.type == ecodes.EV_KEY and ev.code in BUTTON_BITS and ev.value in (0, 1):
                            buttons = update_buttons(buttons, {BUTTON_BITS[ev.code]: bool(ev.value)})
                            acc[2] = True
                        elif ev.type == ecodes.EV_SYN and ev.code == ecodes.SYN_REPORT:
                            dx, dy, changed = acc
                            if dx or dy or changed:
                                t = ev.sec * 1_000_000_000 + ev.usec * 1000 + offset_ns
                                recorder.on_event(t, dx, dy, buttons)
                            pending[fd] = [0, 0, False]
                    elif ev.type == ecodes.EV_KEY:
                        if ev.value == 1:
                            held.add(ev.code)
                            if ev.code in combos:
                                name, mods = combos[ev.code]
                                if all(held & MOD_KEYS[m] for m in mods):
                                    if name == "toggle":
                                        recorder.toggle(time.perf_counter_ns())
                                    else:
                                        recorder.request_quit()
                        elif ev.value == 0:
                            held.discard(ev.code)
    finally:
        for d in devices.values():
            d.close()
