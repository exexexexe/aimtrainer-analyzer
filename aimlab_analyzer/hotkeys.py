"""Platform-neutral hotkey parsing: "ctrl+alt+F8" -> ({"ctrl", "alt"}, "F8")."""

from __future__ import annotations

MODIFIERS = {"ctrl", "alt", "shift", "win"}


def parse_hotkey(text: str) -> tuple[frozenset[str], str]:
    parts = [p.strip() for p in str(text).split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    *mods, key = parts
    mods = {m.lower() for m in mods}
    unknown = mods - MODIFIERS
    if unknown:
        raise ValueError(f"unknown modifier(s) {sorted(unknown)} in hotkey '{text}'")
    if key.lower() in MODIFIERS:
        raise ValueError(f"hotkey '{text}' needs a non-modifier key last")
    key = key.upper() if len(key) <= 3 else key.lower()
    return frozenset(mods), key
