"""Windows Raw Input mouse capture via ctypes (no extra dependencies).

Raw Input gives the unaccelerated counts the mouse reports, independent of the
cursor, Windows pointer speed and "Enhance pointer precision". RIDEV_INPUTSINK lets
us receive input while CS2 / KovaaK's has focus. Hotkeys use RegisterHotKey.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from ..hotkeys import parse_hotkey
from ..session import BTN_LEFT, BTN_MIDDLE, BTN_RIGHT, BTN_X1, BTN_X2
from . import update_buttons

NAME = "windows-rawinput"

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WM_INPUT = 0x00FF
WM_HOTKEY = 0x0312
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
MOUSE_MOVE_ABSOLUTE = 0x01
HWND_MESSAGE = wintypes.HWND(-3)
PM_REMOVE = 0x0001
QS_ALLINPUT = 0x04FF
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
THREAD_PRIORITY_HIGHEST = 2

# usButtonFlags -> (bitmask bit, pressed)
BUTTON_FLAGS = {
    0x0001: (BTN_LEFT, True), 0x0002: (BTN_LEFT, False),
    0x0004: (BTN_RIGHT, True), 0x0008: (BTN_RIGHT, False),
    0x0010: (BTN_MIDDLE, True), 0x0020: (BTN_MIDDLE, False),
    0x0040: (BTN_X1, True), 0x0080: (BTN_X1, False),
    0x0100: (BTN_X2, True), 0x0200: (BTN_X2, False),
}


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _ButtonStruct(ctypes.Structure):
    _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]


class _ButtonUnion(ctypes.Union):
    _fields_ = [("ulButtons", wintypes.ULONG), ("s", _ButtonStruct)]


class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("u", _ButtonUnion),  # ULONG-aligned, so it sits at offset 4 like the C struct
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
user32.RegisterRawInputDevices.restype = wintypes.BOOL
user32.GetRawInputData.argtypes = [
    wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID, ctypes.POINTER(wintypes.UINT), wintypes.UINT,
]
user32.GetRawInputData.restype = wintypes.UINT
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.RegisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
user32.PeekMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.MsgWaitForMultipleObjects.argtypes = [
    wintypes.DWORD, ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD, wintypes.DWORD,
]
user32.MsgWaitForMultipleObjects.restype = wintypes.DWORD
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetCurrentThread.restype = wintypes.HANDLE
kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]

HOTKEY_TOGGLE, HOTKEY_QUIT = 1, 2


def vk_code(key: str) -> int:
    named = {
        "pause": 0x13, "scrolllock": 0x91, "insert": 0x2D, "home": 0x24, "end": 0x23,
        "pageup": 0x21, "pagedown": 0x22, "delete": 0x2E,
    }
    if key.lower() in named:
        return named[key.lower()]
    if key.startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        return 0x70 + int(key[1:]) - 1
    if key.lower().startswith("numpad") and key[6:].isdigit():
        return 0x60 + int(key[6:])
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    raise ValueError(f"unsupported hotkey key '{key}'")


def hotkey_args(text: str) -> tuple[int, int]:
    mods, key = parse_hotkey(text)
    flags = MOD_NOREPEAT
    for m, bit in (("alt", MOD_ALT), ("ctrl", MOD_CONTROL), ("shift", MOD_SHIFT), ("win", MOD_WIN)):
        if m in mods:
            flags |= bit
    return flags, vk_code(key)


def _check(ok, what):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error(), f"{what} failed")


def run(recorder, hotkeys: dict):
    hinst = kernel32.GetModuleHandleW(None)
    wndproc = WNDPROC(lambda h, m, w, l: user32.DefWindowProcW(h, m, w, l))  # keep a ref
    wc = WNDCLASSW(lpfnWndProc=wndproc, hInstance=hinst, lpszClassName="AimlabAnalyzerRawInput")
    user32.RegisterClassW(ctypes.byref(wc))  # returns 0 if already registered; harmless
    hwnd = user32.CreateWindowExW(0, wc.lpszClassName, "aimlab", 0, 0, 0, 0, 0, HWND_MESSAGE, None, hinst, None)
    _check(hwnd, "CreateWindowExW")

    try:
        rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, hwnd)  # generic desktop / mouse
        _check(user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)),
               "RegisterRawInputDevices")

        for hk_id, name in ((HOTKEY_TOGGLE, "toggle"), (HOTKEY_QUIT, "quit")):
            mods, vk = hotkey_args(hotkeys[name])
            if not user32.RegisterHotKey(hwnd, hk_id, mods, vk):
                raise RuntimeError(
                    f"could not register {name} hotkey '{hotkeys[name]}' "
                    "(another program probably owns it; pick a different key in config.yaml)"
                )

        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST)
        _loop(recorder)
    finally:
        user32.UnregisterHotKey(hwnd, HOTKEY_TOGGLE)
        user32.UnregisterHotKey(hwnd, HOTKEY_QUIT)
        user32.DestroyWindow(hwnd)


def _loop(recorder):
    msg = wintypes.MSG()
    raw = RAWINPUT()
    raw_size = wintypes.UINT()
    header_size = ctypes.sizeof(RAWINPUTHEADER)
    buttons = 0
    warned_absolute = False
    perf_ns = time.perf_counter_ns

    while not recorder.should_exit():
        # Wake on any input or every 50 ms so Ctrl+C and --duration are handled.
        user32.MsgWaitForMultipleObjects(0, None, False, 50, QS_ALLINPUT)
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            if msg.message == WM_INPUT:
                t = perf_ns()
                raw_size.value = ctypes.sizeof(raw)
                n = user32.GetRawInputData(msg.lParam, RID_INPUT, ctypes.byref(raw),
                                           ctypes.byref(raw_size), header_size)
                if n != 0xFFFFFFFF and raw.header.dwType == RIM_TYPEMOUSE:
                    m = raw.mouse
                    flags = m.u.s.usButtonFlags
                    if flags:
                        buttons = update_buttons(
                            buttons, dict(v for f, v in BUTTON_FLAGS.items() if flags & f)
                        )
                    if m.usFlags & MOUSE_MOVE_ABSOLUTE:
                        # Tablets / remote desktop. Not relative counts, so drop the motion.
                        if not warned_absolute:
                            recorder.log("[warn] absolute-position mouse input ignored (tablet/RDP?)")
                            warned_absolute = True
                        dx = dy = 0
                    else:
                        dx, dy = m.lLastX, m.lLastY
                    if dx or dy or flags:
                        recorder.on_event(t, dx, dy, buttons)
            elif msg.message == WM_HOTKEY:
                if msg.wParam == HOTKEY_TOGGLE:
                    recorder.toggle(perf_ns())
                elif msg.wParam == HOTKEY_QUIT:
                    recorder.request_quit()
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
