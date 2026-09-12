"""Minimal non-blocking single-key reader for the interactive TUI loop.

Deliberately hand-rolled with termios/tty/select instead of an extra
dependency (curses/prompt_toolkit/textual) -- all we need is "is a key
available, and if so which one", polled from the main render loop.
"""
from __future__ import annotations

import contextlib
import select
import sys

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX platform
    termios = None
    tty = None


@contextlib.contextmanager
def raw_terminal():
    """Put stdin into cbreak mode for the duration of the block, restoring
    the previous settings afterwards (including on exception)."""
    if termios is None or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# xterm-style CSI sequences (ESC [ ...) used by arrow/page keys in effectively
# every terminal emulator (Terminal.app, iTerm2, most Linux terminals over SSH).
_CSI_FINAL_BYTE = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
_CSI_TILDE_CODE = {"5": "PGUP", "6": "PGDN"}
_ESCAPE_PEEK_TIMEOUT = 0.05  # a real Escape keypress won't be followed by more bytes this fast


def read_key(timeout: float) -> str | None:
    """Return a single logical keypress within `timeout` seconds, or None.

    Most keys come back as the literal character. Recognized arrow/page
    keys come back as "UP"/"DOWN"/"LEFT"/"RIGHT"/"PGUP"/"PGDN"; an
    unrecognized escape sequence or a lone Escape keypress comes back as
    "ESC".
    """
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch

    if not select.select([sys.stdin], [], [], _ESCAPE_PEEK_TIMEOUT)[0]:
        return "ESC"
    if sys.stdin.read(1) != "[":
        return "ESC"
    if not select.select([sys.stdin], [], [], _ESCAPE_PEEK_TIMEOUT)[0]:
        return "ESC"
    final = sys.stdin.read(1)

    if final in _CSI_FINAL_BYTE:
        return _CSI_FINAL_BYTE[final]
    if final in _CSI_TILDE_CODE:
        # PageUp/PageDown are "ESC [ 5 ~" / "ESC [ 6 ~" -- consume the '~'.
        if select.select([sys.stdin], [], [], _ESCAPE_PEEK_TIMEOUT)[0]:
            sys.stdin.read(1)
        return _CSI_TILDE_CODE[final]
    return "ESC"
