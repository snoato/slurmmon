"""Minimal non-blocking single-key reader for the interactive TUI loop.

Deliberately hand-rolled with termios/tty/select instead of an extra
dependency (curses/prompt_toolkit/textual) -- all we need is "is a key
available, and if so which one", polled from the main render loop.
"""
from __future__ import annotations

import contextlib
import os
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
_ESCAPE_PEEK_TIMEOUT = 0.15  # a real Escape keypress won't be followed by more bytes this fast;
# generous on purpose since misreading a slightly-delayed CSI sequence as a
# lone Escape (e.g. over a laggier SSH/terminal-multiplexer setup) silently
# breaks PgUp/PgDn, which is worse than Escape taking 0.15s to register


# A byte read one too far that belongs to the *next* sequence (see read_key's
# "ESC ESC" case), handed back out before anything else from the fd.
_pushback: list[str] = []

OSC_PREFIX = "OSC:"
_MAX_OSC_LEN = 256


def _ready(fd: int, timeout: float) -> bool:
    return bool(_pushback) or bool(select.select([fd], [], [], timeout)[0])


def _read_byte(fd: int) -> str:
    if _pushback:
        return _pushback.pop()
    # os.read() on the raw fd, never sys.stdin.read(): the latter goes
    # through Python's *buffered* TextIOWrapper, which can slurp an entire
    # multi-byte escape sequence out of the kernel in one os.read() the
    # moment we ask for just the first byte -- select() on the raw fd then
    # sees nothing left to peek (it's sitting in Python's own buffer, not
    # the kernel's), so a real CSI sequence gets misread as a lone Escape,
    # AND the still-buffered remainder desyncs onto the *next* read_key()
    # call as if it were a brand new (bogus) keypress. Reading raw bytes
    # one at a time keeps select()'s view of "what's pending" accurate.
    data = os.read(fd, 1)
    return data.decode("utf-8", errors="replace") if data else ""


def read_key(timeout: float) -> str | None:
    """Return a single logical keypress within `timeout` seconds, or None.

    Most keys come back as the literal character. Recognized arrow/page
    keys come back as "UP"/"DOWN"/"LEFT"/"RIGHT"/"PGUP"/"PGDN"; an
    unrecognized escape sequence or a lone Escape keypress comes back as
    "ESC". An OSC string -- never a keypress, only ever the terminal's
    reply to a query we sent (see theme.BackgroundWatcher) -- comes back as
    OSC_PREFIX + its payload, and must be consumed whole here: left in the
    input, "\x1b]11;rgb:ffff/..." would otherwise replay as the keys "]",
    "r", "g", "f", ...
    """
    if not sys.stdin.isatty():
        return None
    fd = sys.stdin.fileno()
    if not _ready(fd, timeout):
        return None
    ch = _read_byte(fd)
    if ch != "\x1b":
        return ch

    if not _ready(fd, _ESCAPE_PEEK_TIMEOUT):
        return "ESC"
    intro = _read_byte(fd)
    if intro == "]":
        return _read_osc(fd)
    if intro == "\x1b":
        # A lone Escape keypress with the next sequence (e.g. a query reply)
        # hot on its heels -- that ESC starts the next one, don't eat it.
        _pushback.append(intro)
        return "ESC"
    if intro != "[":
        return "ESC"
    if not _ready(fd, _ESCAPE_PEEK_TIMEOUT):
        return "ESC"
    final = _read_byte(fd)

    if final in _CSI_FINAL_BYTE:
        return _CSI_FINAL_BYTE[final]
    if final in _CSI_TILDE_CODE:
        # PageUp/PageDown are "ESC [ 5 ~" / "ESC [ 6 ~" -- consume the '~'.
        if _ready(fd, _ESCAPE_PEEK_TIMEOUT):
            _read_byte(fd)
        return _CSI_TILDE_CODE[final]
    return "ESC"


def _read_osc(fd: int) -> str:
    """Consume the rest of an OSC string after its "ESC ]", through its BEL
    or ST ("ESC \\") terminator, and return OSC_PREFIX + the payload."""
    payload = []
    while len(payload) < _MAX_OSC_LEN:
        if not _ready(fd, _ESCAPE_PEEK_TIMEOUT):
            return "ESC"  # truncated, or really just an Alt+] keypress
        ch = _read_byte(fd)
        if ch == "\x07":
            break
        if ch == "\x1b":
            if _ready(fd, _ESCAPE_PEEK_TIMEOUT) and (nxt := _read_byte(fd)) != "\\":
                _pushback.append(nxt)
            break
        payload.append(ch)
    return OSC_PREFIX + "".join(payload)
