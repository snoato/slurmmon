"""Light/dark-aware styles for the few things slurmmon paints a background
under: the selected row, group header rows, and the empty part of gauges.

Everything else is foreground-only and uses the terminal's own ANSI colors
(green/yellow/red, cyan, dim, ...), which every terminal theme already
remaps to be readable on its own background. A fixed background grey has no
such luck: `on grey35` under default-colored text is a subtle highlight on
a dark terminal, but on a light one that default text is itself dark, so
the row just turns into an unreadable block.

So those styles are Rich theme names here, not literal colors, and the
interactive loop picks the palette by asking the terminal for its actual
background color (xterm's OSC 11 query, which most modern terminal
emulators answer, over SSH too). It keeps re-asking every few seconds,
so flipping the terminal between light and dark mode mid-session is picked
up without a restart. Until an answer arrives -- or forever, if the
terminal never answers -- NEUTRAL is used, which is legible on either.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from typing import Callable

from rich.console import Console
from rich.theme import Theme

SELECTED = "slurmmon.selected"
GROUP_HEADER = "slurmmon.group_header"  # must stay distinct from SELECTED so a header row never reads as "selected"
GAUGE_TRACK = "slurmmon.gauge_track"  # only its bgcolor is used, see ui._Gauge

DARK = Theme({SELECTED: "on grey35", GROUP_HEADER: "bold on grey23", GAUGE_TRACK: "on grey23"})
LIGHT = Theme({SELECTED: "on grey78", GROUP_HEADER: "bold on grey89", GAUGE_TRACK: "on grey85"})
# Background unknown: nothing here may assume either polarity. Reverse video
# is legible on any background, and grey46 has ~4.5:1 contrast against both
# black and white, so an empty gauge track still shows up on either.
NEUTRAL = Theme({SELECTED: "reverse", GROUP_HEADER: "bold underline", GAUGE_TRACK: "on grey46"})

BACKGROUND_QUERY = "\x1b]11;?\x1b\\"
_REQUERY_INTERVAL = 3.0  # how quickly a mid-session light/dark switch gets picked up
_REPLY_TIMEOUT = 2.0  # the first query unanswered this long -> this terminal doesn't answer, stop asking

# Reply payload (between "ESC ]" and the BEL/ST terminator), e.g.
# "11;rgb:1e1e/1e1e/1e1e". Each channel is 1-4 hex digits; some terminals
# send rgba: with a trailing alpha channel, which is irrelevant here.
_BACKGROUND_REPLY = re.compile(
    r"11;rgba?:([0-9a-f]{1,4})/([0-9a-f]{1,4})/([0-9a-f]{1,4})(?:/[0-9a-f]{1,4})?", re.IGNORECASE
)


def parse_background_reply(payload: str) -> tuple[float, float, float] | None:
    """(r, g, b) in 0..1 from an OSC 11 reply payload, or None if it isn't one."""
    m = _BACKGROUND_REPLY.fullmatch(payload.strip())
    if m is None:
        return None
    return tuple(int(h, 16) / (16 ** len(h) - 1) for h in m.groups())  # type: ignore[return-value]


def is_light(rgb: tuple[float, float, float]) -> bool:
    """Whether dark text reads better than light text on this background,
    i.e. its WCAG relative luminance is past the point where contrast
    against black and against white are equal."""

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > math.sqrt(1.05 * 0.05) - 0.05


def make_console() -> Console:
    """A Console that can render slurmmon's renderables (which refer to the
    theme names above) before, or without, ever hearing back from the terminal."""
    return Console(theme=NEUTRAL)


def _can_query(console: Console) -> bool:
    # The reply comes back on stdin, so both ends must be the terminal. The
    # Linux VT doesn't understand OSC 11 and would print it as literal text.
    return (
        console.is_terminal
        and sys.stdin.isatty()
        and not console.is_dumb_terminal
        and os.environ.get("TERM") != "linux"
    )


class BackgroundWatcher:
    """Owns the OSC 11 query/reply cycle for one interactive session and
    swaps `console`'s palette to match whatever the terminal reports.

    Call tick() once per render-loop iteration (it only actually writes a
    query every _REQUERY_INTERVAL), and feed every OSC reply read_key()
    returns to handle_reply()."""

    def __init__(
        self,
        console: Console,
        *,
        enabled: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if enabled is None:
            enabled = _can_query(console)
        self._console = console
        self._clock = clock
        self._supported: bool | None = None if enabled else False  # None: no reply seen yet, keep trying
        self._sent_at: float | None = None  # set while a query is outstanding
        self._last_sent = -math.inf
        self._theme: Theme | None = None  # what we pushed on top of the console's NEUTRAL base, if anything

    @property
    def awaiting_reply(self) -> bool:
        return self._sent_at is not None

    def tick(self) -> None:
        if self._supported is False:
            return
        now = self._clock()
        if self._sent_at is not None:
            if now - self._sent_at < _REPLY_TIMEOUT:
                return
            self._sent_at = None
            if self._supported is None:
                self._supported = False
                return
        if now - self._last_sent < _REQUERY_INTERVAL:
            return
        self._console.file.write(BACKGROUND_QUERY)
        self._console.file.flush()
        self._sent_at = self._last_sent = now

    def handle_reply(self, payload: str) -> None:
        rgb = parse_background_reply(payload)
        if rgb is None:
            return
        self._sent_at = None
        self._supported = True
        theme = LIGHT if is_light(rgb) else DARK
        if theme is self._theme:
            return
        if self._theme is not None:
            self._console.pop_theme()
        self._console.push_theme(theme)
        self._theme = theme
