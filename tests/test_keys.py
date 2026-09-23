import os

from slurmmon.keys import OSC_PREFIX, read_key


class _FakeStdin:
    """Minimal stand-in for sys.stdin: read_key only ever calls isatty()
    and fileno() on it (all actual byte reads go through os.read on the
    fd directly) -- a pipe's read end is enough to drive it."""

    def __init__(self, fd: int):
        self._fd = fd

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd


def _pipe_pair(monkeypatch):
    r, w = os.pipe()
    monkeypatch.setattr("slurmmon.keys.sys.stdin", _FakeStdin(r))
    return r, w


def test_read_key_plain_character(monkeypatch):
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"q")
    assert read_key(1.0) == "q"


def test_read_key_arrow_delivered_as_one_terminal_burst(monkeypatch):
    # A real terminal emulator writes a whole escape sequence in a single
    # write() syscall, not byte by byte -- this is the exact shape that
    # broke when read_key() used sys.stdin.read() (a buffered reader can
    # slurp the whole burst out of the kernel on the first byte, leaving
    # select() nothing to peek).
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b[A")
    assert read_key(1.0) == "UP"


def test_read_key_pgup_delivered_as_one_terminal_burst(monkeypatch):
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b[5~")
    assert read_key(1.0) == "PGUP"


def test_read_key_pgdn_delivered_as_one_terminal_burst(monkeypatch):
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b[6~")
    assert read_key(1.0) == "PGDN"


def test_read_key_does_not_desync_the_next_keypress(monkeypatch):
    # The actual bug: a burst gets misread, and leftover buffered bytes
    # then get returned as a bogus *next* keypress. Simulate two keys
    # arriving back to back in one burst and confirm both come out right,
    # in order, with nothing left over.
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b[A" + b"j")
    assert read_key(1.0) == "UP"
    assert read_key(1.0) == "j"


def test_read_key_lone_escape_returns_esc(monkeypatch):
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b")
    assert read_key(1.0) == "ESC"


def test_read_key_no_input_returns_none(monkeypatch):
    _pipe_pair(monkeypatch)
    assert read_key(0.05) is None


def test_read_key_background_reply_st_terminated(monkeypatch):
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\")
    assert read_key(1.0) == OSC_PREFIX + "11;rgb:ffff/ffff/ffff"


def test_read_key_background_reply_bel_terminated(monkeypatch):
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07")
    assert read_key(1.0) == OSC_PREFIX + "11;rgb:1e1e/1e1e/1e1e"


def test_read_key_background_reply_is_consumed_whole(monkeypatch):
    # Left half-read, the payload would replay as keypresses -- "r" (refresh),
    # "g" (cycle grouping), "f" (cycle filter), "]" (resize), ...
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\" + b"j")
    assert read_key(1.0).startswith(OSC_PREFIX)
    assert read_key(1.0) == "j"
    assert read_key(0.05) is None


def test_read_key_escape_right_before_a_reply_keeps_both(monkeypatch):
    # A real Escape keypress with a query reply arriving inside its peek
    # window: the reply's own ESC must not get eaten as "the byte after Escape".
    r, w = _pipe_pair(monkeypatch)
    os.write(w, b"\x1b" + b"\x1b]11;rgb:0000/0000/0000\x1b\\")
    assert read_key(1.0) == "ESC"
    assert read_key(1.0) == OSC_PREFIX + "11;rgb:0000/0000/0000"
    assert read_key(0.05) is None
