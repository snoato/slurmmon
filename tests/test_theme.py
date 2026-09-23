import io
from pathlib import Path

import pytest
from rich.console import Console

from slurmmon import cli, ssh_client, theme
from slurmmon.ssh_client import PartitionSelector, RawSections
from slurmmon.state import AppState
from slurmmon.theme import DARK, LIGHT, NEUTRAL, BackgroundWatcher, is_light, parse_background_reply
from slurmmon.ui import render

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_background_reply_four_digit_channels():
    assert parse_background_reply("11;rgb:ffff/ffff/ffff") == (1.0, 1.0, 1.0)


def test_parse_background_reply_two_digit_channels():
    r, g, b = parse_background_reply("11;rgb:1e/1e/1e")
    assert r == g == b == pytest.approx(0x1E / 0xFF)


def test_parse_background_reply_ignores_alpha():
    assert parse_background_reply("11;rgba:0000/8000/ffff/ffff") == pytest.approx((0.0, 0x8000 / 0xFFFF, 1.0))


@pytest.mark.parametrize("payload", ["10;rgb:ffff/ffff/ffff", "11;?", "11;rgb:ffff/ffff", "0;window title", ""])
def test_parse_background_reply_rejects_anything_else(payload):
    assert parse_background_reply(payload) is None


@pytest.mark.parametrize(
    "hex_rgb, light",
    [
        ("ffffff", True),
        ("fdf6e3", True),  # solarized light
        ("f8f8f8", True),
        ("000000", False),
        ("1e1e1e", False),  # vscode dark
        ("002b36", False),  # solarized dark
        ("282a36", False),  # dracula
    ],
)
def test_is_light(hex_rgb, light):
    rgb = tuple(int(hex_rgb[i : i + 2], 16) / 255 for i in (0, 2, 4))
    assert is_light(rgb) is light


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _watcher(enabled: bool = True):
    out = io.StringIO()
    console = Console(file=out, theme=NEUTRAL)
    clock = _Clock()
    return BackgroundWatcher(console, enabled=enabled, clock=clock), console, out, clock


def _selected_bg(console: Console):
    return console.get_style(theme.SELECTED)


def test_watcher_follows_the_reported_background_both_ways():
    watcher, console, out, clock = _watcher()
    assert _selected_bg(console) == NEUTRAL.styles[theme.SELECTED]

    watcher.tick()
    assert out.getvalue() == theme.BACKGROUND_QUERY
    watcher.handle_reply("11;rgb:ffff/ffff/ffff")
    assert _selected_bg(console) == LIGHT.styles[theme.SELECTED]

    # Terminal flipped to dark mode mid-session, then back again.
    for payload, expected in [("11;rgb:1e1e/1e1e/1e1e", DARK), ("11;rgb:fafa/fafa/fafa", LIGHT)]:
        clock.now += theme._REQUERY_INTERVAL
        watcher.tick()
        watcher.handle_reply(payload)
        assert _selected_bg(console) == expected.styles[theme.SELECTED]
        assert console.get_style(theme.GAUGE_TRACK) == expected.styles[theme.GAUGE_TRACK]


def test_watcher_does_not_stack_themes_on_repeated_switches():
    watcher, console, _, _ = _watcher()
    for payload in ["11;rgb:ffff/ffff/ffff", "11;rgb:0000/0000/0000"] * 5:
        watcher.handle_reply(payload)
    # Each switch replaced the previous one: one pop gets back to the base.
    console.pop_theme()
    assert _selected_bg(console) == NEUTRAL.styles[theme.SELECTED]


def test_watcher_does_not_resend_while_a_query_is_outstanding():
    watcher, _, out, clock = _watcher()
    watcher.tick()
    clock.now += theme._REPLY_TIMEOUT / 2
    watcher.tick()
    assert out.getvalue().count(theme.BACKGROUND_QUERY) == 1
    assert watcher.awaiting_reply


def test_watcher_gives_up_on_a_terminal_that_never_answers():
    watcher, console, out, clock = _watcher()
    watcher.tick()
    for _ in range(10):
        clock.now += theme._REPLY_TIMEOUT + theme._REQUERY_INTERVAL
        watcher.tick()
    assert out.getvalue().count(theme.BACKGROUND_QUERY) == 1
    assert not watcher.awaiting_reply
    assert _selected_bg(console) == NEUTRAL.styles[theme.SELECTED]


def test_watcher_keeps_requerying_a_terminal_that_answers():
    watcher, _, out, clock = _watcher()
    for _ in range(3):
        watcher.tick()
        watcher.handle_reply("11;rgb:ffff/ffff/ffff")
        clock.now += theme._REQUERY_INTERVAL
    assert out.getvalue().count(theme.BACKGROUND_QUERY) == 3


def test_watcher_ignores_unrelated_osc_payloads():
    watcher, console, _, _ = _watcher()
    watcher.tick()
    watcher.handle_reply("10;rgb:ffff/ffff/ffff")
    assert watcher.awaiting_reply
    assert _selected_bg(console) == NEUTRAL.styles[theme.SELECTED]


def test_disabled_watcher_never_writes_to_the_terminal():
    watcher, _, out, clock = _watcher(enabled=False)
    for _ in range(3):
        watcher.tick()
        clock.now += 60
    assert out.getvalue() == ""


def _fixture_state(monkeypatch) -> AppState:
    def fake_fetch(*args, **kwargs):
        sections = {name: (FIXTURES / f"{name}.txt").read_text() for name in ("sinfo", "squeue", "sprio", "sshare")}
        return RawSections(partitions="", now="2026-09-11T09:00:00", **sections)

    monkeypatch.setattr(ssh_client, "fetch_raw_local", fake_fetch)
    state = AppState(
        host=None, slurm_user="frank.example", local=True,
        selector=PartitionSelector(mode="prefix", arg=""), interval=30.0,
    )
    for key, value in cli.compute_refresh(None, state.slurm_user, state.selector, True).items():
        setattr(state, key, value)
    return state


def _render_segments(state: AppState, palette):
    console = Console(file=io.StringIO(), width=160, force_terminal=True, color_system="truecolor", theme=NEUTRAL)
    console.push_theme(palette)
    return [seg for line in console.render_lines(render(state, height=30)) for seg in line]


@pytest.mark.parametrize("palette", [DARK, LIGHT, NEUTRAL])
@pytest.mark.parametrize("screen", ["overview", "nodes", "users", "jobs", "myjobs"])
def test_every_screen_renders_under_every_palette(monkeypatch, palette, screen):
    state = _fixture_state(monkeypatch)
    state.screen = screen
    # On the users screen, row 2 is frank.example: both "me" and selected.
    state.selected = 2
    assert _render_segments(state, palette)


def test_light_palette_paints_no_dark_backgrounds(monkeypatch):
    state = _fixture_state(monkeypatch)
    dark_bgs = {DARK.styles[name].bgcolor for name in (theme.SELECTED, theme.GROUP_HEADER, theme.GAUGE_TRACK)}
    for screen in ["overview", "users", "jobs"]:
        state.screen = screen
        state.selected = 1
        bgs = {seg.style.bgcolor for seg in _render_segments(state, LIGHT) if seg.style}
        assert bgs & dark_bgs == set()
        if screen != "overview":
            assert LIGHT.styles[theme.SELECTED].bgcolor in bgs
