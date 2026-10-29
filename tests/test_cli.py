import argparse
import threading
from collections import deque

import pytest

from slurmmon import cli
from slurmmon.cli import main, resolve_config
from slurmmon.ssh_client import PartitionSelector
from slurmmon.state import AppState


def _args(**overrides) -> argparse.Namespace:
    base = {"local": False, "host": "", "slurm_user": ""}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_remote_requires_host():
    with pytest.raises(ValueError, match="--host"):
        resolve_config(_args(slurm_user="alice"))


def test_remote_requires_slurm_user():
    with pytest.raises(ValueError, match="--slurm-user"):
        resolve_config(_args(host="cluster.example.edu"))


def test_remote_resolves_with_both_set():
    host, user, local = resolve_config(_args(host="cluster.example.edu", slurm_user="alice"))
    assert (host, user, local) == ("cluster.example.edu", "alice", False)


def test_local_does_not_require_host():
    host, user, local = resolve_config(_args(local=True, slurm_user="alice"))
    assert host is None
    assert user == "alice"
    assert local is True


def test_local_defaults_slurm_user_to_local_username(monkeypatch):
    monkeypatch.setattr("slurmmon.cli.getpass.getuser", lambda: "whoami-result")
    host, user, local = resolve_config(_args(local=True))
    assert host is None
    assert user == "whoami-result"
    assert local is True


def test_local_ignores_host_even_if_set():
    host, user, local = resolve_config(_args(local=True, host="irrelevant", slurm_user="alice"))
    assert host is None


def test_ctrl_c_during_once_exits_cleanly(monkeypatch):
    def _raise(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "compute_refresh", _raise)
    assert main(["--local", "--slurm-user", "alice", "--once"]) == 130


def test_ctrl_c_during_interactive_exits_cleanly(monkeypatch):
    def _raise(state):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_interactive", _raise)
    assert main(["--local", "--slurm-user", "alice"]) == 0


def _key_state(**overrides) -> AppState:
    state = AppState(host="h", slurm_user="alice", selector=PartitionSelector(mode="prefix", arg=""), interval=30.0)
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _apply(state: AppState, keys: list[str]) -> deque:
    pending = deque(keys)
    cli._apply_keys(state, pending, lock=threading.Lock(), refresh_now=threading.Event(), stop_event=threading.Event())
    return pending


def test_a_burst_of_moves_is_applied_in_one_go_and_clamped_key_by_key():
    # At the bottom of a 10-row list: overshooting by 10 then coming back 3
    # must land 3 rows above the end, exactly as if each key had been drawn.
    state = _key_state(screen="myjobs", list_length=10, selected=8)
    assert _apply(state, ["DOWN"] * 10 + ["UP"] * 3) == deque()
    assert state.selected == 6

    _apply(state, ["PGUP"] * 5)
    assert state.selected == 0


def test_a_key_that_switches_lists_holds_back_the_rest_until_it_is_drawn():
    # The DOWNs are meant for the my jobs list, whose length (and job ids,
    # for Enter) only the next render knows.
    state = _key_state(screen="overview", list_length=3, selected=0)
    pending = _apply(state, ["m", "DOWN", "DOWN"])
    assert state.screen == "myjobs"
    assert state.selected == 0
    assert pending == deque(["DOWN", "DOWN"])


def test_moves_and_render_work_together_end_to_end():
    from slurmmon.aggregate import MyJobRow
    from slurmmon.parse import GresCount, Job
    from slurmmon.ui import render

    state = _key_state(screen="myjobs", group_mode="off")
    state.my_jobs = [
        MyJobRow(job=Job(str(i), "alice", "p", "RUNNING", 1, 1, 1024, GresCount(), "1:00", "None", "N/A", "n", "", "j"))
        for i in range(5)
    ]
    render(state, height=40, width=120)
    _apply(state, ["DOWN"] * 1000)
    assert state.selected == 4
    render(state, height=40, width=120)
    assert state.current_list_job_ids[state.selected] == "4"


def test_detail_overlay_scroll_is_clamped_key_by_key():
    state = _key_state(detail_job_id="42", detail_max_scroll=5, detail_scroll=0)
    _apply(state, ["DOWN"] * 20 + ["UP"])
    assert state.detail_scroll == 4


def test_quit_stops_handling_the_rest_of_a_burst():
    state = _key_state(screen="myjobs", list_length=10, selected=0)
    stop_event = threading.Event()
    pending = deque(["q", "DOWN"])
    cli._apply_keys(state, pending, lock=threading.Lock(), refresh_now=threading.Event(), stop_event=stop_event)
    assert stop_event.is_set()
    assert state.selected == 0
