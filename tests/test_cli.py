import argparse

import pytest

from slurmmon import cli
from slurmmon.cli import main, resolve_config


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
