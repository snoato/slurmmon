from slurmmon.aggregate import MyJobRow
from slurmmon.parse import GresCount, Job
from slurmmon.ssh_client import PartitionSelector
from slurmmon.state import AppState
from slurmmon.ui import render_jobs, render_myjobs


def _job(job_id: str, user: str, state: str) -> Job:
    return Job(
        job_id, user, "gpu_a", state, 1, 8, 8192, GresCount(),
        "1:00" if state == "RUNNING" else "0:00",
        "None" if state == "RUNNING" else "Priority",
        "N/A", "cn-01" if state == "RUNNING" else "", "2026-01-01T00:00:00",
    )


def _make_state(**overrides) -> AppState:
    defaults = dict(
        host="h", slurm_user="alice", selector=PartitionSelector(mode="prefix", arg=""), interval=30.0,
    )
    defaults.update(overrides)
    return AppState(**defaults)


def test_myjobs_filter_narrows_current_list_job_ids():
    state = _make_state()
    state.my_jobs = [
        MyJobRow(job=_job("1", "alice", "RUNNING")),
        MyJobRow(job=_job("2", "alice", "PENDING")),
        MyJobRow(job=_job("3", "alice", "RUNNING")),
    ]

    state.job_filter = "ALL"
    render_myjobs(state, height=40)
    assert state.current_list_job_ids == ["1", "2", "3"]

    state.job_filter = "RUNNING"
    state.selected = 0
    render_myjobs(state, height=40)
    assert state.current_list_job_ids == ["1", "3"]

    state.job_filter = "PENDING"
    render_myjobs(state, height=40)
    assert state.current_list_job_ids == ["2"]


def test_myjobs_selection_survives_render_and_clamps():
    state = _make_state()
    state.my_jobs = [MyJobRow(job=_job(str(i), "alice", "RUNNING")) for i in range(5)]
    state.selected = 999  # out of range on purpose

    render_myjobs(state, height=40)

    # the render must clamp selected back into range so a subsequent Enter
    # lookup (state.current_list_job_ids[state.selected]) can't index error.
    assert 0 <= state.selected < len(state.current_list_job_ids)


def test_jobs_screen_populates_current_list_job_ids_in_display_order():
    state = _make_state()
    state.jobs = [
        _job("10", "bob", "PENDING"),
        _job("20", "alice", "RUNNING"),
        _job("30", "carol", "RUNNING"),
    ]

    render_jobs(state, height=40)

    # render_jobs sorts running by user then appends pending by job id --
    # current_list_job_ids must match exactly what's displayed, since Enter
    # indexes into it directly.
    assert state.current_list_job_ids == ["20", "30", "10"]
