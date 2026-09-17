from slurmmon.aggregate import MyJobRow
from slurmmon.parse import GresCount, Job
from slurmmon.ssh_client import PartitionSelector
from slurmmon.state import AppState
from slurmmon.ui import _flatten_grouped, _group_by_prefix, _name_prefix, render_jobs, render_myjobs


def _job(job_id: str, user: str, state: str, name: str, gpu: int = 0, shard: int = 0) -> Job:
    return Job(
        job_id, user, "gpu_a", state, 1, 8, 8192, GresCount(gpu=gpu, shard=shard),
        "1:00" if state == "RUNNING" else "0:00",
        "None" if state == "RUNNING" else "Priority",
        "N/A", "cn-01" if state == "RUNNING" else "", "2026-01-01T00:00:00", name,
    )


def _make_state(**overrides) -> AppState:
    defaults = dict(
        host="h", slurm_user="alice", selector=PartitionSelector(mode="prefix", arg=""), interval=30.0,
    )
    defaults.update(overrides)
    return AppState(**defaults)


def test_name_prefix_strips_trailing_number_and_separator():
    assert _name_prefix("sweep-3") == "sweep"
    assert _name_prefix("sweep_007") == "sweep"
    assert _name_prefix("sweep7") == "sweep"


def test_name_prefix_leaves_names_without_trailing_digits_untouched():
    assert _name_prefix("conv-arm") == "conv-arm"
    assert _name_prefix("lingo-p3-train") == "lingo-p3-train"


def test_name_prefix_handles_empty_and_all_digit_names():
    assert _name_prefix("") == "(unnamed)"
    assert _name_prefix("123") == "123"


def test_group_by_prefix_aggregates_counts_and_resources():
    jobs = [
        _job("1", "alice", "RUNNING", "sweep-1", shard=4),
        _job("2", "alice", "PENDING", "sweep-2", shard=4),
        _job("3", "bob", "RUNNING", "sweep-3", gpu=1),
        _job("4", "bob", "RUNNING", "solo-job"),
    ]
    groups = _group_by_prefix(jobs)
    by_prefix = {g.prefix: g for g in groups}

    sweep = by_prefix["sweep"]
    assert sweep.count == 3
    assert sweep.running == 2
    assert sweep.pending == 1
    assert sweep.gpu == 1
    assert sweep.shard == 8
    assert sweep.users == ["alice", "bob"]

    solo = by_prefix["solo-job"]
    assert solo.count == 1

    # sorted by count descending, then alphabetically.
    assert groups[0].prefix == "sweep"


def test_myjobs_summary_mode_populates_group_prefixes_not_job_ids():
    state = _make_state()
    state.my_jobs = [
        MyJobRow(job=_job("1", "alice", "RUNNING", "sweep-1")),
        MyJobRow(job=_job("2", "alice", "RUNNING", "sweep-2")),
        MyJobRow(job=_job("3", "alice", "RUNNING", "solo-job")),
    ]
    state.group_mode = "summary"

    render_myjobs(state, height=40)

    assert state.current_list_is_groups is True
    assert set(state.current_list_job_ids) == {"sweep", "solo-job"}


def test_myjobs_drill_down_filters_to_matching_prefix_only():
    state = _make_state()
    state.my_jobs = [
        MyJobRow(job=_job("1", "alice", "RUNNING", "sweep-1")),
        MyJobRow(job=_job("2", "alice", "RUNNING", "sweep-2")),
        MyJobRow(job=_job("3", "alice", "RUNNING", "solo-job")),
    ]
    state.group_mode = "summary"
    state.name_filter = "sweep"  # simulates Enter having been pressed on the "sweep" group

    render_myjobs(state, height=40)

    # drilled view must be flat (real job ids), restricted to the group.
    assert state.current_list_is_groups is False
    assert set(state.current_list_job_ids) == {"1", "2"}


def test_myjobs_headers_mode_keeps_real_job_ids_selectable_in_group_order():
    state = _make_state()
    state.my_jobs = [
        MyJobRow(job=_job("2", "alice", "RUNNING", "sweep-2")),
        MyJobRow(job=_job("1", "alice", "RUNNING", "sweep-1")),
        MyJobRow(job=_job("3", "alice", "RUNNING", "solo-job")),
    ]
    state.group_mode = "headers"

    render_myjobs(state, height=40)

    # headers mode never hides jobs behind a group id -- Enter must still
    # reach a real job.
    assert state.current_list_is_groups is False
    # "sweep" (2 jobs) sorts before "solo-job" (1 job); within a group, jobs
    # sort by job_id for stable, readable ordering regardless of input order.
    assert state.current_list_job_ids == ["1", "2", "3"]


def test_jobs_screen_summary_mode_includes_users_and_resets_on_off_render():
    state = _make_state()
    state.jobs = [
        _job("1", "alice", "RUNNING", "train-1"),
        _job("2", "bob", "RUNNING", "train-2"),
        _job("3", "carol", "PENDING", "infer"),
    ]
    state.group_mode = "summary"

    render_jobs(state, height=40)
    assert state.current_list_is_groups is True
    assert set(state.current_list_job_ids) == {"train", "infer"}

    state.group_mode = "off"
    render_jobs(state, height=40)
    assert state.current_list_is_groups is False
    assert set(state.current_list_job_ids) == {"1", "2", "3"}


def test_jobs_screen_headers_mode_lists_every_job_across_groups():
    state = _make_state()
    state.jobs = [
        _job("1", "alice", "RUNNING", "train-1"),
        _job("2", "bob", "RUNNING", "train-2"),
        _job("3", "carol", "PENDING", "infer"),
    ]
    state.group_mode = "headers"

    render_jobs(state, height=40)
    assert state.current_list_is_groups is False
    assert set(state.current_list_job_ids) == {"1", "2", "3"}


def test_flatten_grouped_selection_skips_over_header_rows():
    jobs = [
        _job("2", "alice", "RUNNING", "sweep-2"),
        _job("1", "alice", "RUNNING", "sweep-1"),
        _job("3", "bob", "RUNNING", "solo"),
    ]
    display, job_positions = _flatten_grouped(jobs, lambda j: j)

    # display interleaves ("header", group) with ("row", job) entries; the
    # positions recorded for each selectable job must point at "row" entries.
    kinds_at_positions = [display[p][0] for p in job_positions]
    assert kinds_at_positions == ["row", "row", "row"]

    # "sweep" group (2 jobs) is listed before "solo" (1 job); the header for
    # each group appears immediately before its first job.
    assert display[0] == ("header", display[0][1])
    assert display[0][1].prefix == "sweep"
    assert display[job_positions[0]][1].job_id == "1"  # sorted by job_id within the group
    assert display[job_positions[1]][1].job_id == "2"


def test_group_name_prefix_column_is_width_bounded():
    # Real production job names run to 100+ chars (seen in practice, e.g.
    # sweep/config-encoding names). Rich's Table always stays within the
    # console width regardless (it auto-ellipsizes a ratio column too), so
    # an unbounded column was never a hard *overflow* bug -- the real
    # problem was proportional: as the only ratio column, it claimed nearly
    # all free width before truncating, squeezing CPU/Mem/GPU/etc. down to
    # their bare minimums. Assert the column is explicitly capped instead
    # of relying on Rich's default sizing.
    from rich.table import Table

    from slurmmon.ui import _group_columns

    table = Table()
    _group_columns(table, with_users=True)
    name_col = table.columns[0]
    assert name_col.max_width == 40
    assert name_col.overflow == "ellipsis"
