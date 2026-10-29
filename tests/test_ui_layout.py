import io

import pytest
from rich.cells import cell_len
from rich.console import Console

from slurmmon import theme
from slurmmon.aggregate import MyJobRow
from slurmmon.parse import GresCount, Job, PrioRow
from slurmmon.ssh_client import PartitionSelector
from slurmmon.state import AppState
from slurmmon.ui import _MY_JOBS_COLS, _Col, _FittedTable, _fit_widths, _Row, _share_out, my_jobs_panel, render

# The shape of a real list that broke: mostly running jobs on one partition,
# plus a few pending ones with a 4-partition list and a long reason.
LONG_PARTITIONS = "rleap_gpu_24gb,rleap_gpu_48gb,rleap_cpu,rleap_cpu_modern"


def _job(job_id: str, name: str, state: str = "RUNNING", partition: str = "rleap_gpu_24gb") -> Job:
    running = state == "RUNNING"
    return Job(
        job_id, "alice", partition, state, 1, 16, 32768, GresCount(shard=4),
        "7:27:40" if running else "0:00", "None" if running else "Dependency",
        "N/A", "cn-01" if running else "", "2026-10-26T18:00:00", name,
    )


def _state(screen: str = "myjobs", **overrides) -> AppState:
    state = AppState(host="h", slurm_user="alice", selector=PartitionSelector(mode="prefix", arg=""), interval=30.0)
    rows = [MyJobRow(job=_job(f"49326{30 + i // 5}_{i % 5}", "dias-val-satellite"), fairshare=0.34) for i in range(40)]
    rows += [MyJobRow(job=_job(f"494944{i}_{i}", "ds-e4-exists")) for i in range(10)]
    rows += [
        MyJobRow(
            job=_job(f"49298{58 + i}_0", "dias-val-blocks-3", state="PENDING", partition=LONG_PARTITIONS),
            prio=PrioRow(f"49298{58 + i}_0", 100, 0, 10, 3400, 0),
        )
        for i in range(4)
    ]
    state.my_jobs = rows
    state.jobs = [r.job for r in rows]
    state.server_now = "2026-10-28T12:00:00"
    state.screen = screen
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _frame(state: AppState, width: int, height: int) -> list[str]:
    console = Console(file=io.StringIO(), width=width, height=height, theme=theme.NEUTRAL)
    rendered = render(state, height=height, width=width)
    return ["".join(seg.text for seg in line) for line in console.render_lines(rendered, console.options, pad=False)]


def test_share_out_is_proportional_to_weights():
    assert _share_out(30, [2.0, 1.0], [0, 0], [100, 100]) == [20, 10]


def test_share_out_gives_what_a_column_cannot_use_to_the_others():
    # A column whose content is only 5 wide shouldn't sit on 15 while the
    # other one's content is being cut off.
    assert _share_out(30, [1.0, 1.0], [0, 0], [5, 100]) == [5, 25]


def test_share_out_always_fills_the_budget_even_past_every_cap():
    widths = _share_out(30, [1.0, 1.0], [0, 0], [5, 5])
    assert widths == [15, 15]


def test_share_out_respects_floors_when_they_fit():
    assert _share_out(20, [1.0, 10.0], [8, 0], [100, 100]) == [8, 12]


def test_share_out_drops_floors_when_they_do_not_fit():
    widths = _share_out(5, [1.0, 1.0], [4, 4], [100, 100])
    assert sum(widths) == 5


def test_share_out_with_no_room_at_all():
    assert _share_out(-7, [1.0, 1.0], [4, 4], [10, 10]) == [0, 0]


def test_fit_widths_sizes_fixed_columns_to_their_widest_cell():
    cols = (_Col("ID", fixed=True), _Col("Name"), _Col("T", fixed=True))
    rows = [_Row(["1", "a" * 50, "1:00"]), _Row(["12345", "b", "10:00:00"])]
    widths = _fit_widths(cols, rows, 30)
    assert widths[0] == 5 and widths[2] == 8
    assert sum(widths) + 2 * 2 == 30  # the flexible column takes the rest, gaps included


def test_fit_widths_weighs_columns_by_their_filled_cells_only():
    # B is blank on most rows (as group header rows leave most columns),
    # but where it has content it's as wide as A's: they should grow alike.
    cols = (_Col("A"), _Col("B"))
    rows = [_Row(["a" * 10, "b" * 10])] * 5 + [_Row(["a" * 10, ""])] * 6
    assert _fit_widths(cols, rows, 40 + 2) == [20, 20]


@pytest.mark.parametrize("width", [60, 80, 155, 250])
def test_fitted_table_spans_exactly_the_width_it_is_given(width):
    rows = [_Row(["4932630_0", "dias-val-satellite", LONG_PARTITIONS, "RUNNING", "16", "32G", "4 shard", "7:27:40", "x" * 90])]
    console = Console(file=io.StringIO(), width=width, theme=theme.NEUTRAL)
    lines = console.render_lines(_FittedTable(_MY_JOBS_COLS, rows, rows), console.options, pad=False)
    assert len(lines) == 2  # column headers + the one row: nothing wrapped
    assert all(sum(cell_len(seg.text) for seg in line) == width for line in lines)


@pytest.mark.parametrize("group_mode", ["headers", "off"])
@pytest.mark.parametrize("width,height", [(250, 50), (155, 50), (100, 30), (80, 24)])
def test_myjobs_fills_the_screen_exactly_with_the_last_job_reachable(group_mode, width, height):
    # Was: pending jobs' long reasons wrapped onto extra lines and the
    # footer's wrapping went uncounted, so the frame outgrew the terminal and
    # its bottom -- including the very row you'd scrolled to -- was cut off.
    state = _state(group_mode=group_mode, selected=10_000)
    lines = _frame(state, width, height)
    assert len(lines) == height
    assert "4929861_0" in "\n".join(lines)  # the last job, selected
    assert "[q] quit" in "\n".join(lines[-3:])


def test_myjobs_columns_do_not_move_when_scrolling():
    top = _frame(_state(selected=0), 155, 30)
    bottom = _frame(_state(selected=10_000), 155, 30)
    assert "4932630_0" in "\n".join(top) and "4929861_0" in "\n".join(bottom)
    # The column-header line: where each column starts and how wide it is.
    assert top[2] == bottom[2]


def test_myjobs_always_shows_fixed_columns_in_full():
    text = "\n".join(_frame(_state(selected=10_000), 90, 30))
    for whole in ("4929861_0", "PENDING", "RUNNING", "32G", "4 shard", "queued 1d18h"):
        assert whole in text
    assert "…" in text  # while the flexible ones (partitions, reason) got cut short


@pytest.mark.parametrize("width", [100, 155])
def test_overview_my_jobs_panel_truncates_instead_of_wrapping(width):
    # Was: at 155 columns each long reason wrapped onto 4 lines; at 100,
    # Rich squeezed every column down instead, JobID/State/Time included.
    state = _state(screen="overview")
    state.my_jobs = state.my_jobs[-3:]  # all pending, long partition list + reason
    console = Console(file=io.StringIO(), width=width, theme=theme.NEUTRAL)
    lines = ["".join(s.text for s in line) for line in console.render_lines(my_jobs_panel(state), console.options, pad=False)]
    assert len(lines) == 2 + 1 + 3  # borders + column headers + one line per job
    for whole in ("4929861_0", "PENDING", "4 shard", "queued 1d18h"):
        assert whole in lines[-2]


def test_overview_my_jobs_panel_with_no_jobs():
    state = _state(screen="overview", my_jobs=[])
    console = Console(file=io.StringIO(), width=100, theme=theme.NEUTRAL)
    text = "\n".join("".join(s.text for s in line) for line in console.render_lines(my_jobs_panel(state), console.options))
    assert "no jobs on the selected partitions" in text


@pytest.mark.parametrize("group_mode", ["headers", "off"])
@pytest.mark.parametrize("width,height", [(155, 50), (120, 22), (80, 24)])
def test_queue_fills_the_screen_exactly(group_mode, width, height):
    # Same height accounting as my jobs; in "headers" mode the queue also
    # used to draw an uncounted rule under every group header.
    lines = _frame(_state(screen="jobs", group_mode=group_mode, selected=10_000), width, height)
    assert len(lines) == height


def test_list_screens_record_list_length_for_key_handling():
    state = _state(group_mode="headers")
    _frame(state, 155, 30)
    assert state.list_length == len(state.my_jobs)
    state.group_mode = "summary"
    _frame(state, 155, 30)
    assert state.list_length == 3  # dias-val-satellite, ds-e4-exists, dias-val-blocks
