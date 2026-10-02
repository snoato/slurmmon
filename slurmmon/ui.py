"""Rich-based rendering: one glanceable overview screen plus keyboard-
switchable detail screens (nodes / users / jobs / my jobs), an on-demand
job-detail overlay (Enter on a job row), and a group-by-name-prefix view
of jobs/my jobs ('g') with drill-down back into a flat job list."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from rich.bar import Bar
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import theme
from .aggregate import PartitionStats, UserUsage
from .parse import Job
from .reasons import explain_reason
from .state import AppState

_SLURM_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

_BAR_WIDTH = 18


def _pct_color(pct: float) -> str:
    if pct >= 0.9:
        return "red"
    if pct >= 0.7:
        return "yellow"
    return "green"


class _Gauge(Bar):
    """A Bar whose empty-track color comes from the console's theme at
    render time (theme.GAUGE_TRACK), so it can follow the terminal's
    light/dark background instead of being baked in when the frame is built."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        track = console.get_style(theme.GAUGE_TRACK).bgcolor
        # reverse=False: a selected row can be reverse video (theme.NEUTRAL),
        # which would otherwise swap fill and track -- 7% drawn as 93%.
        self.style = Style(color=self.style.color, bgcolor=track, reverse=False)
        return super().__rich_console__(console, options)


def _gauge(pct: float, width: int = _BAR_WIDTH) -> RenderableType:
    color = _pct_color(pct)
    bar = _Gauge(size=1.0, begin=0, end=pct, width=width, color=color)
    label = Text(f" {pct * 100:4.0f}%", style=color)
    return _inline(bar, label)


def _inline(*renderables: RenderableType) -> RenderableType:
    t = Table.grid(padding=(0, 0))
    t.add_row(*renderables)
    return t


def _fmt_gb(mb: float) -> str:
    return f"{mb / 1024:.0f}G"


def _fmt_ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    return f"{seconds / 60:.1f}m ago"


def _parse_slurm_timestamp(raw: str) -> datetime | None:
    raw = raw.strip()
    if not raw or raw.upper() in ("N/A", "UNKNOWN"):
        return None
    try:
        return datetime.strptime(raw, _SLURM_TIME_FORMAT)
    except ValueError:
        return None


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def _queued_for(submit_time: str, server_now: str | None) -> str | None:
    """How long a pending job has been queued, using the *query host's* own
    clock as "now" -- not the local machine's -- so this stays correct even
    when slurmmon is run (over SSH) from a different timezone than the
    cluster."""
    submitted = _parse_slurm_timestamp(submit_time)
    if submitted is None:
        return None
    now = _parse_slurm_timestamp(server_now) if server_now else None
    if now is None:
        now = datetime.now()
    return _fmt_duration((now - submitted).total_seconds())


# Strips one trailing run of digits (and an optional preceding separator)
# off a job name, e.g. "sweep-3" / "sweep_007" / "sweep7" -> "sweep". A name
# with no trailing digits (e.g. "conv-arm") is left as-is, so jobs sharing
# one literal name still group together trivially.
_TRAILING_NUM_RE = re.compile(r"^(.*?)[-_.]?\d+$")


def _name_prefix(name: str) -> str:
    name = (name or "").strip() or "(unnamed)"
    m = _TRAILING_NUM_RE.match(name)
    if m and m.group(1):
        return m.group(1)
    return name


@dataclass
class _JobGroup:
    prefix: str
    jobs: list[Job] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.jobs)

    @property
    def running(self) -> int:
        return sum(1 for j in self.jobs if j.is_running)

    @property
    def pending(self) -> int:
        return sum(1 for j in self.jobs if j.is_pending)

    @property
    def cpus(self) -> int:
        return sum(j.cpus for j in self.jobs)

    @property
    def mem_mb(self) -> float:
        return sum(j.mem_mb for j in self.jobs)

    @property
    def gpu(self) -> int:
        return sum(j.gres.gpu for j in self.jobs)

    @property
    def shard(self) -> int:
        return sum(j.gres.shard for j in self.jobs)

    @property
    def users(self) -> list[str]:
        return sorted({j.user for j in self.jobs})


def _group_by_prefix(jobs: list[Job]) -> list[_JobGroup]:
    groups: dict[str, _JobGroup] = {}
    for j in jobs:
        prefix = _name_prefix(j.name)
        groups.setdefault(prefix, _JobGroup(prefix=prefix)).jobs.append(j)
    return sorted(groups.values(), key=lambda g: (-g.count, g.prefix))


def _gpu_summary(gpu: int, shard: int) -> str:
    bits = []
    if gpu:
        bits.append(f"{gpu}g")
    if shard:
        bits.append(f"{shard}sh")
    return " ".join(bits) if bits else ""


GROUP_MODES = ["off", "headers", "summary"]
_GROUP_MODE_LABELS = {
    "off": "off (per-job list)",
    "headers": "headers (per-job list with group section headers)",
    "summary": "summary (one aggregated row per group)",
}


def _flatten_grouped(rows: list, get_job) -> tuple[list[tuple[str, object]], list[int]]:
    """Group arbitrary row objects by `get_job(row).name` prefix for the
    "headers" display mode: unlike "summary" mode, individual rows stay
    visible, just with a header row inserted before each group.

    Returns (display, job_positions):
    - display: ("header", _JobGroup) / ("row", row) entries in render order
    - job_positions[i]: display-index of the i-th real row, so selection
      (state.selected) can keep addressing jobs 0..N-1 directly, ignoring
      the header rows interspersed among them for viewport purposes.
    """
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(_name_prefix(get_job(r).name), []).append(r)
    ordered = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    display: list[tuple[str, object]] = []
    job_positions: list[int] = []
    for prefix, rows_in_group in ordered:
        rows_in_group = sorted(rows_in_group, key=lambda r: get_job(r).job_id)
        group = _JobGroup(prefix=prefix, jobs=[get_job(r) for r in rows_in_group])
        display.append(("header", group))
        for r in rows_in_group:
            job_positions.append(len(display))
            display.append(("row", r))
    return display, job_positions


def _scroll_window_headlined(
    display: list, job_positions: list[int], scroll: int, height: int | None, row_delta: int, selected: int
) -> tuple[list, int, int, int]:
    """Like `_scroll_window_with_selection`, but the thing being counted for
    selection (real job rows, via `job_positions`) and the thing being
    windowed for display (`display`, header rows included) differ in length
    -- header rows take up viewport space but are never selectable."""
    total_jobs = len(job_positions)
    if total_jobs == 0:
        return [], 0, 0, 0

    selected = min(max(0, selected), total_jobs - 1)
    auto_rows = (height or _DEFAULT_HEIGHT) - _LIST_SCREEN_CHROME
    visible_rows = max(_MIN_VISIBLE_ROWS, min(auto_rows, auto_rows + row_delta))
    max_offset = max(0, len(display) - visible_rows)

    offset = min(max(0, scroll), max_offset)
    anchor = job_positions[selected]
    if anchor < offset:
        offset = anchor
    elif anchor >= offset + visible_rows:
        offset = anchor - visible_rows + 1
    offset = min(max(0, offset), max_offset)

    return display[offset : offset + visible_rows], offset, total_jobs, selected


# A fixed white, not ANSI "white": light terminal themes commonly remap that
# to a mid grey (it's meant as a text color on *their* background), which
# then washes out against the blue/red badge.
_BADGE_TEXT = "grey100"


def header(state: AppState) -> RenderableType:
    location = "local" if state.local else state.host
    parts = [
        Text(" slurmmon ", style=f"bold {_BADGE_TEXT} on blue"),
        Text(f" {location} "),
        Text(f"partitions={state.partitions_desc} ", style="dim"),
        Text(f"user={state.slurm_user} ", style="dim"),
    ]
    if state.fetching:
        parts.append(Text(" fetching... ", style="cyan"))
    if state.last_error:
        parts.append(Text(f" ⚠ {state.last_error} ", style=f"bold {_BADGE_TEXT} on red"))
    elif state.is_stale and state.last_success is not None:
        parts.append(Text(f" stale ({_fmt_ago(state.age_seconds)}) ", style="bold black on yellow"))
    else:
        parts.append(Text(f" updated {_fmt_ago(state.age_seconds)} ", style="dim"))
    parts.append(Text(f" every {state.interval:g}s ", style="dim"))
    row = Table.grid(expand=True)
    row.add_column(ratio=1)
    row.add_row(Text.assemble(*parts))
    return row


def footer(state: AppState) -> RenderableType:
    bits = ["[o] overview", "[n] nodes", "[u] users", "[j] jobs", "[m] my jobs"]
    if state.screen != "overview":
        bits.append("[↑/↓ PgUp/PgDn]")
        if state.screen in ("jobs", "myjobs"):
            if state.name_filter is not None:
                bits.append("[Enter] info")
                bits.append("[g] back to groups")
            else:
                bits.append("[Enter] drill down" if state.group_mode == "summary" else "[Enter] info")
                bits.append(f"[g] group: {state.group_mode} → cycle")
        if state.screen == "myjobs":
            bits.append(f"[f] state filter: {state.job_filter} → cycle")
        bits.append("[[/]] size")
    bits += ["[+/-] interval", "[r] refresh", "[q] quit"]
    return Text("  ".join(bits), style="dim", justify="center")


def _my_jobs_columns(table: Table) -> None:
    table.add_column("JobID", style="bold", no_wrap=True)
    table.add_column("Name", no_wrap=True, max_width=18, overflow="ellipsis")
    table.add_column("Partition", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Time", no_wrap=True)
    table.add_column("Reason / Priority", ratio=1, min_width=16)


def _my_job_row(table: Table, row, state: AppState, *, highlight: bool = False) -> None:
    job = row.job
    state_style = {
        "RUNNING": "green",
        "PENDING": "yellow",
    }.get(job.state, "")
    gpu_str = ""
    if job.gres.gpu:
        gpu_str = f"{job.gres.gpu} gpu"
    elif job.gres.shard:
        gpu_str = f"{job.gres.shard} shard"

    if job.is_running:
        time_col = job.time_used
        reason_col = f"fairshare {row.fairshare:.2f}" if row.fairshare is not None else ""
    else:
        queued = _queued_for(job.submit_time, state.server_now)
        time_col = f"queued {queued}" if queued else "-"

        reason_bits = [job.reason]
        explanation = explain_reason(job.reason)
        if explanation:
            reason_bits.append(f"— {explanation}")
        if row.prio is not None:
            reason_bits.append(
                f"(age={row.prio.age} fair={row.prio.fairshare} qos={row.prio.qos})"
            )
        if job.start_or_eta and job.start_or_eta not in ("N/A", "Unknown"):
            reason_bits.append(f"→ eta ~{job.start_or_eta}")
        reason_col = " ".join(reason_bits)

    table.add_row(
        job.job_id,
        job.name,
        job.partition,
        Text(job.state, style=state_style),
        str(job.cpus),
        _fmt_gb(job.mem_mb),
        gpu_str,
        time_col,
        reason_col,
        style=_SELECTED_ROW_STYLE if highlight else None,
    )


_OVERVIEW_MY_JOBS_LIMIT = 8
_JOB_FILTERS = ["ALL", "RUNNING", "PENDING"]


def my_jobs_panel(state: AppState) -> RenderableType:
    table = Table(expand=False, box=None, pad_edge=False)
    _my_jobs_columns(table)

    if not state.my_jobs:
        table.add_row(Text("no jobs on the selected partitions", style="dim"), "", "", "", "", "", "", "", "")
    shown = state.my_jobs[:_OVERVIEW_MY_JOBS_LIMIT]
    for row in shown:
        _my_job_row(table, row, state)

    remaining = len(state.my_jobs) - len(shown)
    title = f"My jobs ({state.slurm_user})"
    if remaining > 0:
        title += f"  (+{remaining} more, press 'm')"
    return Panel(table, title=title, border_style="blue")


def _group_columns(table: Table, *, with_users: bool) -> None:
    # Real job names can run very long (seen in practice: 100+ char sweep/
    # config-encoding names). Rich stays within the console width either
    # way, but as the only ratio column it would claim nearly all free
    # width before truncating, squeezing the numeric columns down to their
    # minimums -- bound it instead so those columns stay readable.
    table.add_column("Name prefix", style="bold", no_wrap=True, max_width=40, overflow="ellipsis")
    if with_users:
        table.add_column("Users", justify="right", no_wrap=True)
    table.add_column("Jobs", justify="right", no_wrap=True)
    table.add_column("Running", justify="right", no_wrap=True)
    table.add_column("Pending", justify="right", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)


def _group_row(table: Table, g: _JobGroup, *, with_users: bool, highlight: bool) -> None:
    cells = [g.prefix]
    if with_users:
        cells.append(str(len(g.users)))
    cells += [
        str(g.count),
        Text(str(g.running), style="green") if g.running else "0",
        Text(str(g.pending), style="yellow") if g.pending else "0",
        str(g.cpus),
        _fmt_gb(g.mem_mb),
        _gpu_summary(g.gpu, g.shard) or "-",
    ]
    table.add_row(*cells, style=_SELECTED_ROW_STYLE if highlight else None)


def _group_count_summary(g: _JobGroup) -> str:
    bits = [f"{g.count} jobs"]
    if g.running and g.pending:
        bits.append(f"{g.running} running / {g.pending} pending")
    return "  —  ".join(bits)


def _add_myjobs_header_row(table: Table, g: _JobGroup) -> None:
    # Matches _my_jobs_columns' 9-column shape: JobID, Name, Partition,
    # State, CPU, Mem, GPU, Time, Reason/Priority. Aggregate sums land in
    # the same columns as the per-job values below them for easy comparison;
    # the free-text job/state-count summary goes in the wide ratio column.
    table.add_row(
        "", g.prefix, "", "",
        str(g.cpus), _fmt_gb(g.mem_mb), _gpu_summary(g.gpu, g.shard) or "-",
        "", _group_count_summary(g),
        style=_GROUP_HEADER_ROW_STYLE, end_section=True,
    )


def _add_queue_header_row(table: Table, g: _JobGroup) -> None:
    # Matches render_jobs' 10-column shape: JobID, Name, User, Partition,
    # State, CPU, Mem, GPU, Time/Reason, Nodes.
    users = f"{len(g.users)} users" if len(g.users) != 1 else g.users[0]
    table.add_row(
        "", g.prefix, users, "", "",
        str(g.cpus), _fmt_gb(g.mem_mb), _gpu_summary(g.gpu, g.shard) or "-",
        "", _group_count_summary(g),
        style=_GROUP_HEADER_ROW_STYLE, end_section=True,
    )


def _group_mode_title_tag(state: AppState) -> str:
    if state.name_filter is not None:
        return f"[name: {state.name_filter}*, 'g' for groups]"
    return f"[group: {state.group_mode}, 'g' to cycle]"


def render_myjobs(state: AppState, height: int | None = None) -> RenderableType:
    items = state.my_jobs
    if state.job_filter != "ALL":
        items = [r for r in items if r.job.state == state.job_filter]

    base = f"My jobs ({state.slurm_user}) [state: {state.job_filter}, 'f' to cycle]"

    if state.group_mode == "summary" and state.name_filter is None:
        groups = _group_by_prefix([r.job for r in items])
        shown, offset, total, selected = _scroll_window_with_selection(
            groups, state.scroll, height, state.row_delta, state.selected
        )
        state.scroll = offset
        state.selected = selected
        state.current_list_job_ids = [g.prefix for g in groups]
        state.current_list_is_groups = True

        table = Table(expand=False, box=None, pad_edge=False)
        _group_columns(table, with_users=False)
        for i, g in enumerate(shown, start=offset):
            _group_row(table, g, with_users=False, highlight=(i == selected))

        title = _list_title(
            f"{base} {_group_mode_title_tag(state)}", offset, len(shown), total, hint=_MOVE_AND_ENTER_HINT
        )
        return Group(header(state), Panel(table, title=title, border_style="blue"), footer(state))

    state.current_list_is_groups = False

    if state.group_mode == "headers" and state.name_filter is None:
        display, job_positions = _flatten_grouped(items, lambda r: r.job)
        shown, offset, total_jobs, selected = _scroll_window_headlined(
            display, job_positions, state.scroll, height, state.row_delta, state.selected
        )
        state.scroll = offset
        state.selected = selected
        # current_list_job_ids must be in the same order job_positions/selected
        # index into -- _flatten_grouped sorts each group's rows by job_id,
        # which is NOT necessarily `items`' original order, so this must come
        # from the flattened display, not from `items` directly.
        state.current_list_job_ids = [r.job.job_id for kind, r in display if kind == "row"]

        table = Table(expand=False, box=None, pad_edge=False)
        _my_jobs_columns(table)
        for abs_idx, (kind, payload) in enumerate(shown, start=offset):
            if kind == "header":
                _add_myjobs_header_row(table, payload)
            else:
                _my_job_row(table, payload, state, highlight=(abs_idx == job_positions[selected]))

        num_groups = sum(1 for kind, _ in display if kind == "header")
        title_base = f"{base} {_group_mode_title_tag(state)}"
        title = (
            f"{title_base} (none)"
            if total_jobs == 0
            else f"{title_base} ({num_groups} groups, {total_jobs} jobs -- {_MOVE_AND_ENTER_HINT})"
        )
        return Group(header(state), Panel(table, title=title, border_style="blue"), footer(state))

    if state.name_filter is not None:
        items = [r for r in items if _name_prefix(r.job.name) == state.name_filter]

    shown, offset, total, selected = _scroll_window_with_selection(
        items, state.scroll, height, state.row_delta, state.selected
    )
    state.scroll = offset
    state.selected = selected
    state.current_list_job_ids = [r.job.job_id for r in items]

    table = Table(expand=False, box=None, pad_edge=False)
    _my_jobs_columns(table)
    for i, row in enumerate(shown, start=offset):
        _my_job_row(table, row, state, highlight=(i == selected))

    title = _list_title(f"{base} {_group_mode_title_tag(state)}", offset, len(shown), total, hint=_MOVE_AND_ENTER_HINT)
    return Group(header(state), Panel(table, title=title, border_style="blue"), footer(state))


def _partition_row(table: Table, name: str, ps: PartitionStats) -> None:
    nodes_str = Text.assemble(
        (str(ps.nodes_idle), "green"), " idle / ",
        (str(ps.nodes_busy), "yellow"), " busy / ",
        (str(ps.nodes_unavail), "red"), " down",
    )
    gpu_cell = _gauge(ps.gpu_pct) if ps.gpu_avail_total > 0 else Text("-", style="dim")
    table.add_row(
        name,
        nodes_str,
        _inline(_gauge(ps.cpu_pct), Text(f"  {ps.cpus_used}/{ps.cpus_avail_total}", style="dim")),
        _inline(_gauge(ps.mem_pct), Text(f"  {_fmt_gb(ps.mem_used_mb)}/{_fmt_gb(ps.mem_avail_total_mb)}", style="dim")),
        _inline(gpu_cell, Text(f"  {ps.gpu_used:.1f}/{ps.gpu_avail_total:.0f}", style="dim") if ps.gpu_avail_total > 0 else Text("")),
        str(ps.pending_jobs),
    )


def partitions_panel(state: AppState) -> RenderableType:
    table = Table(expand=False, box=None, pad_edge=False)
    table.add_column("Partition", style="bold", ratio=1, min_width=16)
    table.add_column("Nodes", ratio=1, min_width=22)
    table.add_column("CPU", no_wrap=True)
    table.add_column("Memory", no_wrap=True)
    table.add_column("GPU (equiv)", no_wrap=True)
    table.add_column("Pending", justify="right", no_wrap=True)

    for name in sorted(state.partition_stats):
        _partition_row(table, name, state.partition_stats[name])
    table.add_section()
    if state.cluster_stats is not None:
        _partition_row(table, "TOTAL", state.cluster_stats)
    return Panel(table, title="Partitions", border_style="magenta")


def _user_row(table: Table, u: UserUsage, *, is_me: bool, selected: bool = False) -> None:
    # Applied per cell rather than folded into the row style: a theme style
    # name can't be combined with other style words in one style string.
    me_style = "bold cyan" if is_me else ""
    table.add_row(
        Text(u.user, style=me_style),
        Text(str(u.job_count), style=me_style),
        Text(f"{u.cpus} ({u.cpu_share * 100:.0f}%)", style=me_style),
        Text(f"{_fmt_gb(u.mem_mb)} ({u.mem_share * 100:.0f}%)", style=me_style),
        Text(f"{u.gpu_equiv:.1f} ({u.gpu_share * 100:.0f}%)", style=me_style),
        _gauge(u.dominant_share, width=14),
        style=_SELECTED_ROW_STYLE if selected else None,
    )


def top_users_panel(state: AppState, limit: int = 12) -> RenderableType:
    table = Table(expand=False, box=None, pad_edge=False)
    table.add_column("User", style="bold", ratio=1, min_width=12)
    table.add_column("Jobs", justify="right", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Dominant share", no_wrap=True)

    shown = state.user_usage[:limit]
    for u in shown:
        _user_row(table, u, is_me=(u.user == state.slurm_user))

    remaining = len(state.user_usage) - len(shown)
    title = "Who's using the cluster"
    if remaining > 0:
        title += f"  (+{remaining} more, press 'u')"
    return Panel(table, title=title, border_style="green")


_LIST_SCREEN_CHROME = 8  # header(1) + panel top(1) + table top(1) + col header(1) + header sep(1) + table bottom(1) + panel bottom(1) + footer(1)
_DEFAULT_HEIGHT = 40  # fallback when no real terminal size is available (e.g. --once)
_MIN_VISIBLE_ROWS = 3


def _scroll_window(items: list, scroll: int, height: int | None, row_delta: int = 0) -> tuple[list, int, int]:
    """Return (visible_items, offset, total), clamping `scroll` into range.

    `row_delta` is a user-controlled +/- nudge on top of the terminal-height-
    derived page size (see the '['/']' keys) -- positive shows more rows,
    negative shows fewer, both still clamped to sane bounds.
    """
    total = len(items)
    auto_rows = (height or _DEFAULT_HEIGHT) - _LIST_SCREEN_CHROME
    # row_delta only ever shrinks from the terminal-height-derived max (a
    # negative "compact this" dial) -- it can't grow past what actually fits
    # in the real terminal, since this is a single fixed-size alt-screen frame.
    visible_rows = max(_MIN_VISIBLE_ROWS, min(auto_rows, auto_rows + row_delta))
    max_offset = max(0, total - visible_rows)
    offset = min(max(0, scroll), max_offset)
    return items[offset : offset + visible_rows], offset, total


def _scroll_window_with_selection(
    items: list, scroll: int, height: int | None, row_delta: int, selected: int
) -> tuple[list, int, int, int]:
    """Like `_scroll_window`, but also keeps `selected` (a cursor row,
    absolute index into `items`) inside the visible window, nudging the
    offset to follow it -- and returns the *clamped* selected index too, so
    the caller can persist it back onto AppState (selection must never
    silently drift out of range between renders, or an Enter-triggered
    lookup by index could go stale/out of bounds)."""
    total = len(items)
    if total == 0:
        return items, 0, 0, 0

    auto_rows = (height or _DEFAULT_HEIGHT) - _LIST_SCREEN_CHROME
    visible_rows = max(_MIN_VISIBLE_ROWS, min(auto_rows, auto_rows + row_delta))
    max_offset = max(0, total - visible_rows)

    selected = min(max(0, selected), total - 1)
    offset = min(max(0, scroll), max_offset)
    if selected < offset:
        offset = selected
    elif selected >= offset + visible_rows:
        offset = selected - visible_rows + 1
    offset = min(max(0, offset), max_offset)

    return items[offset : offset + visible_rows], offset, total, selected


# Theme style names, not literal colors: both paint a background, so the
# actual shade has to depend on the terminal's light/dark mode (see theme.py).
_SELECTED_ROW_STYLE = theme.SELECTED
_GROUP_HEADER_ROW_STYLE = theme.GROUP_HEADER


_MOVE_HINT = "↑/↓ PgUp/PgDn to move"
_MOVE_AND_ENTER_HINT = "↑/↓ PgUp/PgDn to move, Enter for info"


def _list_title(base: str, offset: int, shown: int, total: int, *, hint: str = _MOVE_HINT) -> str:
    if total == 0:
        return f"{base} (none)"
    if shown >= total:
        return f"{base} ({total})"
    return f"{base} ({offset + 1}-{offset + shown} of {total} -- {hint})"


def render_overview(state: AppState) -> RenderableType:
    return Group(
        header(state),
        my_jobs_panel(state),
        partitions_panel(state),
        top_users_panel(state),
        footer(state),
    )


def render_nodes(state: AppState, height: int | None = None) -> RenderableType:
    all_nodes = sorted(state.nodes, key=lambda n: (n.partition, n.name))
    shown, offset, total, selected = _scroll_window_with_selection(
        all_nodes, state.scroll, height, state.row_delta, state.selected
    )
    state.scroll = offset
    state.selected = selected

    table = Table(expand=False)
    table.add_column("Node", style="bold", no_wrap=True)
    table.add_column("Partition", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("CPU (alloc/total)", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("Gres", ratio=1, min_width=10)
    for i, n in enumerate(shown, start=offset):
        state_style = {"idle": "green", "busy": "yellow", "unavailable": "red"}[n.bucket]
        gres = ""
        if n.gres.gpu:
            gres = f"gpu:{n.gres.gpu_model or '?'}:{n.gres.gpu}"
        if n.gres.shard:
            gres += (", " if gres else "") + f"shard:{n.gres.gpu_model or '?'}:{n.gres.shard}"
        table.add_row(
            n.name,
            n.partition,
            Text(n.raw_state, style=state_style),
            f"{n.cpus_alloc}/{n.cpus_total}",
            _fmt_gb(n.mem_mb),
            gres or "-",
            style=_SELECTED_ROW_STYLE if i == selected else None,
        )
    title = _list_title("Nodes", offset, len(shown), total)
    return Group(header(state), Panel(table, title=title, border_style="magenta"), footer(state))


def render_users(state: AppState, height: int | None = None) -> RenderableType:
    shown, offset, total, selected = _scroll_window_with_selection(
        state.user_usage, state.scroll, height, state.row_delta, state.selected
    )
    state.scroll = offset
    state.selected = selected

    table = Table(expand=False)
    table.add_column("User", style="bold", ratio=1, min_width=12)
    table.add_column("Jobs", justify="right", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Dominant share", no_wrap=True)
    for i, u in enumerate(shown, start=offset):
        _user_row(table, u, is_me=(u.user == state.slurm_user), selected=(i == selected))
    title = _list_title("All users", offset, len(shown), total)
    return Group(header(state), Panel(table, title=title, border_style="green"), footer(state))


def _queue_columns(table: Table) -> None:
    table.add_column("JobID", style="bold", no_wrap=True)
    table.add_column("Name", no_wrap=True, max_width=18, overflow="ellipsis")
    table.add_column("User", no_wrap=True)
    table.add_column("Partition", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Time / Reason", no_wrap=True)
    table.add_column("Nodes", ratio=1, min_width=8)


def _queue_job_row(table: Table, j: Job, state: AppState, *, highlight: bool) -> None:
    state_style = {"RUNNING": "green", "PENDING": "yellow"}.get(j.state, "")
    gpu_str = f"{j.gres.gpu}g" if j.gres.gpu else (f"{j.gres.shard}sh" if j.gres.shard else "")
    time_or_reason = j.time_used if j.is_running else j.reason
    table.add_row(
        j.job_id,
        j.name,
        Text(j.user, style="bold cyan" if j.user == state.slurm_user else None),
        j.partition,
        Text(j.state, style=state_style),
        str(j.cpus),
        _fmt_gb(j.mem_mb),
        gpu_str,
        time_or_reason,
        j.nodelist or "-",
        style=_SELECTED_ROW_STYLE if highlight else None,
    )


def render_jobs(state: AppState, height: int | None = None) -> RenderableType:
    running = [j for j in state.jobs if j.is_running]
    pending = [j for j in state.jobs if j.is_pending]
    running.sort(key=lambda j: j.user)
    pending.sort(key=lambda j: j.job_id)
    all_jobs = running + pending

    base = "Queue (running + pending)"

    if state.group_mode == "summary" and state.name_filter is None:
        groups = _group_by_prefix(all_jobs)
        shown, offset, total, selected = _scroll_window_with_selection(
            groups, state.scroll, height, state.row_delta, state.selected
        )
        state.scroll = offset
        state.selected = selected
        state.current_list_job_ids = [g.prefix for g in groups]
        state.current_list_is_groups = True

        table = Table(expand=False)
        _group_columns(table, with_users=True)
        for i, g in enumerate(shown, start=offset):
            _group_row(table, g, with_users=True, highlight=(i == selected))

        title = _list_title(
            f"{base} {_group_mode_title_tag(state)}", offset, len(shown), total, hint=_MOVE_AND_ENTER_HINT
        )
        return Group(header(state), Panel(table, title=title, border_style="blue"), footer(state))

    state.current_list_is_groups = False

    if state.group_mode == "headers" and state.name_filter is None:
        display, job_positions = _flatten_grouped(all_jobs, lambda j: j)
        shown, offset, total_jobs, selected = _scroll_window_headlined(
            display, job_positions, state.scroll, height, state.row_delta, state.selected
        )
        state.scroll = offset
        state.selected = selected
        state.current_list_job_ids = [j.job_id for kind, j in display if kind == "row"]

        table = Table(expand=False)
        _queue_columns(table)
        for abs_idx, (kind, payload) in enumerate(shown, start=offset):
            if kind == "header":
                _add_queue_header_row(table, payload)
            else:
                _queue_job_row(table, payload, state, highlight=(abs_idx == job_positions[selected]))

        num_groups = sum(1 for kind, _ in display if kind == "header")
        title_base = f"{base} {_group_mode_title_tag(state)}"
        title = (
            f"{title_base} (none)"
            if total_jobs == 0
            else f"{title_base} ({num_groups} groups, {total_jobs} jobs -- {_MOVE_AND_ENTER_HINT})"
        )
        return Group(header(state), Panel(table, title=title, border_style="blue"), footer(state))

    if state.name_filter is not None:
        all_jobs = [j for j in all_jobs if _name_prefix(j.name) == state.name_filter]

    shown, offset, total, selected = _scroll_window_with_selection(
        all_jobs, state.scroll, height, state.row_delta, state.selected
    )
    state.scroll = offset
    state.selected = selected
    state.current_list_job_ids = [j.job_id for j in all_jobs]

    table = Table(expand=False)
    _queue_columns(table)
    for i, j in enumerate(shown, start=offset):
        _queue_job_row(table, j, state, highlight=(i == selected))

    title = _list_title(f"{base} {_group_mode_title_tag(state)}", offset, len(shown), total, hint=_MOVE_AND_ENTER_HINT)
    return Group(header(state), Panel(table, title=title, border_style="blue"), footer(state))


def render_detail(state: AppState, height: int | None = None) -> RenderableType:
    """Full-screen overlay for the 'scontrol show job <id>' output fetched
    on demand when Enter is pressed on the jobs/myjobs screens."""
    title = f"Job {state.detail_job_id}"
    if state.detail_loading:
        body: RenderableType = Text(f"loading scontrol show job {state.detail_job_id} ...", style="cyan")
        border = "cyan"
    elif state.detail_error:
        body = Text(state.detail_error, style="red")
        title += " -- error"
        border = "red"
    else:
        lines = (state.detail_text or "").splitlines() or ["(no output)"]
        shown, offset, total = _scroll_window(lines, state.detail_scroll, height)
        state.detail_scroll = offset
        body = Text("\n".join(shown))
        title = _list_title(title, offset, len(shown), total)
        border = "cyan"
    hint = Text("[↑/↓ PgUp/PgDn] scroll   [Enter/Esc/q] back", style="dim", justify="center")
    return Group(header(state), Panel(body, title=title, border_style=border), hint)


def render(state: AppState, height: int | None = None) -> RenderableType:
    if state.detail_job_id is not None:
        return render_detail(state, height)
    if state.screen == "nodes":
        return render_nodes(state, height)
    if state.screen == "users":
        return render_users(state, height)
    if state.screen == "jobs":
        return render_jobs(state, height)
    if state.screen == "myjobs":
        return render_myjobs(state, height)
    return render_overview(state)
