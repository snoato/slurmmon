"""Rich-based rendering: one glanceable overview screen plus keyboard-
switchable detail screens (nodes / users / jobs / my jobs), and an
on-demand job-detail overlay (Enter on a job row)."""
from __future__ import annotations

import time
from datetime import datetime

from rich.bar import Bar
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .aggregate import PartitionStats, UserUsage
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


def _gauge(pct: float, width: int = _BAR_WIDTH) -> RenderableType:
    color = _pct_color(pct)
    bar = Bar(size=1.0, begin=0, end=pct, width=width, color=color, bgcolor="grey23")
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


def header(state: AppState) -> RenderableType:
    location = "local" if state.local else state.host
    parts = [
        Text(" slurmmon ", style="bold white on blue"),
        Text(f" {location} "),
        Text(f"partitions={state.partitions_desc} ", style="dim"),
        Text(f"user={state.slurm_user} ", style="dim"),
    ]
    if state.fetching:
        parts.append(Text(" fetching... ", style="cyan"))
    if state.last_error:
        parts.append(Text(f" ⚠ {state.last_error} ", style="bold white on red"))
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
            bits.append("[Enter] info")
        if state.screen == "myjobs":
            bits.append("[f] filter")
        bits.append("[[/]] size")
    bits += ["[+/-] interval", "[r] refresh", "[q] quit"]
    return Text("  ".join(bits), style="dim", justify="center")


def _my_jobs_columns(table: Table) -> None:
    table.add_column("JobID", style="bold", no_wrap=True)
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
    }.get(job.state, "white")
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
        table.add_row(Text("no jobs on the selected partitions", style="dim"), "", "", "", "", "", "", "")
    shown = state.my_jobs[:_OVERVIEW_MY_JOBS_LIMIT]
    for row in shown:
        _my_job_row(table, row, state)

    remaining = len(state.my_jobs) - len(shown)
    title = f"My jobs ({state.slurm_user})"
    if remaining > 0:
        title += f"  (+{remaining} more, press 'm')"
    return Panel(table, title=title, border_style="blue")


def render_myjobs(state: AppState, height: int | None = None) -> RenderableType:
    items = state.my_jobs
    if state.job_filter != "ALL":
        items = [r for r in items if r.job.state == state.job_filter]

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

    base = f"My jobs ({state.slurm_user})"
    if state.job_filter != "ALL":
        base += f" [filter: {state.job_filter}, 'f' to cycle]"
    else:
        base += " ['f' to filter]"
    title = _list_title(base, offset, len(shown), total, hint=_MOVE_AND_ENTER_HINT)
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
    name_style = "bold cyan" if is_me else None
    row_style = " ".join(filter(None, [name_style, _SELECTED_ROW_STYLE if selected else None])) or None
    table.add_row(
        Text(u.user, style=name_style),
        str(u.job_count),
        Text.assemble(str(u.cpus), f" ({u.cpu_share * 100:.0f}%)"),
        Text.assemble(_fmt_gb(u.mem_mb), f" ({u.mem_share * 100:.0f}%)"),
        Text.assemble(f"{u.gpu_equiv:.1f}", f" ({u.gpu_share * 100:.0f}%)"),
        _gauge(u.dominant_share, width=14),
        style=row_style,
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


_SELECTED_ROW_STYLE = "on grey35"


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


def render_jobs(state: AppState, height: int | None = None) -> RenderableType:
    running = [j for j in state.jobs if j.is_running]
    pending = [j for j in state.jobs if j.is_pending]
    running.sort(key=lambda j: j.user)
    pending.sort(key=lambda j: j.job_id)
    all_jobs = running + pending

    shown, offset, total, selected = _scroll_window_with_selection(
        all_jobs, state.scroll, height, state.row_delta, state.selected
    )
    state.scroll = offset
    state.selected = selected
    state.current_list_job_ids = [j.job_id for j in all_jobs]

    table = Table(expand=False)
    table.add_column("JobID", style="bold", no_wrap=True)
    table.add_column("User", no_wrap=True)
    table.add_column("Partition", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Time / Reason", no_wrap=True)
    table.add_column("Nodes", ratio=1, min_width=8)

    for i, j in enumerate(shown, start=offset):
        state_style = {"RUNNING": "green", "PENDING": "yellow"}.get(j.state, "white")
        gpu_str = f"{j.gres.gpu}g" if j.gres.gpu else (f"{j.gres.shard}sh" if j.gres.shard else "")
        time_or_reason = j.time_used if j.is_running else j.reason
        table.add_row(
            j.job_id,
            Text(j.user, style="bold cyan" if j.user == state.slurm_user else None),
            j.partition,
            Text(j.state, style=state_style),
            str(j.cpus),
            _fmt_gb(j.mem_mb),
            gpu_str,
            time_or_reason,
            j.nodelist or "-",
            style=_SELECTED_ROW_STYLE if i == selected else None,
        )
    title = _list_title("Queue (running + pending)", offset, len(shown), total, hint=_MOVE_AND_ENTER_HINT)
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
