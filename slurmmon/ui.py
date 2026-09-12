"""Rich-based rendering: one glanceable overview screen plus three
keyboard-switchable detail screens (nodes / users / jobs)."""
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
    scroll_hint = "  [↑/↓ PgUp/PgDn] scroll  [[/]] page size" if state.screen != "overview" else ""
    hints = f"[o] overview  [n] nodes  [u] users  [j] jobs  [m] my jobs{scroll_hint}  [+/-] interval  [r] refresh  [q] quit"
    return Text(hints, style="dim", justify="center")


def _my_jobs_columns(table: Table) -> None:
    table.add_column("JobID", style="bold", no_wrap=True)
    table.add_column("Partition", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Time", no_wrap=True)
    table.add_column("Reason / Priority", ratio=1, min_width=16)


def _my_job_row(table: Table, row, state: AppState) -> None:
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
    )


_OVERVIEW_MY_JOBS_LIMIT = 8


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
    shown, offset, total = _scroll_window(state.my_jobs, state.scroll, height, state.row_delta)

    table = Table(expand=False, box=None, pad_edge=False)
    _my_jobs_columns(table)
    for row in shown:
        _my_job_row(table, row, state)

    title = _list_title(f"My jobs ({state.slurm_user})", offset, len(shown), total)
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


def _user_row(table: Table, u: UserUsage, *, highlight: bool) -> None:
    style = "bold cyan" if highlight else None
    table.add_row(
        Text(u.user, style=style),
        str(u.job_count),
        Text.assemble(str(u.cpus), f" ({u.cpu_share * 100:.0f}%)"),
        Text.assemble(_fmt_gb(u.mem_mb), f" ({u.mem_share * 100:.0f}%)"),
        Text.assemble(f"{u.gpu_equiv:.1f}", f" ({u.gpu_share * 100:.0f}%)"),
        _gauge(u.dominant_share, width=14),
        style=style,
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
        _user_row(table, u, highlight=(u.user == state.slurm_user))

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


def _list_title(base: str, offset: int, shown: int, total: int) -> str:
    if total == 0:
        return f"{base} (none)"
    if shown >= total:
        return f"{base} ({total})"
    return f"{base} ({offset + 1}-{offset + shown} of {total} -- ↑/↓ PgUp/PgDn to scroll)"


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
    shown, offset, total = _scroll_window(all_nodes, state.scroll, height, state.row_delta)

    table = Table(expand=False)
    table.add_column("Node", style="bold", no_wrap=True)
    table.add_column("Partition", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("CPU (alloc/total)", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("Gres", ratio=1, min_width=10)
    for n in shown:
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
        )
    title = _list_title("Nodes", offset, len(shown), total)
    return Group(header(state), Panel(table, title=title, border_style="magenta"), footer(state))


def render_users(state: AppState, height: int | None = None) -> RenderableType:
    shown, offset, total = _scroll_window(state.user_usage, state.scroll, height, state.row_delta)

    table = Table(expand=False)
    table.add_column("User", style="bold", ratio=1, min_width=12)
    table.add_column("Jobs", justify="right", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Mem", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Dominant share", no_wrap=True)
    for u in shown:
        _user_row(table, u, highlight=(u.user == state.slurm_user))
    title = _list_title("All users", offset, len(shown), total)
    return Group(header(state), Panel(table, title=title, border_style="green"), footer(state))


def render_jobs(state: AppState, height: int | None = None) -> RenderableType:
    running = [j for j in state.jobs if j.is_running]
    pending = [j for j in state.jobs if j.is_pending]
    running.sort(key=lambda j: j.user)
    pending.sort(key=lambda j: j.job_id)
    all_jobs = running + pending
    shown, offset, total = _scroll_window(all_jobs, state.scroll, height, state.row_delta)

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

    for j in shown:
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
        )
    title = _list_title("Queue (running + pending)", offset, len(shown), total)
    return Group(header(state), Panel(table, title=title, border_style="blue"), footer(state))


def render(state: AppState, height: int | None = None) -> RenderableType:
    if state.screen == "nodes":
        return render_nodes(state, height)
    if state.screen == "users":
        return render_users(state, height)
    if state.screen == "jobs":
        return render_jobs(state, height)
    if state.screen == "myjobs":
        return render_myjobs(state, height)
    return render_overview(state)
