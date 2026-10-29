from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import os
import sys
import threading
import time
from collections import deque

from rich.live import Live

from . import parse, ssh_client
from .aggregate import build_cluster_stats, build_my_jobs, build_partition_stats, build_user_usage
from .keys import OSC_PREFIX, raw_terminal, read_key, read_keys
from .ssh_client import PartitionSelector
from .state import DEFAULT_GROUP_MODE, AppState
from .theme import BackgroundWatcher, make_console
from .ui import GROUP_MODES, _JOB_FILTERS, render

MIN_INTERVAL = 2.0
MAX_INTERVAL = 300.0
FRAME_WAIT = 0.15  # longest to wait for input before redrawing anyway ("updated Xs ago" etc.)
PAGE_STEP = 15
ROW_DELTA_STEP = 2
MIN_ROW_DELTA = -30
ENTER_KEYS = ("\r", "\n")
# On quit, how long to wait for an in-flight terminal background-color reply
# before restoring the terminal -- arriving any later, it would land in the
# user's shell prompt as literal "^[]11;rgb:..." junk.
EXIT_REPLY_WAIT = 0.5

DEFAULT_PARTITION_PREFIX = ""  # empty prefix matches every partition
DEFAULT_INTERVAL = 30.0


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SLURMMON_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"SLURMMON_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def compute_refresh(host: str | None, slurm_user: str, selector: PartitionSelector, local: bool) -> dict:
    """Fetch + parse + aggregate one snapshot. Returns a dict of AppState
    field updates, or raises ssh_client.SlurmFetchError."""
    if local:
        raw = ssh_client.fetch_raw_local(slurm_user, selector)
    else:
        assert host is not None
        raw = ssh_client.fetch_raw_remote(host, slurm_user, selector)

    nodes = []
    for line in raw.sinfo.splitlines():
        n = parse.parse_sinfo_line(line)
        if n is not None:
            nodes.append(n)

    jobs = []
    for line in raw.squeue.splitlines():
        j = parse.parse_squeue_line(line)
        if j is not None:
            jobs.append(j)

    prio = []
    for line in raw.sprio.splitlines():
        p = parse.parse_sprio_line(line)
        if p is not None:
            prio.append(p)

    shares = []
    for line in raw.sshare.splitlines():
        s = parse.parse_sshare_line(line)
        if s is not None:
            shares.append(s)

    cluster = build_cluster_stats(nodes, jobs)
    return {
        "partitions": sorted({n.partition for n in nodes}),
        "nodes": nodes,
        "jobs": jobs,
        "partition_stats": build_partition_stats(nodes, jobs),
        "cluster_stats": cluster,
        "user_usage": build_user_usage(jobs, cluster),
        "my_jobs": build_my_jobs(jobs, slurm_user, prio, shares),
        "server_now": raw.now.strip() or None,
        "last_success": time.time(),
        "last_error": None,
    }


def _partition_dict(ps) -> dict:
    return {
        "cpus_total": ps.cpus_total,
        "cpus_unavail": ps.cpus_unavail,
        "cpus_used": ps.cpus_used,
        "cpus_avail_total": ps.cpus_avail_total,
        "cpu_pct": round(ps.cpu_pct, 4),
        "mem_total_mb": ps.mem_total_mb,
        "mem_used_mb": round(ps.mem_used_mb, 1),
        "mem_avail_total_mb": ps.mem_avail_total_mb,
        "mem_pct": round(ps.mem_pct, 4),
        "gpu_total": ps.gpu_total,
        "shard_total": ps.shard_total,
        "shards_per_gpu": round(ps.shards_per_gpu, 2),
        "gpu_avail_total": round(ps.gpu_avail_total, 2),
        "gpu_used": round(ps.gpu_used, 2),
        "gpu_pct": round(ps.gpu_pct, 4),
        "nodes_idle": ps.nodes_idle,
        "nodes_busy": ps.nodes_busy,
        "nodes_unavail": ps.nodes_unavail,
        "running_jobs": ps.running_jobs,
        "pending_jobs": ps.pending_jobs,
    }


def snapshot_to_dict(state: AppState) -> dict:
    return {
        "host": "local" if state.local else state.host,
        "slurm_user": state.slurm_user,
        "partition_selection": state.partitions_desc,
        "server_now": state.server_now,
        "last_success": state.last_success,
        "last_error": state.last_error,
        "partitions": {name: _partition_dict(ps) for name, ps in state.partition_stats.items()},
        "cluster": _partition_dict(state.cluster_stats) if state.cluster_stats else None,
        "users": [
            {
                "user": u.user,
                "job_count": u.job_count,
                "cpus": u.cpus,
                "mem_mb": u.mem_mb,
                "gpu_equiv": round(u.gpu_equiv, 2),
                "cpu_share": round(u.cpu_share, 4),
                "mem_share": round(u.mem_share, 4),
                "gpu_share": round(u.gpu_share, 4),
                "dominant_share": round(u.dominant_share, 4),
                "dominant_resource": u.dominant_resource,
            }
            for u in state.user_usage
        ],
        "my_jobs": [
            {
                "job_id": r.job.job_id,
                "partition": r.job.partition,
                "state": r.job.state,
                "cpus": r.job.cpus,
                "mem_mb": r.job.mem_mb,
                "gpu": r.job.gres.gpu,
                "shard": r.job.gres.shard,
                "time_used": r.job.time_used,
                "reason": r.job.reason,
                "start_or_eta": r.job.start_or_eta,
                "submit_time": r.job.submit_time,
                "fairshare": r.fairshare,
            }
            for r in state.my_jobs
        ],
    }


def _fetch_job_detail_async(state: AppState, job_id: str, lock: threading.Lock) -> None:
    """Kick off an on-demand `scontrol show job <id>` fetch (Enter on a job
    row) in a short-lived background thread, so it never blocks the render
    loop. Rides the same ControlMaster connection as the regular poll."""

    def worker() -> None:
        try:
            text = ssh_client.fetch_job_detail(state.host, job_id, state.local)
        except ssh_client.SlurmFetchError as exc:
            with lock:
                if state.detail_job_id == job_id:
                    state.detail_loading = False
                    state.detail_error = str(exc)
            return
        with lock:
            if state.detail_job_id == job_id:
                state.detail_loading = False
                state.detail_text = text

    with lock:
        state.detail_job_id = job_id
        state.detail_loading = True
        state.detail_text = None
        state.detail_error = None
        state.detail_scroll = 0
    threading.Thread(target=worker, daemon=True).start()


_MOVES = {"UP": -1, "DOWN": 1, "PGUP": -PAGE_STEP, "PGDN": PAGE_STEP}


def _move(state: AppState, delta: int) -> None:
    """Move the cursor (or scroll the job-detail overlay), clamped against
    what the last frame showed. Rendering clamps again anyway, but a batch
    of keys is handled before the next one (see _apply_keys): without this,
    DOWN x10 then UP x3 at the bottom of a list would net out to 7 rows
    *past* the end, clamped back to the last row -- not 3 rows above it."""
    if state.detail_job_id is not None:
        state.detail_scroll = max(0, min(state.detail_scroll + delta, state.detail_max_scroll))
    else:
        state.selected = max(0, min(state.selected + delta, state.list_length - 1))


def _shown_list(state: AppState) -> tuple:
    """Everything that decides which list the cursor is moving through.
    When a key changes it, the new list has to be rendered before the next
    key can be handled: list_length and current_list_job_ids (what moving
    and Enter act on) only catch up on the next render."""
    return (state.screen, state.job_filter, state.group_mode, state.name_filter, state.detail_job_id)


def _handle_key(
    state: AppState, key: str, *, lock: threading.Lock, refresh_now: threading.Event, stop_event: threading.Event
) -> None:
    if state.detail_job_id is not None:
        # Job-detail overlay is open: it owns the keyboard until closed,
        # scrolling its own text instead of the underlying screen's selection.
        if key in ("q", "ESC", *ENTER_KEYS):
            state.detail_job_id = None
        elif key in _MOVES:
            _move(state, _MOVES[key])
        return

    if key == "q":
        stop_event.set()
    elif key in ("o", "n", "u", "j", "m"):
        state.screen = {"o": "overview", "n": "nodes", "u": "users", "j": "jobs", "m": "myjobs"}[key]
        state.scroll = 0
        state.selected = 0
        state.group_mode = DEFAULT_GROUP_MODE
        state.name_filter = None
    elif key in _MOVES:
        _move(state, _MOVES[key])
    elif key == "f" and state.screen == "myjobs":
        idx = (_JOB_FILTERS.index(state.job_filter) + 1) % len(_JOB_FILTERS)
        state.job_filter = _JOB_FILTERS[idx]
        state.name_filter = None
        state.scroll = 0
        state.selected = 0
    elif key == "g" and state.screen in ("jobs", "myjobs"):
        if state.name_filter is not None:
            state.name_filter = None  # drilled -> back to the "summary" overview
        else:
            idx = (GROUP_MODES.index(state.group_mode) + 1) % len(GROUP_MODES)
            state.group_mode = GROUP_MODES[idx]
        state.scroll = 0
        state.selected = 0
    elif key in ENTER_KEYS and state.screen in ("jobs", "myjobs"):
        if state.current_list_job_ids and 0 <= state.selected < len(state.current_list_job_ids):
            target = state.current_list_job_ids[state.selected]
            if state.current_list_is_groups:
                state.name_filter = target
                state.scroll = 0
                state.selected = 0
            else:
                _fetch_job_detail_async(state, target, lock)
    elif key == "[":
        state.row_delta = max(MIN_ROW_DELTA, state.row_delta - ROW_DELTA_STEP)
    elif key == "]":
        state.row_delta = min(0, state.row_delta + ROW_DELTA_STEP)
    elif key == "r":
        refresh_now.set()
    elif key == "+":
        state.interval = min(state.interval + 1, MAX_INTERVAL)
    elif key == "-":
        state.interval = max(state.interval - 1, MIN_INTERVAL)


def _apply_keys(
    state: AppState,
    pending: deque[str],
    *,
    lock: threading.Lock,
    refresh_now: threading.Event,
    stop_event: threading.Event,
) -> None:
    """Handle the keys in `pending`, oldest first, until it's empty -- or
    until one of them switches to a different list (screen, filter,
    grouping, drill-down, detail overlay), leaving the rest queued for after
    that list has been rendered. Keys that only move within the current list
    (the ones that arrive in floods) are all applied without a redraw in
    between."""
    while pending and not stop_event.is_set():
        before = _shown_list(state)
        _handle_key(state, pending.popleft(), lock=lock, refresh_now=refresh_now, stop_event=stop_event)
        if _shown_list(state) != before:
            return


@contextlib.contextmanager
def _awaiting_background_reply_on_exit(watcher: BackgroundWatcher):
    try:
        yield
    finally:
        deadline = time.monotonic() + EXIT_REPLY_WAIT
        while watcher.awaiting_reply and (left := deadline - time.monotonic()) > 0:
            key = read_key(left)
            if key and key.startswith(OSC_PREFIX):
                watcher.handle_reply(key[len(OSC_PREFIX):])


def run_interactive(state: AppState) -> None:
    console = make_console()
    watcher = BackgroundWatcher(console)
    stop_event = threading.Event()
    refresh_now = threading.Event()
    lock = threading.Lock()

    def fetcher() -> None:
        while not stop_event.is_set():
            with lock:
                state.fetching = True
            try:
                updates = compute_refresh(state.host, state.slurm_user, state.selector, state.local)
            except ssh_client.SlurmFetchError as exc:
                updates = {"last_error": str(exc)}
            with lock:
                state.fetching = False
                for key, value in updates.items():
                    setattr(state, key, value)
            refresh_now.wait(timeout=state.interval)
            refresh_now.clear()

    thread = threading.Thread(target=fetcher, daemon=True)
    thread.start()

    try:
        with (
            raw_terminal(),
            Live(console=console, screen=True, auto_refresh=False) as live,
            _awaiting_background_reply_on_exit(watcher),  # last, so it exits first: still in cbreak mode
        ):
            pending: deque[str] = deque()
            while not stop_event.is_set():
                watcher.tick()
                size = console.size
                with lock:
                    live.update(render(state, height=size.height, width=size.width), refresh=True)
                # Everything typed since the last frame, not just one key (see
                # read_keys) -- without waiting if some is still queued up
                # from before that frame.
                for key in read_keys(0 if pending else FRAME_WAIT):
                    if key.startswith(OSC_PREFIX):
                        watcher.handle_reply(key[len(OSC_PREFIX):])
                    else:
                        pending.append(key)
                _apply_keys(state, pending, lock=lock, refresh_now=refresh_now, stop_event=stop_event)
    finally:
        stop_event.set()
        refresh_now.set()
        thread.join(timeout=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slurmmon",
        description="htop-like terminal monitor for Slurm partitions, jobs, and per-user usage.",
    )
    p.add_argument(
        "--local", action="store_true", default=_env_bool("LOCAL", False),
        help=(
            "run sinfo/squeue/sprio/sshare directly on this machine instead of "
            "over SSH -- use this when Slurm client tools are already on PATH "
            "here (e.g. running slurmmon on the cluster's own login node). "
            "Makes --host irrelevant. (env: SLURMMON_LOCAL=1)"
        ),
    )
    p.add_argument(
        "--host", default=_env("HOST", ""),
        help="SSH host with Slurm client tools on PATH (required unless --local; or set env SLURMMON_HOST)",
    )
    p.add_argument(
        "--slurm-user", default=_env("USER", ""),
        help=(
            "Slurm username to track (required; or set env SLURMMON_USER). "
            "With --local, defaults to your local username."
        ),
    )
    p.add_argument(
        "--partitions", default=_env("PARTITIONS", ""),
        help=(
            "comma-separated exact partition names, or 'auto' to use whatever "
            "partitions your Slurm account is allowed to submit to. Overrides "
            "--partition-prefix when set. (env: SLURMMON_PARTITIONS)"
        ),
    )
    p.add_argument(
        "--partition-prefix", default=_env("PARTITION_PREFIX", DEFAULT_PARTITION_PREFIX),
        help=(
            "only consider partitions starting with this prefix; ignored if "
            "--partitions is set (default: empty, i.e. all partitions; "
            "env: SLURMMON_PARTITION_PREFIX)"
        ),
    )
    p.add_argument(
        "--interval", type=float, default=float(_env("INTERVAL", str(DEFAULT_INTERVAL))),
        help=f"refresh interval in seconds (default: {DEFAULT_INTERVAL:g}, env: SLURMMON_INTERVAL)",
    )
    p.add_argument("--once", action="store_true", help="fetch one snapshot, print one frame, and exit")
    p.add_argument("--json", action="store_true", help="dump the aggregated snapshot as JSON instead of rendering")
    return p


def _build_selector(args: argparse.Namespace) -> PartitionSelector:
    partitions = args.partitions.strip()
    if not partitions:
        return PartitionSelector(mode="prefix", arg=args.partition_prefix)
    if partitions.lower() == "auto":
        return PartitionSelector(mode="auto")
    return PartitionSelector(mode="list", arg=partitions)


def resolve_config(args: argparse.Namespace) -> tuple[str | None, str, bool]:
    """Validate host/slurm-user/--local together and fill in the one sane
    implicit default (local username, only in --local mode). Returns
    (host_or_None, slurm_user, local); raises ValueError on invalid input."""
    local = bool(args.local)

    host = args.host or None
    if not local and not host:
        raise ValueError("--host is required unless --local is set (or set SLURMMON_HOST)")

    slurm_user = args.slurm_user
    if not slurm_user:
        if local:
            slurm_user = getpass.getuser()
        else:
            raise ValueError("--slurm-user is required (or set SLURMMON_USER)")

    return (None if local else host, slurm_user, local)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        host, slurm_user, local = resolve_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # unreachable, parser.error() exits

    state = AppState(
        host=host,
        slurm_user=slurm_user,
        local=local,
        selector=_build_selector(args),
        interval=args.interval,
    )

    if args.once or args.json:
        try:
            for key, value in compute_refresh(state.host, state.slurm_user, state.selector, state.local).items():
                setattr(state, key, value)
        except ssh_client.SlurmFetchError as exc:
            print(f"slurmmon: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 130
        if args.json:
            print(json.dumps(snapshot_to_dict(state), indent=2))
        else:
            make_console().print(render(state))
        return 0

    try:
        run_interactive(state)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
