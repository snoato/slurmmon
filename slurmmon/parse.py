"""Parsers for raw Slurm CLI output into plain dataclasses.

All parsing here is defensive: a malformed/short line is skipped rather than
raising, since a live cluster occasionally emits a job row mid-transition
(e.g. a job completing between our sinfo and squeue calls) and the tool
should keep rendering rather than crash on one bad line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# sinfo/scontrol node-state trailing flag characters (see `man sinfo`, NODE STATE CODES).
# Notably '-' means "planned by the backfill scheduler for a higher-priority
# job" -- the node is still up and running the current job, just earmarked.
_NODE_STATE_FLAGS = "*~#!%$@^-"

# Node states that mean "actually schedulable right now".
_IDLE_STATES = {"idle"}
_BUSY_STATES = {"mixed", "allocated", "completing"}
# Everything else (down, drain, draining, fail, failing, unknown, reserved,
# maint, future, planned, ...) is treated as unavailable capacity.


def classify_node_state(raw: str) -> tuple[str, str]:
    """Return (base_state_lower, bucket) with bucket in {idle, busy, unavailable}."""
    base = raw.strip().rstrip(_NODE_STATE_FLAGS)
    base = base.split("+", 1)[0]  # compound states like MIXED+PLANNED
    base_lower = base.lower()
    if base_lower in _IDLE_STATES:
        bucket = "idle"
    elif base_lower in _BUSY_STATES:
        bucket = "busy"
    else:
        bucket = "unavailable"
    return base_lower, bucket


@dataclass(frozen=True)
class GresCount:
    gpu: int = 0
    shard: int = 0
    gpu_model: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.gpu == 0 and self.shard == 0


def parse_gres(raw: str) -> GresCount:
    """Parse a node Gres string ('gpu:A10:4,shard:A10:96') or a job
    TresPerNode string ('gres/gpu:1', 'gres/shard:8'). Both shapes share
    the same '<kind>[:<model>]:<count>' grammar once the 'gres/' prefix
    used on the job side is stripped.
    """
    if not raw:
        return GresCount()
    raw = raw.strip()
    if raw in ("(null)", "N/A", "n/a", ""):
        return GresCount()

    gpu = shard = 0
    model: str | None = None
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        segs = part.split(":")
        kind = segs[0].removeprefix("gres/")
        try:
            count = int(segs[-1])
        except ValueError:
            continue
        if kind == "gpu":
            gpu += count
            if len(segs) == 3 and model is None:
                model = segs[1]
        elif kind == "shard":
            shard += count
            if len(segs) == 3 and model is None:
                model = segs[1]
    return GresCount(gpu=gpu, shard=shard, gpu_model=model)


def parse_mem_mb(raw: str) -> float:
    """Parse a memory value into MB.

    Bare digits (as emitted by `sinfo -o %m`, e.g. node RealMemory) are
    already MB. Suffixed values (as emitted by `squeue -o %m`, e.g. '64G',
    '128G') carry a K/M/G/T unit that needs converting.
    """
    raw = raw.strip()
    if not raw or raw.upper() in ("N/A", "0"):
        return 0.0
    m = re.match(r"^([\d.]+)\s*([KMGTkmgt]?)", raw)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = {"": 1.0, "K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}[unit]
    return val * mult


@dataclass
class Node:
    name: str
    partition: str
    raw_state: str
    state: str
    bucket: str  # idle / busy / unavailable
    cpus_alloc: int
    cpus_idle: int
    cpus_other: int
    cpus_total: int
    mem_mb: float
    gres: GresCount


def parse_sinfo_line(line: str) -> Node | None:
    """Parse one line of `sinfo -h -N -o '%N|%P|%T|%C|%m|%G'`."""
    parts = line.rstrip("\n").split("|")
    if len(parts) != 6:
        return None
    name, partition, raw_state, cpus_field, mem_field, gres_field = parts
    partition = partition.rstrip("*").strip()
    cpu_parts = cpus_field.split("/")
    if len(cpu_parts) != 4:
        return None
    try:
        alloc, idle, other, total = (int(x) for x in cpu_parts)
    except ValueError:
        return None
    state, bucket = classify_node_state(raw_state)
    return Node(
        name=name.strip(),
        partition=partition,
        raw_state=raw_state.strip(),
        state=state,
        bucket=bucket,
        cpus_alloc=alloc,
        cpus_idle=idle,
        cpus_other=other,
        cpus_total=total,
        mem_mb=parse_mem_mb(mem_field),
        gres=parse_gres(gres_field),
    )


@dataclass
class Job:
    job_id: str
    user: str
    partition: str
    state: str
    num_nodes: int
    cpus: int
    mem_mb: float
    gres: GresCount
    time_used: str
    reason: str
    start_or_eta: str
    nodelist: str
    submit_time: str
    name: str

    @property
    def is_running(self) -> bool:
        return self.state == "RUNNING"

    @property
    def is_pending(self) -> bool:
        return self.state == "PENDING"


def parse_squeue_line(line: str) -> Job | None:
    """Parse one line of
    `squeue -h -o '%i|%u|%P|%T|%D|%C|%m|%b|%M|%r|%S|%N|%V|%j'`.

    Job name (%j) is last since it's the one free-form field here -- Slurm
    doesn't allow '|' in job names, but keeping it last means a stray odd
    character in a name can never shift the fixed-shape fields before it.
    """
    parts = line.rstrip("\n").split("|")
    if len(parts) != 14:
        return None
    job_id, user, partition, state, num_nodes, cpus, mem, tres, time_used, reason, start_eta, nodelist, submit_time, name = parts
    try:
        num_nodes_i = int(num_nodes)
    except ValueError:
        num_nodes_i = 0
    try:
        cpus_i = int(cpus)
    except ValueError:
        cpus_i = 0
    return Job(
        job_id=job_id.strip(),
        user=user.strip(),
        partition=partition.strip(),
        state=state.strip().upper(),
        num_nodes=num_nodes_i,
        cpus=cpus_i,
        mem_mb=parse_mem_mb(mem),
        gres=parse_gres(tres),
        time_used=time_used.strip(),
        reason=reason.strip(),
        start_or_eta=start_eta.strip(),
        nodelist=nodelist.strip(),
        name=name.strip() or "(unnamed)",
        submit_time=submit_time.strip(),
    )


@dataclass
class PrioRow:
    job_id: str
    priority: int
    site: int
    age: int
    fairshare: int
    qos: int


def parse_sprio_line(line: str) -> PrioRow | None:
    """Parse one line of `sprio -h -u <user>` (columns:
    JOBID PARTITION USER PRIORITY SITE AGE FAIRSHARE QOS).
    """
    tokens = line.split()
    if len(tokens) != 8:
        return None
    job_id = tokens[0]
    try:
        priority, site, age, fairshare, qos = (int(x) for x in tokens[3:8])
    except ValueError:
        return None
    return PrioRow(job_id=job_id, priority=priority, site=site, age=age, fairshare=fairshare, qos=qos)


@dataclass
class ShareRow:
    account: str
    user: str
    raw_shares: str
    raw_usage: int
    fairshare: float | None
    norm_usage: float | None


def parse_sshare_line(line: str) -> ShareRow | None:
    """Parse one line of
    `sshare -n -P -u <user> -o Account,User,RawShares,RawUsage,FairShare,NormUsage`.
    """
    parts = line.rstrip("\n").split("|")
    if len(parts) < 5:
        return None
    account, user, raw_shares, raw_usage, fairshare = parts[:5]
    norm_usage = parts[5] if len(parts) > 5 else ""
    try:
        raw_usage_i = int(raw_usage.strip())
    except ValueError:
        raw_usage_i = 0

    def _opt_float(s: str) -> float | None:
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    return ShareRow(
        account=account.strip(),
        user=user.strip(),
        raw_shares=raw_shares.strip(),
        raw_usage=raw_usage_i,
        fairshare=_opt_float(fairshare),
        norm_usage=_opt_float(norm_usage),
    )


@dataclass
class Snapshot:
    """Everything parsed from one bundled remote fetch."""

    nodes: list[Node] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    prio: list[PrioRow] = field(default_factory=list)
    shares: list[ShareRow] = field(default_factory=list)
