"""Turn parsed Slurm data into the numbers the UI renders.

Accounting model (deliberately consistent, see README for rationale):

- CPU capacity/usage comes straight from `sinfo`'s per-node alloc/idle/total
  columns -- this is an exact hard allocation, not an estimate.
- Memory and GPU usage are computed by summing *requested/allocated* TRES
  across running jobs (`squeue`), because Slurm does not expose real-time
  per-node memory/GPU consumption in this cluster's configuration (no
  `GresUsed` on `scontrol show node`). This means memory numbers reflect
  what jobs reserved, not necessarily their live RSS.
- Fractional GPU "shard" allocations and whole-GPU allocations are combined
  into one GPU-equivalent unit using a shards-per-GPU ratio derived live
  from each partition's node Gres string (never hardcoded).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parse import Job, Node, Snapshot


def _pct(used: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, used / total))


@dataclass
class PartitionStats:
    name: str

    cpus_total: int = 0
    cpus_unavail: int = 0
    cpus_used: int = 0

    mem_total_mb: float = 0.0
    mem_unavail_mb: float = 0.0
    mem_used_mb: float = 0.0

    gpu_total: int = 0
    gpu_unavail: int = 0
    shard_total: int = 0
    shard_unavail: int = 0
    gpu_used: float = 0.0  # GPU-equivalent (whole gpu + shard/shards_per_gpu)

    nodes_idle: int = 0
    nodes_busy: int = 0
    nodes_unavail: int = 0

    running_jobs: int = 0
    pending_jobs: int = 0

    @property
    def cpus_avail_total(self) -> int:
        return self.cpus_total - self.cpus_unavail

    @property
    def mem_avail_total_mb(self) -> float:
        return self.mem_total_mb - self.mem_unavail_mb

    @property
    def gpu_avail_total(self) -> float:
        """Whole-GPU-equivalent capacity that is actually up."""
        return self._gpu_equiv_avail()

    def _gpu_equiv_avail(self) -> float:
        # `gpu` and `shard` gres entries on a node describe the *same*
        # physical GPUs at two granularities (whole-unit vs. fractional
        # slice), not separate pools -- capacity is just the physical count.
        return float(self.gpu_total - self.gpu_unavail)

    @property
    def shards_per_gpu(self) -> float:
        gpu_avail = self.gpu_total - self.gpu_unavail
        shard_avail = self.shard_total - self.shard_unavail
        if gpu_avail > 0:
            return shard_avail / gpu_avail
        if self.gpu_total > 0:
            return self.shard_total / self.gpu_total
        return 0.0

    @property
    def cpu_pct(self) -> float:
        return _pct(self.cpus_used, self.cpus_avail_total)

    @property
    def mem_pct(self) -> float:
        return _pct(self.mem_used_mb, self.mem_avail_total_mb)

    @property
    def gpu_pct(self) -> float:
        return _pct(self.gpu_used, self._gpu_equiv_avail())


def _new_partition_stats_from_nodes(name: str, nodes: list[Node]) -> PartitionStats:
    ps = PartitionStats(name=name)
    for n in nodes:
        ps.cpus_total += n.cpus_total
        ps.mem_total_mb += n.mem_mb
        ps.gpu_total += n.gres.gpu
        ps.shard_total += n.gres.shard
        if n.bucket == "unavailable":
            ps.nodes_unavail += 1
            ps.cpus_unavail += n.cpus_total
            ps.mem_unavail_mb += n.mem_mb
            ps.gpu_unavail += n.gres.gpu
            ps.shard_unavail += n.gres.shard
        else:
            ps.cpus_used += n.cpus_alloc
            if n.bucket == "idle":
                ps.nodes_idle += 1
            else:
                ps.nodes_busy += 1
    return ps


def _job_gpu_equiv(job: Job, shards_per_gpu: float) -> float:
    equiv = float(job.gres.gpu)
    if shards_per_gpu > 0:
        equiv += job.gres.shard / shards_per_gpu
    return equiv


def build_partition_stats(nodes: list[Node], jobs: list[Job]) -> dict[str, PartitionStats]:
    by_partition: dict[str, list[Node]] = {}
    for n in nodes:
        by_partition.setdefault(n.partition, []).append(n)

    stats = {name: _new_partition_stats_from_nodes(name, ns) for name, ns in by_partition.items()}

    for job in jobs:
        ps = stats.get(job.partition)
        if ps is None:
            continue
        if job.is_running:
            ps.running_jobs += 1
            ps.mem_used_mb += job.mem_mb
            ps.gpu_used += _job_gpu_equiv(job, ps.shards_per_gpu)
        elif job.is_pending:
            ps.pending_jobs += 1

    return stats


def build_cluster_stats(nodes: list[Node], jobs: list[Job]) -> PartitionStats:
    """Cluster-wide (all given partitions combined) rollup, de-duplicating
    any node that happens to be listed under more than one partition."""
    unique_nodes: dict[str, Node] = {}
    for n in nodes:
        unique_nodes.setdefault(n.name, n)
    cluster = _new_partition_stats_from_nodes("cluster", list(unique_nodes.values()))
    spg = cluster.shards_per_gpu
    for job in jobs:
        if job.is_running:
            cluster.running_jobs += 1
            cluster.mem_used_mb += job.mem_mb
            cluster.gpu_used += _job_gpu_equiv(job, spg)
        elif job.is_pending:
            cluster.pending_jobs += 1
    return cluster


@dataclass
class UserUsage:
    user: str
    job_count: int = 0
    cpus: int = 0
    mem_mb: float = 0.0
    gpu_equiv: float = 0.0
    cpu_share: float = 0.0
    mem_share: float = 0.0
    gpu_share: float = 0.0

    @property
    def dominant_share(self) -> float:
        return max(self.cpu_share, self.mem_share, self.gpu_share)

    @property
    def dominant_resource(self) -> str:
        shares = {"cpu": self.cpu_share, "mem": self.mem_share, "gpu": self.gpu_share}
        return max(shares, key=shares.get)


def build_user_usage(jobs: list[Job], cluster: PartitionStats) -> list[UserUsage]:
    by_user: dict[str, UserUsage] = {}
    spg = cluster.shards_per_gpu
    for job in jobs:
        if not job.is_running:
            continue
        u = by_user.setdefault(job.user, UserUsage(user=job.user))
        u.job_count += 1
        u.cpus += job.cpus
        u.mem_mb += job.mem_mb
        u.gpu_equiv += _job_gpu_equiv(job, spg)

    cpu_total = cluster.cpus_avail_total
    mem_total = cluster.mem_avail_total_mb
    gpu_total = cluster._gpu_equiv_avail()
    for u in by_user.values():
        u.cpu_share = _pct(u.cpus, cpu_total)
        u.mem_share = _pct(u.mem_mb, mem_total)
        u.gpu_share = _pct(u.gpu_equiv, gpu_total)

    return sorted(by_user.values(), key=lambda u: u.dominant_share, reverse=True)


@dataclass
class MyJobRow:
    job: Job
    prio: object = None  # PrioRow | None, kept loosely typed to avoid import cycle noise
    fairshare: float | None = None


def build_my_jobs(jobs: list[Job], slurm_user: str, prio_rows: list, share_rows: list) -> list[MyJobRow]:
    prio_by_id = {p.job_id: p for p in prio_rows}

    def _find_prio(job_id: str):
        if job_id in prio_by_id:
            return prio_by_id[job_id]
        base = job_id.split("_", 1)[0]
        return prio_by_id.get(base)

    fairshare = None
    for row in share_rows:
        if row.user == slurm_user and row.fairshare is not None:
            fairshare = row.fairshare
            break

    mine = [j for j in jobs if j.user == slurm_user]
    mine.sort(key=lambda j: (j.state != "RUNNING", j.job_id))
    return [MyJobRow(job=j, prio=_find_prio(j.job_id), fairshare=fairshare) for j in mine]
