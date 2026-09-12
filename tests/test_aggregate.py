from pathlib import Path

from slurmmon.aggregate import (
    build_cluster_stats,
    build_my_jobs,
    build_partition_stats,
    build_user_usage,
)
from slurmmon.parse import (
    parse_sinfo_line,
    parse_sprio_line,
    parse_sshare_line,
    parse_squeue_line,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text().splitlines()


def _load():
    nodes = [n for line in _lines("sinfo.txt") if (n := parse_sinfo_line(line))]
    jobs = [j for line in _lines("squeue.txt") if (j := parse_squeue_line(line))]
    prio = [p for line in _lines("sprio.txt") if (p := parse_sprio_line(line))]
    shares = [s for line in _lines("sshare.txt") if (s := parse_sshare_line(line))]
    return nodes, jobs, prio, shares


def test_partition_stats_excludes_down_nodes_from_capacity():
    nodes, jobs, _, _ = _load()
    stats = build_partition_stats(nodes, jobs)
    cpu_a = stats["cpu_a"]

    # cn-102 and cn-116 are down (96 cpus each) -> excluded from avail total.
    assert cpu_a.cpus_total == 96 * 4
    assert cpu_a.cpus_unavail == 96 * 2
    assert cpu_a.cpus_avail_total == 96 * 2
    # cn-101 alloc=10, cn-115 alloc=8 -> 18 used out of the 192 available.
    assert cpu_a.cpus_used == 18
    assert cpu_a.nodes_unavail == 2
    assert cpu_a.nodes_busy == 2


def test_gpu_shard_normalization_per_partition():
    nodes, jobs, _, _ = _load()
    stats = build_partition_stats(nodes, jobs)
    gpu_a = stats["gpu_a"]

    # 4 nodes total, cn-407 down -> 3 up nodes, each 4 gpu / 96 shard -> 24 shards/gpu.
    assert gpu_a.shards_per_gpu == 24
    # up nodes: cn-401, cn-405, cn-414 -> 12 gpu avail, 288 shard avail == 12 gpu-equiv.
    assert gpu_a.gpu_avail_total == 12

    # running jobs on gpu_a: shard:8 (carol) + shard:4 (dave) + shard:12 (bob)
    # = 24 shards -> 24/24 = 1.0 gpu-equivalent used.
    assert gpu_a.gpu_used == 1.0


def test_pending_jobs_do_not_count_as_used():
    nodes, jobs, _, _ = _load()
    stats = build_partition_stats(nodes, jobs)
    gpu_b = stats["gpu_b"]
    # alice's pending gres/gpu:1 request on gpu_b must not be counted as
    # allocated -- only cn-418's running whole-node allocation should show up
    # as CPU usage; gpu_used should be 0 since no running job on that
    # partition requests gpu/shard in the fixture.
    assert gpu_b.pending_jobs == 1
    assert gpu_b.gpu_used == 0.0


def test_cluster_stats_dedupes_and_sums_partitions():
    nodes, jobs, _, _ = _load()
    cluster = build_cluster_stats(nodes, jobs)
    per_partition = build_partition_stats(nodes, jobs)

    assert cluster.cpus_total == sum(p.cpus_total for p in per_partition.values())
    assert cluster.running_jobs == sum(p.running_jobs for p in per_partition.values())
    assert cluster.pending_jobs == sum(p.pending_jobs for p in per_partition.values())


def test_user_usage_ranks_by_dominant_share():
    nodes, jobs, _, _ = _load()
    cluster = build_cluster_stats(nodes, jobs)
    usage = build_user_usage(jobs, cluster)

    users = {u.user: u for u in usage}
    # bob runs jobs on both cpu_b (64 cpu) and gpu_a (10 cpu + 12 shard) ->
    # should be present with combined cpu.
    assert users["bob.example"].cpus == 64 + 10
    assert users["bob.example"].job_count == 2

    # ranking is sorted descending by dominant_share.
    shares = [u.dominant_share for u in usage]
    assert shares == sorted(shares, reverse=True)

    # a user with only a pending job (frank's PENDING row) contributes no
    # running usage and should not appear in the running-jobs usage table.
    assert "frank.example" not in users or users["frank.example"].job_count == 1


def test_my_jobs_held_job_has_no_prio_row_and_no_eta():
    nodes, jobs, prio, shares = _load()
    my_jobs = build_my_jobs(jobs, "alice.example", prio, shares)
    ids = {r.job.job_id: r for r in my_jobs}

    held = ids["4102899"]
    assert held.job.reason == "JobHeldUser"
    assert held.job.start_or_eta == "N/A"
    assert held.prio is None
    # fairshare is still attached from sshare, independent of per-job sprio row.
    assert held.fairshare == 0.351351


def test_my_jobs_priority_job_has_matching_prio_row():
    nodes, jobs, prio, shares = _load()
    my_jobs = build_my_jobs(jobs, "alice.example", prio, shares)
    ids = {r.job.job_id: r for r in my_jobs}

    scheduled = ids["4028669"]
    assert scheduled.job.start_or_eta == "2026-09-11T15:20:00"
    assert scheduled.prio is not None
    assert scheduled.prio.fairshare == 3514
