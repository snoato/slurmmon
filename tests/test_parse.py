from pathlib import Path

from slurmmon.parse import (
    classify_node_state,
    parse_gres,
    parse_mem_mb,
    parse_sinfo_line,
    parse_sprio_line,
    parse_squeue_line,
    parse_sshare_line,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text().splitlines()


def test_classify_node_state_idle():
    assert classify_node_state("idle") == ("idle", "idle")


def test_classify_node_state_busy_variants():
    assert classify_node_state("mixed") == ("mixed", "busy")
    assert classify_node_state("mixed-") == ("mixed", "busy")
    assert classify_node_state("allocated") == ("allocated", "busy")


def test_classify_node_state_down_and_flags():
    assert classify_node_state("down*") == ("down", "unavailable")
    assert classify_node_state("down") == ("down", "unavailable")
    assert classify_node_state("drain$") == ("drain", "unavailable")


def test_classify_node_state_compound():
    # scontrol-style compound state; only the first component matters for
    # scheduling-availability purposes.
    assert classify_node_state("mixed+planned") == ("mixed", "busy")


def test_parse_gres_node_whole_gpu_and_shard():
    g = parse_gres("gpu:A10:4,shard:A10:96")
    assert g.gpu == 4
    assert g.shard == 96
    assert g.gpu_model == "A10"


def test_parse_gres_job_side_prefixed():
    assert parse_gres("gres/gpu:1").gpu == 1
    assert parse_gres("gres/shard:8").shard == 8


def test_parse_gres_empty():
    for raw in ("(null)", "N/A", "", "n/a"):
        g = parse_gres(raw)
        assert g.is_empty


def test_parse_mem_mb_bare_digits_is_mb():
    assert parse_mem_mb("190000") == 190000.0


def test_parse_mem_mb_suffixed():
    assert parse_mem_mb("64G") == 64 * 1024
    assert parse_mem_mb("8G") == 8 * 1024


def test_parse_mem_mb_empty():
    assert parse_mem_mb("N/A") == 0.0
    assert parse_mem_mb("") == 0.0


def test_parse_sinfo_fixture_rows():
    nodes = [n for line in _lines("sinfo.txt") if (n := parse_sinfo_line(line))]
    assert len(nodes) == 12

    down_node = next(n for n in nodes if n.name == "cn-102")
    assert down_node.bucket == "unavailable"
    assert down_node.cpus_total == 96

    gpu_node = next(n for n in nodes if n.name == "cn-401")
    assert gpu_node.bucket == "busy"
    assert gpu_node.gres.gpu == 4
    assert gpu_node.gres.shard == 96

    planned_node = next(n for n in nodes if n.name == "cn-427")
    assert planned_node.state == "mixed"
    assert planned_node.bucket == "busy"


def test_parse_squeue_fixture_rows():
    jobs = [j for line in _lines("squeue.txt") if (j := parse_squeue_line(line))]
    assert len(jobs) == 11

    held = next(j for j in jobs if j.job_id == "4102899")
    assert held.is_pending
    assert held.reason == "JobHeldUser"
    assert held.start_or_eta == "N/A"
    assert held.gres.gpu == 1

    eta_job = next(j for j in jobs if j.job_id == "4028669")
    assert eta_job.reason == "Priority"
    assert eta_job.start_or_eta == "2026-09-11T15:20:00"

    running = next(j for j in jobs if j.job_id == "4107107_0")
    assert running.is_running
    assert running.gres.shard == 8
    assert running.nodelist == "cn-401"


def test_parse_sprio_fixture_rows():
    rows = [p for line in _lines("sprio.txt") if (p := parse_sprio_line(line))]
    assert len(rows) == 2
    row = next(r for r in rows if r.job_id == "4028669")
    assert row.priority == 23572
    assert row.age == 59
    assert row.fairshare == 3514
    assert row.qos == 20000


def test_parse_sshare_fixture_rows():
    rows = [s for line in _lines("sshare.txt") if (s := parse_sshare_line(line))]
    assert len(rows) == 6
    mine = next(r for r in rows if r.user == "alice.example")
    assert mine.account == "lab"
    assert mine.fairshare == 0.351351
    assert mine.norm_usage == 0.053928

    root_row = next(r for r in rows if r.account == "root")
    assert root_row.user == ""
    assert root_row.fairshare is None
