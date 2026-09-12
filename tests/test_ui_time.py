from slurmmon.reasons import explain_reason
from slurmmon.ui import _fmt_duration, _parse_slurm_timestamp, _queued_for


def test_parse_slurm_timestamp_valid():
    dt = _parse_slurm_timestamp("2026-09-11T14:06:59")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2026, 9, 11, 14, 6, 59)


def test_parse_slurm_timestamp_na_is_none():
    assert _parse_slurm_timestamp("N/A") is None
    assert _parse_slurm_timestamp("") is None
    assert _parse_slurm_timestamp("Unknown") is None


def test_fmt_duration_buckets():
    assert _fmt_duration(30) == "30s"
    assert _fmt_duration(90) == "1m"
    assert _fmt_duration(3661) == "1h1m"
    assert _fmt_duration(90000) == "1d1h"


def test_queued_for_uses_server_now_not_wallclock():
    # Even if the local wall clock were wildly different, _queued_for must
    # only ever compare against the passed-in server_now.
    queued = _queued_for("2026-09-11T12:00:00", server_now="2026-09-11T13:30:00")
    assert queued == "1h30m"


def test_queued_for_falls_back_to_wallclock_when_no_server_now():
    # Can't assert an exact value without freezing time, just that it
    # doesn't blow up and returns *something* duration-shaped.
    queued = _queued_for("2020-01-01T00:00:00", server_now=None)
    assert queued is not None
    assert queued.endswith("d") or "d" in queued


def test_queued_for_unparseable_submit_time_returns_none():
    assert _queued_for("N/A", server_now="2026-09-11T13:30:00") is None


def test_explain_reason_known():
    assert explain_reason("JobHeldUser") == "held by you"
    assert explain_reason("Priority") == "queued behind higher-priority jobs"


def test_explain_reason_unknown_returns_none():
    assert explain_reason("SomeBrandNewReasonCode") is None
