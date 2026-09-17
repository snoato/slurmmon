"""Runs one bundled, sentinel-delimited query script and hands back the four
raw text blocks -- either over a reused (ControlMaster) SSH connection, or
directly as a local subprocess when Slurm client tools are already on this
machine's PATH.

Everything (partition discovery, sinfo, squeue, sprio, sshare) happens in a
single `bash -s` invocation (remote or local) so a refresh loop only ever
pays for one round trip, not four.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

_QUERY_SCRIPT = r"""
set -u
SUSER="$1"
MODE="$2"
ARG="$3"

case "$MODE" in
  list)
    PARTS=$(echo "$ARG" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -)
    ;;
  auto)
    # Partitions this user's Slurm account(s) are actually allowed to submit
    # to, derived from each partition's AllowAccounts/DenyAccounts ACL --
    # not group membership, which this cluster leaves as ALL everywhere.
    ACCTS=$(sacctmgr -n -P show assoc user="$SUSER" format=Account | sort -u | paste -sd, -)
    PARTS=$(scontrol show partition -o | awk -v accts="$ACCTS" '
      BEGIN {
        n = split(accts, alist, ",")
        for (i = 1; i <= n; i++) accset[alist[i]] = 1
      }
      {
        part = ""; allow = "ALL"; deny = ""
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^PartitionName=/) { sub(/^PartitionName=/, "", $i); part = $i }
          if ($i ~ /^AllowAccounts=/) { sub(/^AllowAccounts=/, "", $i); allow = $i }
          if ($i ~ /^DenyAccounts=/)  { sub(/^DenyAccounts=/, "", $i); deny = $i }
        }
        ok = 0
        if (allow == "ALL") { ok = 1 }
        else {
          split(allow, al, ",")
          for (j in al) if (al[j] in accset) ok = 1
        }
        if (deny != "" && deny != "(null)") {
          split(deny, dl, ",")
          for (j in dl) if (dl[j] in accset) ok = 0
        }
        if (ok && part != "") print part
      }' | sort -u | paste -sd, -)
    ;;
  prefix|*)
    # "__EMPTY__" is a stand-in for an intentionally empty prefix (match
    # every partition) -- see fetch_raw() for why a real empty string can't
    # be sent as-is.
    if [ "$ARG" = "__EMPTY__" ]; then ARG=""; fi
    PARTS=$(sinfo -h -o '%P' | tr -d '*' | grep "^${ARG}" | sort -u | paste -sd, -)
    ;;
esac

if [ -z "$PARTS" ]; then
  echo "NO_PARTITIONS_MATCHED: mode=${MODE} arg=${ARG}" >&2
  exit 3
fi

echo "__SENTINEL__PARTITIONS__"
echo "$PARTS"

echo "__SENTINEL__NOW__"
date +%Y-%m-%dT%H:%M:%S

echo "__SENTINEL__SINFO__"
sinfo -h -N -p "$PARTS" -o '%N|%P|%T|%C|%m|%G'

echo "__SENTINEL__SQUEUE__"
squeue -h -p "$PARTS" -o '%i|%u|%P|%T|%D|%C|%m|%b|%M|%r|%S|%N|%V|%j'

echo "__SENTINEL__SPRIO__"
sprio -h -u "$SUSER" 2>/dev/null || true

echo "__SENTINEL__SSHARE__"
sshare -n -P -u "$SUSER" -o Account,User,RawShares,RawUsage,FairShare,NormUsage 2>/dev/null || true

echo "__SENTINEL__END__"
"""


class SlurmFetchError(RuntimeError):
    """Raised when the bundled query command fails or times out."""


@dataclass(frozen=True)
class PartitionSelector:
    """How to pick which partitions to monitor.

    mode="prefix": arg is a partition-name prefix (e.g. "gpu_"), matched via
      `sinfo`. An empty prefix matches every partition.
    mode="list": arg is a comma-separated list of exact partition names.
    mode="auto": arg is ignored; partitions are derived from which of the
      user's Slurm accounts each partition's AllowAccounts/DenyAccounts ACL
      permits.
    """

    mode: str
    arg: str = ""

    def describe(self) -> str:
        if self.mode == "auto":
            return "auto (your allowed partitions)"
        if self.mode == "list":
            return self.arg
        if not self.arg:
            return "all"
        return f"{self.arg}*"


@dataclass
class RawSections:
    partitions: str
    now: str
    sinfo: str
    squeue: str
    sprio: str
    sshare: str


def _control_path() -> str:
    # Use /tmp directly rather than tempfile.gettempdir(): on macOS that
    # resolves to a long /var/folders/... path that blows past the ~104-byte
    # UNIX socket path limit once ssh appends its %C hash.
    base = f"/tmp/slurmmon-cm-{os.getuid()}"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "%C")


def build_ssh_command(host: str, ssh_opts: list[str] | None = None) -> list[str]:
    opts = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=10m",
        "-o", f"ControlPath={_control_path()}",
        *(ssh_opts or []),
    ]
    return ["ssh", *opts, host, "bash", "-s", "--"]


def fetch_raw_remote(host: str, slurm_user: str, selector: PartitionSelector, timeout: float = 15.0) -> RawSections:
    """Run the bundled query over SSH against `host`."""
    # ssh joins the remote command's argv with plain spaces (no quoting), so
    # an empty-string argument (mode="auto", or an empty mode="prefix" for
    # "all partitions") vanishes entirely instead of arriving as an empty
    # positional parameter -- always send a non-empty placeholder; the
    # script restores it to "" where that's meaningful (prefix mode).
    arg = selector.arg or "__EMPTY__"
    cmd = build_ssh_command(host) + [slurm_user, selector.mode, arg]
    return _run_and_split(cmd, timeout, error_prefix=f"ssh {host}")


def fetch_raw_local(slurm_user: str, selector: PartitionSelector, timeout: float = 15.0) -> RawSections:
    """Run the bundled query directly on this machine (no SSH) -- for use
    when Slurm client tools are already on this machine's PATH, e.g. running
    slurmmon on the cluster's own login node."""
    cmd = ["bash", "-s", "--", slurm_user, selector.mode, selector.arg]
    return _run_and_split(cmd, timeout, error_prefix="local slurm query")


def fetch_job_detail(host: str | None, jobid: str, local: bool, timeout: float = 10.0) -> str:
    """Run `scontrol show job <jobid>` on demand (not part of the regular
    bundled poll) and return its raw output. Reuses the same ControlMaster
    socket as the polling connection in remote mode, so it rides the
    already-open connection instead of paying for a new one."""
    if local:
        cmd = ["scontrol", "show", "job", jobid]
    else:
        assert host is not None
        opts = [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=10m",
            "-o", f"ControlPath={_control_path()}",
        ]
        cmd = ["ssh", *opts, host, "scontrol", "show", "job", jobid]

    error_prefix = "local scontrol" if local else f"ssh {host}"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SlurmFetchError(f"{error_prefix} timed out after {timeout}s") from exc
    except OSError as exc:
        raise SlurmFetchError(f"failed to run {error_prefix}: {exc}") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise SlurmFetchError(f"{error_prefix} exited {proc.returncode}: {stderr or '(no stderr)'}")
    return proc.stdout


def _run_and_split(cmd: list[str], timeout: float, error_prefix: str) -> RawSections:
    try:
        proc = subprocess.run(
            cmd,
            input=_QUERY_SCRIPT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SlurmFetchError(f"{error_prefix} timed out after {timeout}s") from exc
    except OSError as exc:
        raise SlurmFetchError(f"failed to run {error_prefix}: {exc}") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise SlurmFetchError(f"{error_prefix} exited {proc.returncode}: {stderr or '(no stderr)'}")

    return _split_sections(proc.stdout)


def _split_sections(output: str) -> RawSections:
    markers = [
        "__SENTINEL__PARTITIONS__",
        "__SENTINEL__NOW__",
        "__SENTINEL__SINFO__",
        "__SENTINEL__SQUEUE__",
        "__SENTINEL__SPRIO__",
        "__SENTINEL__SSHARE__",
        "__SENTINEL__END__",
    ]
    positions = []
    for m in markers:
        idx = output.find(m)
        if idx == -1:
            raise SlurmFetchError(f"malformed remote output: missing {m!r}")
        positions.append((idx, m))
    positions.sort()

    blocks: dict[str, str] = {}
    for i in range(len(positions) - 1):
        start_idx, name = positions[i]
        end_idx, _ = positions[i + 1]
        start = start_idx + len(name)
        blocks[name] = output[start:end_idx].strip("\n")

    return RawSections(
        partitions=blocks["__SENTINEL__PARTITIONS__"],
        now=blocks["__SENTINEL__NOW__"],
        sinfo=blocks["__SENTINEL__SINFO__"],
        squeue=blocks["__SENTINEL__SQUEUE__"],
        sprio=blocks["__SENTINEL__SPRIO__"],
        sshare=blocks["__SENTINEL__SSHARE__"],
    )
