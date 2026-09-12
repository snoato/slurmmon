"""Human-readable glosses for Slurm's pending-job REASON codes.

Not exhaustive -- Slurm has dozens of these (see `man squeue`, JOB REASON
CODES), most tied to specific limit types that rarely come up in practice.
This covers the ones a user actually runs into day to day; anything else is
shown as-is with no added explanation rather than guessed at.
"""
from __future__ import annotations

_REASONS: dict[str, str] = {
    "Priority": "queued behind higher-priority jobs",
    "Dependency": "waiting on another job to finish",
    "Resources": "waiting for resources to free up",
    "JobHeldUser": "held by you",
    "JobHeldAdmin": "held by an admin",
    "BeginTime": "waiting for its scheduled start time",
    "ReqNodeNotAvail": "a required node isn't available",
    "NodeDown": "a required node is down",
    "PartitionDown": "partition is down",
    "PartitionInactive": "partition is inactive",
    "PartitionNodeLimit": "outside the partition's node-count limits",
    "PartitionTimeLimit": "exceeds the partition's time limit",
    "InvalidAccount": "invalid account",
    "InvalidQOS": "invalid QOS",
    "QOSJobLimit": "QOS job-count limit reached",
    "QOSResourceLimit": "QOS resource limit reached",
    "QOSUsageThreshold": "QOS usage threshold reached",
    "QOSMaxJobsPerUserLimit": "your per-user QOS job limit reached",
    "AssociationJobLimit": "account job-count limit reached",
    "AssociationResourceLimit": "account resource limit reached",
    "AssociationTimeLimit": "account time limit reached",
    "Licenses": "waiting on a software license",
    "Reservation": "waiting for an advance reservation",
    "BadConstraints": "job's constraints can't be satisfied",
    "SystemFailure": "a Slurm system failure occurred",
}


def explain_reason(reason: str) -> str | None:
    """Return a short plain-English gloss for a Slurm pending-job reason
    code, or None if we don't have one."""
    return _REASONS.get(reason)
