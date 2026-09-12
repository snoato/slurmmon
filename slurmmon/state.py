"""Shared app-level state passed from the fetch/aggregate step to the UI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .aggregate import MyJobRow, PartitionStats, UserUsage
from .parse import Job, Node
from .ssh_client import PartitionSelector


@dataclass
class AppState:
    host: str | None  # None when local=True
    slurm_user: str
    selector: PartitionSelector
    interval: float
    local: bool = False

    partitions: list[str] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    partition_stats: dict[str, PartitionStats] = field(default_factory=dict)
    cluster_stats: PartitionStats | None = None
    user_usage: list[UserUsage] = field(default_factory=list)
    my_jobs: list[MyJobRow] = field(default_factory=list)

    last_success: float | None = None
    last_error: str | None = None
    fetching: bool = False
    server_now: str | None = None  # "now" as reported by the query host itself, for tz-correct "queued Xh ago"

    screen: str = "overview"  # overview | nodes | users | jobs
    scroll: int = 0  # row offset into the current detail screen's list; reset on screen switch

    @property
    def partitions_desc(self) -> str:
        return self.selector.describe()

    @property
    def is_stale(self) -> bool:
        if self.last_success is None:
            return True
        return (time.time() - self.last_success) > (self.interval * 3)

    @property
    def age_seconds(self) -> float | None:
        if self.last_success is None:
            return None
        return time.time() - self.last_success
