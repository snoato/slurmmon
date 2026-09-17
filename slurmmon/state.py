"""Shared app-level state passed from the fetch/aggregate step to the UI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .aggregate import MyJobRow, PartitionStats, UserUsage
from .parse import Job, Node
from .ssh_client import PartitionSelector

# "headers" keeps every job visible (unlike "summary", which collapses a
# group to one row) while still surfacing per-group totals -- the more
# informative default of the two grouped modes.
DEFAULT_GROUP_MODE = "headers"


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

    screen: str = "overview"  # overview | nodes | users | jobs | myjobs
    scroll: int = 0  # row offset into the current detail screen's list; reset on screen switch
    row_delta: int = 0  # user-adjusted +/- on the auto page size ('[' / ']'); persists across screens

    selected: int = 0  # cursor row on the jobs/myjobs screens; reset on screen switch or filter change
    job_filter: str = "ALL"  # ALL | RUNNING | PENDING -- cycled with 'f' on the myjobs screen
    current_list_job_ids: list[str] = field(default_factory=list)  # job ids (or, when current_list_is_groups, name-prefixes) in display order, for Enter lookup
    current_list_is_groups: bool = False  # whether current_list_job_ids holds group prefixes rather than real job ids

    group_mode: str = DEFAULT_GROUP_MODE  # off | headers | summary -- cycled with 'g' on jobs/myjobs (see GROUP_MODES)
    name_filter: str | None = None  # set when Enter drills out of "summary" mode; restricts the flat list to that prefix

    detail_job_id: str | None = None  # non-None while the job-detail overlay is open
    detail_text: str | None = None
    detail_error: str | None = None
    detail_loading: bool = False
    detail_scroll: int = 0

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
