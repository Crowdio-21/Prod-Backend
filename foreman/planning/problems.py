"""Problem report intake: rate-limited, de-duplicated per (type, stage, epoch).

Urgent types (NEXT_HOP_DOWN, OOM, LOAD_FAILED) are surfaced immediately so the
coordinator can reroute or replan; the rest are counted towards the
controller's ``problems`` trigger. During churn a storm of identical reports
collapses to one action per (type, stage, epoch), and a global cap drops the
excess instead of amplifying load.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from .contract import URGENT_PROBLEMS, ProblemMsg, ProblemType


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass(frozen=True)
class ProblemVerdict:
    status: str  # urgent | counted | duplicate | rate_limited | storm_dropped
    key: Optional[Tuple[str, Optional[int], Optional[int]]] = None


class ProblemAggregator:
    def __init__(
        self,
        per_worker_rate: float = 2.0,
        per_worker_burst: float = 5.0,
        dedupe_window_s: float = 10.0,
        global_cap_per_s: float = 50.0,
        count_window_s: float = 30.0,
    ) -> None:
        self.rate = per_worker_rate
        self.burst = per_worker_burst
        self.dedupe_window_s = dedupe_window_s
        self.global_cap_per_s = global_cap_per_s
        self.count_window_s = count_window_s
        self._buckets: Dict[str, _Bucket] = {}
        self._seen: Dict[Tuple[str, Optional[int], Optional[int]], float] = {}
        self._recent: Deque[float] = deque()
        self._global: Deque[float] = deque()
        self.stats: Dict[str, int] = {}

    def _count(self, status: str) -> None:
        self.stats[status] = self.stats.get(status, 0) + 1

    def accept(self, worker_id: str, problem: ProblemMsg, now: float) -> ProblemVerdict:
        while self._global and now - self._global[0] > 1.0:
            self._global.popleft()
        if len(self._global) >= self.global_cap_per_s:
            self._count("storm_dropped")
            return ProblemVerdict("storm_dropped")
        self._global.append(now)

        bucket = self._buckets.setdefault(worker_id, _Bucket(self.burst, now))
        bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
        bucket.updated = now
        if bucket.tokens < 1.0:
            self._count("rate_limited")
            return ProblemVerdict("rate_limited")
        bucket.tokens -= 1.0

        key = (problem.type.value, problem.stage, problem.epoch)
        last = self._seen.get(key)
        if last is not None and now - last < self.dedupe_window_s:
            self._count("duplicate")
            return ProblemVerdict("duplicate", key)
        self._seen[key] = now
        if problem.type in URGENT_PROBLEMS:
            self._count("urgent")
            return ProblemVerdict("urgent", key)
        self._recent.append(now)
        self._count("counted")
        return ProblemVerdict("counted", key)

    def recent_count(self, now: float) -> int:
        while self._recent and now - self._recent[0] > self.count_window_s:
            self._recent.popleft()
        return len(self._recent)

    def reset_epoch(self) -> None:
        self._recent.clear()
