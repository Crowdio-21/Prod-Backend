"""Adaptive scheduler: placement driven by what the phones report, with no method to choose.

It is the bandwidth-aware scheduler (completion time = rtt + bytes / bandwidth + learned compute
+ result bytes / bandwidth, mapped with Sufferage) with the cost of each worker scaled by how
healthy that worker is right now:

    T'(task, worker) = T(task, worker) * penalty(worker)

``penalty`` is at least 1 and comes only from signals the phones already report:

* **Reliability.** A failed attempt wastes about one task time and is retried, so the expected
  time is T / p, where p is the chance an attempt succeeds. p is
  ``(completed + k) / (completed + failed + k)`` with ``k = prior_successes`` (default 5)
  pseudo-successes. A phone never seen before is not penalised (p = 1, so it gets tried and
  builds a record); each failure pulls p down, and a long clean record keeps it at 1. p is
  floored (``min_reliability``) so one bad patch cannot exile a phone for good.
* **Battery.** A phone that is discharging below ``low_battery`` percent (default 20, the
  threshold ``Worker.is_low_battery`` already uses) costs ``low_battery_penalty`` times more,
  and below ``critical_battery`` percent ``critical_battery_penalty`` times more, so work
  drains to charging or well-charged phones first without ever being refused outright.

When every phone is healthy the penalty is 1 for all of them and the schedule is exactly the
bandwidth-aware one; the scheduler only departs from it when the data says a phone is worse than
its links and compute history suggest. Each decision records the penalty it used, and
``link_snapshot`` reports every phone's penalty with the reasons, so a placement can be explained.

Thermal state is not reported by the workers today (``Worker`` has no such field), so it is not
modelled here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Set

from .bandwidth_scheduler import BandwidthAwareScheduler, LinkEstimate, TaskProfile
from .scheduler_interface import Task


class AdaptiveScheduler(BandwidthAwareScheduler):
    def __init__(
        self,
        *args,
        min_reliability: float = 0.2,
        prior_successes: float = 5.0,
        low_battery: float = 20.0,
        critical_battery: float = 10.0,
        low_battery_penalty: float = 1.5,
        critical_battery_penalty: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < min_reliability <= 1.0:
            raise ValueError("min_reliability must be in (0, 1]")
        self.min_reliability = min_reliability
        self.prior_successes = prior_successes
        self.low_battery = low_battery
        self.critical_battery = critical_battery
        self.low_battery_penalty = low_battery_penalty
        self.critical_battery_penalty = critical_battery_penalty
        self._penalty: Dict[str, float] = {}
        self._why: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------ health model
    def health(self, worker: object) -> Dict[str, float]:
        """Penalty factors for one worker record (anything with the ``Worker`` attributes)."""
        completed = max(0, int(getattr(worker, "tasks_completed", 0) or 0))
        failed = max(0, int(getattr(worker, "tasks_failed", 0) or 0))
        k = self.prior_successes
        reliability = max(self.min_reliability, (completed + k) / (completed + failed + k))
        factors = {"reliability": reliability, "reliability_factor": 1.0 / reliability, "battery_factor": 1.0}
        level = getattr(worker, "battery_level", None)
        charging = getattr(worker, "is_charging", True)
        if level is not None and not charging:
            if level < self.critical_battery:
                factors["battery_factor"] = self.critical_battery_penalty
            elif level < self.low_battery:
                factors["battery_factor"] = self.low_battery_penalty
        factors["penalty"] = factors["reliability_factor"] * factors["battery_factor"]
        return factors

    def _refresh(self, all_workers: Mapping[str, object]) -> None:
        for worker_id, worker in all_workers.items():
            factors = self.health(worker)
            self._penalty[worker_id] = factors["penalty"]
            self._why[worker_id] = factors

    # ------------------------------------------------------------ cost model
    def cost(self, profile: TaskProfile, worker_id: str, est: LinkEstimate,
             level_down: float = math.inf, level_up: float = math.inf) -> float:
        return super().cost(profile, worker_id, est, level_down, level_up) * self._penalty.get(worker_id, 1.0)

    # ------------------------------------------------------------ selection
    async def select_worker(self, task: Task, available_workers: Set[str], all_workers: dict) -> Optional[str]:
        self._refresh(all_workers)
        return await super().select_worker(task, available_workers, all_workers)

    async def select_task(self, pending_tasks: List[Task], worker_id: str) -> Optional[Task]:
        return await super().select_task(pending_tasks, worker_id)

    async def batch_select_workers(self, tasks: List[Task], available_workers: Set[str],
                                   all_workers: dict) -> List[tuple]:
        self._refresh(all_workers)
        return await super().batch_select_workers(tasks, available_workers, all_workers)

    # ------------------------------------------------------------ feedback
    def _commit(self, profile: TaskProfile, worker_id: str, seconds: float, est: LinkEstimate) -> None:
        super()._commit(profile, worker_id, seconds, est)
        self.decisions[-1]["health_penalty"] = round(self._penalty.get(worker_id, 1.0), 3)

    def link_snapshot(self) -> dict:
        snapshot = super().link_snapshot()
        snapshot["health"] = {w: {k: round(v, 3) for k, v in why.items()} for w, why in sorted(self._why.items())}
        return snapshot
