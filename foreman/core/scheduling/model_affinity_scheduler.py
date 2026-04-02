"""Model-affinity scheduler wrapper for DNN pipeline workloads.

Wraps any base scheduler and re-orders candidate workers by model
residency so that workers already holding the required partition are
preferred.  For tasks without a model requirement the base scheduler
is used unmodified.

Priority tiers (highest → lowest):
  1. Worker has the partition loaded in memory  (zero reload cost)
  2. Worker has the partition cached on disk     (skip download)
  3. Worker has NO model loaded                  (idle – assign new stage)
  4. Worker has a DIFFERENT model loaded         (requires unload + load)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .scheduler_interface import Task, TaskScheduler, Worker


class ModelAffinityScheduler(TaskScheduler):
    """Scheduler decorator that adds model-affinity awareness."""

    def __init__(
        self,
        base_scheduler: TaskScheduler,
        model_load_tracker,
    ):
        self._base = base_scheduler
        self._tracker = model_load_tracker

    # -------------------------------------------------------------- #
    #  Internal helpers
    # -------------------------------------------------------------- #

    def _partition_id_for_task(self, task: Task) -> Optional[str]:
        """Extract required model_partition_id from a Task's metadata."""
        # The SchedulerTask dataclass doesn't carry device_requirements
        # directly, but TaskDispatcher stores model_partition_id in
        # execution_metadata which it reads from the DB task object.
        # We use a lightweight convention: if the task has an attribute
        # ``model_partition_id`` (set by the dispatcher before scheduling)
        # we use it.
        return getattr(task, "model_partition_id", None)

    def _tiered_candidates(
        self,
        partition_id: str,
        available_workers: Set[str],
    ) -> List[Set[str]]:
        """Return candidate worker sets ordered by affinity tier."""
        resident = set(self._tracker.workers_with_model_resident(partition_id))
        cached = set(self._tracker.workers_with_model_cached(partition_id))

        tier1 = available_workers & resident
        tier2 = (available_workers & cached) - tier1
        tier3 = (
            {
                w
                for w in available_workers
                if self._tracker.get_resident_model(w) is None
            }
            - tier1
            - tier2
        )
        tier4 = available_workers - tier1 - tier2 - tier3

        return [tier1, tier2, tier3, tier4]

    # -------------------------------------------------------------- #
    #  TaskScheduler interface
    # -------------------------------------------------------------- #

    async def select_worker(
        self,
        task: Task,
        available_workers: Set[str],
        all_workers: Dict[str, Worker],
    ) -> Optional[str]:
        partition_id = self._partition_id_for_task(task)
        if not partition_id:
            return await self._base.select_worker(task, available_workers, all_workers)

        for tier in self._tiered_candidates(partition_id, available_workers):
            if not tier:
                continue
            selected = await self._base.select_worker(task, tier, all_workers)
            if selected:
                return selected

        return await self._base.select_worker(task, available_workers, all_workers)

    async def select_task(
        self,
        pending_tasks: List[Task],
        worker_id: str,
    ) -> Optional[Task]:
        return await self._base.select_task(pending_tasks, worker_id)

    async def batch_select_workers(
        self,
        tasks: List[Task],
        available_workers: Set[str],
        all_workers: Dict[str, Worker],
    ) -> List[tuple]:
        """Batch assignment with per-task affinity awareness."""
        assignments: list[tuple] = []
        remaining = set(available_workers)

        for task in tasks:
            if not remaining:
                break
            worker_id = await self.select_worker(task, remaining, all_workers)
            if worker_id:
                assignments.append((task, worker_id))
                remaining.discard(worker_id)

        return assignments
