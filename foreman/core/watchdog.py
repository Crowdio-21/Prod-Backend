"""Periodic recovery for work that no worker event would ever pick up again.

Dispatch is otherwise driven by results, errors and registrations. When none of those can arrive (the
only busy worker vanished, a restarted phone no longer has its task, a scheduler deferred a task and
nothing is running), the job sits there. This loop:

* takes back tasks whose worker has been disconnected for ``disconnect_grace_s``;
* takes back tasks a reconnected worker showed no sign of for ``resume_grace_s``;
* takes back tasks with no result, progress or checkpoint for ``task_timeout_s``;
* offers pending tasks to idle workers.

Environment (seconds): CROWDIO_WATCHDOG_INTERVAL (10, 0 disables the loop), CROWDIO_DISCONNECT_GRACE (30),
CROWDIO_RESUME_GRACE (120), CROWDIO_TASK_TIMEOUT (600, 0 disables). Raise the task timeout if single tasks,
including a first model download, can legitimately run longer without reporting progress.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .utils import (
    _get_assigned_tasks,
    _get_pending_tasks,
    _requeue_task_if_assigned,
    _update_worker_status,
)


def _env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


@dataclass
class _Tracked:
    worker_id: Optional[str]
    active_at: float
    lost_since: Optional[float] = None
    resume_by: Optional[float] = None


class TaskWatchdog:
    def __init__(
        self,
        connection_manager,
        job_manager,
        task_dispatcher,
        completion_handler,
        *,
        interval_s: Optional[float] = None,
        disconnect_grace_s: Optional[float] = None,
        resume_grace_s: Optional[float] = None,
        task_timeout_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        def pick(value, name, default):
            return _env(name, default) if value is None else value

        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher
        self.completion_handler = completion_handler
        self.interval_s = pick(interval_s, "CROWDIO_WATCHDOG_INTERVAL", 10.0)
        self.disconnect_grace_s = pick(disconnect_grace_s, "CROWDIO_DISCONNECT_GRACE", 30.0)
        self.resume_grace_s = pick(resume_grace_s, "CROWDIO_RESUME_GRACE", 120.0)
        self.task_timeout_s = pick(task_timeout_s, "CROWDIO_TASK_TIMEOUT", 600.0)
        self.clock = clock
        self._tracked: Dict[str, _Tracked] = {}
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------ signals
    def touch(self, task_id: str) -> None:
        """The worker showed it is still working on ``task_id``."""
        entry = self._tracked.get(task_id)
        if entry is not None:
            entry.active_at = self.clock()
            entry.resume_by = None

    def expect_resume(self, task_id: str, worker_id: str) -> None:
        """``worker_id`` reconnected while holding ``task_id``; it may or may not still be running it."""
        now = self.clock()
        self._tracked[task_id] = _Tracked(worker_id, now, resume_by=now + self.resume_grace_s)

    # ------------------------------------------------------------ lifecycle
    def ensure_started(self) -> None:
        if self._task is None and self.interval_s > 0:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001 - the watchdog must outlive any one failure
                print(f"[WATCHDOG] check failed: {exc}")

    # ------------------------------------------------------------ checks
    async def tick(self) -> None:
        await self._recover()
        await self._dispatch()

    async def _recover(self) -> None:
        now = self.clock()
        assigned = await _get_assigned_tasks()
        for task_id in set(self._tracked) - {t.id for t in assigned}:
            del self._tracked[task_id]
        connected = self.connection_manager.get_all_worker_ids()
        for task in assigned:
            entry = self._tracked.get(task.id)
            if entry is None or entry.worker_id != task.worker_id:
                entry = self._tracked[task.id] = _Tracked(task.worker_id, now)
            reason, timed_out = None, False
            if task.worker_id not in connected:
                if entry.lost_since is None:
                    entry.lost_since = now
                if now - entry.lost_since >= self.disconnect_grace_s:
                    reason = "worker disconnected"
            else:
                entry.lost_since = None
                if entry.resume_by is not None and now >= entry.resume_by:
                    reason = "no sign of the task since the worker reconnected"
                elif self.task_timeout_s > 0 and now - entry.active_at >= self.task_timeout_s:
                    reason, timed_out = f"no result or progress for {self.task_timeout_s:.0f}s", True
            if reason is not None:
                await self._take_back(task, reason, timed_out)

    async def _take_back(self, task, reason: str, timed_out: bool) -> None:
        worker_id = task.worker_id
        if not await _requeue_task_if_assigned(task.id, worker_id):
            return  # the result arrived in the meantime
        self._tracked.pop(task.id, None)
        self.task_dispatcher.forget_dispatch(task.id, worker_id)
        print(f"[WATCHDOG] re-queued task {task.id} from worker {worker_id}: {reason}")
        if worker_id in self.connection_manager.get_all_worker_ids():
            active_task_id, _ = self.connection_manager.get_worker_active_task(worker_id)
            if active_task_id in (None, task.id):
                self.connection_manager.clear_worker_active_task(worker_id)
                self.connection_manager.mark_worker_available(worker_id)
                await _update_worker_status(worker_id, "online", current_task_id=None)
        # A task that keeps timing out counts against the same limit as one that keeps failing.
        if timed_out and self.job_manager.count_attempt(task.id):
            await self.completion_handler.handle_job_failure(task.job_id, f"task {task.id}: {reason}")

    async def _dispatch(self) -> None:
        if not self.connection_manager.get_available_workers():
            return
        for job_id in dict.fromkeys(t.job_id for t in await _get_pending_tasks()):
            func_code = self.job_manager.get_func_code(job_id)
            # Jobs from an earlier Foreman run have no cached code and no client: leave them
            if func_code and self.connection_manager.get_available_workers():
                await self.task_dispatcher.assign_tasks_for_job(job_id, func_code, [])
