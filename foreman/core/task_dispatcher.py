"""
Task dispatching with pluggable scheduling algorithms
"""

import json
from typing import List, Optional, Any, Dict

from .scheduling import TaskScheduler, Task as SchedulerTask, Worker
from .utils import (
    _get_pending_tasks,
    _get_assigned_tasks,
    _update_task_status,
    _update_worker_status,
    _claim_task_for_worker,
    _get_latest_task_failure_with_checkpoint,
)
from common.protocol import create_assign_task_message, create_resume_task_message
from common.serializer import get_runtime_info


class TaskDispatcher:
    """
    Dispatches tasks to workers using pluggable scheduling algorithms

    Responsibilities:
    - Use scheduler to select best worker for task
    - Use scheduler to select best task for worker
    - Send task assignments to workers
    - Update task and worker status
    """

    def __init__(self, scheduler: TaskScheduler, connection_manager, job_manager):
        """
        Initialize task dispatcher

        Args:
            scheduler: Scheduling algorithm to use
            connection_manager: ConnectionManager instance
            job_manager: JobManager instance
        """
        self.scheduler = scheduler
        self.connection_manager = connection_manager
        self.job_manager = job_manager

    # ==================== Orphan Task Recovery ====================

    async def recover_orphaned_tasks(self) -> int:
        """
        Find tasks assigned to disconnected workers and reset them to pending
        
        An orphaned task is one where:
        - Status is "assigned"
        - The assigned worker_id is no longer connected via WebSocket
        
        Returns:
            Number of tasks recovered
        """
        # Get all connected worker IDs
        connected_workers = self.connection_manager.get_all_worker_ids()
        
        # Get all tasks that are currently assigned
        assigned_tasks = await _get_assigned_tasks()
        
        if not assigned_tasks:
            return 0
        
        recovered = 0
        for task in assigned_tasks:
            # Check if the assigned worker is still connected
            if task.worker_id and task.worker_id not in connected_workers:
                # Worker is disconnected - reset task to pending for reassignment
                await _update_task_status(task.id, "pending", worker_id=None)
                recovered += 1
                print(
                    f"TaskDispatcher: 🔄 Recovered orphaned task {task.id} "
                    f"from disconnected worker {task.worker_id}"
                )
        
        if recovered > 0:
            print(f"TaskDispatcher: Recovered {recovered} orphaned tasks for reassignment")
        
        return recovered

    # ==================== Task Assignment ====================

    async def assign_tasks_for_job(
        self, job_id: str, func_code: str, args_list: List[Any]
    ) -> int:
        """
        Assign available tasks for a specific job to available workers

        Args:
            job_id: Job identifier
            func_code: Function code to execute
            args_list: List of task arguments

        Returns:
            Number of tasks assigned
        """
        print(f"TaskDispatcher: Assigning tasks for job {job_id}")

        # Get pending tasks for this job
        pending_tasks = await _get_pending_tasks(job_id)

        if not pending_tasks:
            print(f"TaskDispatcher: No pending tasks for job {job_id}")
            return 0

        tasks_assigned = 0

        for task in pending_tasks:
            available_workers = self.connection_manager.get_available_workers()
            

            if not available_workers:
                print(
                    f"TaskDispatcher: No available workers, {len(pending_tasks) - tasks_assigned} tasks remain"
                )
                break

            # Get worker objects for scheduler
            all_workers = await self._get_all_workers()

            # Create scheduler task
            scheduler_task = SchedulerTask(
                id=task.id,
                job_id=task.job_id,
                args=task.args,
                priority=getattr(task, "priority", 0),
            )

            # Let scheduler select best worker
            worker_id = await self.scheduler.select_worker(
                scheduler_task, available_workers, all_workers
            )

            if worker_id:
                task_args = json.loads(task.args) if task.args else []
                success = await self._assign_task_to_worker(
                    job_id, task.id, func_code, task_args, worker_id
                )
                if success:
                    tasks_assigned += 1

        print(f"TaskDispatcher: Assigned {tasks_assigned} tasks for job {job_id}")
        return tasks_assigned

    async def assign_task_to_available_worker(
        self, worker_id: str, worker_threshold: int = 2
    ) -> bool:
        """
        Assign pending tasks when enough workers are available

        When the number of available workers exceeds the threshold,
        triggers batch assignment for all jobs (like initial start).
        Otherwise, skips assignment and waits for more workers.
        
        Also recovers orphaned tasks (assigned to disconnected workers)
        before attempting new assignments.

        Args:
            worker_id: Worker identifier that just became available
            worker_threshold: Minimum number of available workers before batch assignment

        Returns:
            True if tasks were assigned, False otherwise
        """
        # First, recover any orphaned tasks from disconnected workers
        recovered = await self.recover_orphaned_tasks()
        if recovered > 0:
            print(f"TaskDispatcher: Recovered {recovered} orphaned tasks before assignment")
        
        # Check how many workers are currently available
        available_workers = self.connection_manager.get_available_workers()
        num_available = len(available_workers)

        print(
            f"TaskDispatcher: Worker {worker_id} available. "
            f"Total available workers: {num_available}, threshold: {worker_threshold}"
        )

        # If not enough workers available, wait for more
        if num_available < worker_threshold:
            print(
                f"TaskDispatcher: Waiting for more workers "
                f"({num_available}/{worker_threshold} available)"
            )
            return False

        # Enough workers available - trigger batch assignment for all jobs
        print(
            f"TaskDispatcher: Threshold reached ({num_available} >= {worker_threshold}), "
            f"triggering batch assignment for all jobs"
        )

        # Get all pending tasks to find unique job IDs
        pending_tasks = await _get_pending_tasks()

        if not pending_tasks:
            print(f"TaskDispatcher: No pending tasks available")
            return False

        # Get unique job IDs from pending tasks
        job_ids = set(task.job_id for task in pending_tasks)

        total_assigned = 0
        for job_id in job_ids:
            # Get func_code from job manager
            func_code = self.job_manager.get_func_code(job_id)

            if not func_code:
                print(f"TaskDispatcher: No func_code found for job {job_id}, skipping")
                continue

            # Use assign_tasks_for_job (like initial start) for batch assignment
            assigned = await self.assign_tasks_for_job(job_id, func_code, [])
            total_assigned += assigned

        print(f"TaskDispatcher: Batch assignment completed, {total_assigned} tasks assigned")
        return total_assigned > 0

    async def _assign_task_to_worker(
        self, job_id: str, task_id: str, func_code: str, task_args: Any, worker_id: str
    ) -> bool:
        """
        Assign a specific task to a specific worker.
        
        If the task has checkpoint data from a previous failure, sends a RESUME_TASK
        message so the worker can continue from where the last worker left off.
        Otherwise, sends a regular ASSIGN_TASK message to start fresh.

        Args:
            job_id: Job identifier
            task_id: Task identifier
            func_code: Function code to execute
            task_args: Task arguments
            worker_id: Worker identifier

        Returns:
            True if assignment successful, False otherwise
        """
        try:
            # Check if we have checkpoint data for this task from a previous failure
            checkpoint_data = await _get_latest_task_failure_with_checkpoint(task_id)
            
            if checkpoint_data and checkpoint_data.get("state"):
                # We have checkpoint data - send RESUME_TASK message
                message = create_resume_task_message(
                    task_id=task_id,
                    job_id=job_id,
                    func_code=func_code,
                    checkpoint_state=checkpoint_data.get("state", {}),
                    task_args=[task_args] if not isinstance(task_args, list) else task_args,
                    task_kwargs={},
                    progress_percent=checkpoint_data.get("progress_percent", 0),
                    checkpoint_count=checkpoint_data.get("checkpoint_count", 0)
                )
                is_resume = True
                print(
                    f"TaskDispatcher: 🔄 Resuming task {task_id} from checkpoint "
                    f"(progress: {checkpoint_data.get('progress_percent', 0):.1f}%)"
                )
            else:
                # No checkpoint - send regular ASSIGN_TASK message
                message = create_assign_task_message(
                    func_code, [task_args], task_id, job_id  # Wrap in list for single task
                )
                is_resume = False

            # Get worker websocket
            websocket = self.connection_manager.get_worker_websocket(worker_id)

            if not websocket:
                print(f"TaskDispatcher: No websocket found for worker {worker_id}")
                return False

            # Atomically claim task; if someone else grabbed it, skip
            claimed_job_id = await _claim_task_for_worker(task_id, worker_id)
            if not claimed_job_id:
                print(f"TaskDispatcher: Task {task_id} was already claimed, skipping")
                return False

            # Send to worker
            await websocket.send(message.to_json())

            # Mark worker as busy
            self.connection_manager.mark_worker_busy(worker_id)
            await _update_worker_status(worker_id, "busy", current_task_id=task_id)

            action = "Resumed" if is_resume else "Assigned"
            print(
                f"TaskDispatcher: {action} task {task_id} to worker {worker_id} | foreman_runtime={get_runtime_info()}"
            )

            return True

        except Exception as e:
            print(
                f"TaskDispatcher: Error assigning task {task_id} to worker {worker_id}: {e}"
            )

            # Put worker back in available pool on error
            self.connection_manager.mark_worker_available(worker_id)

            # If we had claimed the task, put it back to pending
            try:
                await _update_task_status(task_id, "pending", worker_id=None)
            except Exception as reset_error:
                print(
                    f"TaskDispatcher: Error resetting task {task_id} after send failure: {reset_error}"
                )

            try:
                await _update_worker_status(worker_id, "online", current_task_id=None)
            except Exception as status_error:
                print(f"TaskDispatcher: Error updating worker status: {status_error}")

            return False

    # ==================== Worker Information ====================

    async def _get_all_workers(self) -> Dict[str, Worker]:
        """
        Get all worker objects with EXTENDED device specifications for MCDM

        Returns workers with:
        - Basic stats: tasks_completed, tasks_failed, status
        - Device specs: cpu_cores, cpu_frequency, ram, battery
        - Performance: cpu_usage, success_rate, avg_duration
        """
        all_worker_ids = self.connection_manager.get_all_worker_ids()
        workers = {}

        # Import here to avoid circular dependency (explicit module path)
        from foreman.core.utils.utils import _get_worker_stats_extended

        for worker_id in all_worker_ids:
            try:
                # Fetch COMPLETE worker data including device specs
                stats = await _get_worker_stats_extended(worker_id)

                if stats:
                    # Calculate success rate
                    completed = stats.get("tasks_completed", 0)
                    failed = stats.get("tasks_failed", 0)
                    total = completed + failed
                    success_rate = completed / total if total > 0 else 1.0

                    workers[worker_id] = Worker(
                        id=worker_id,
                        status=stats.get("status", "online"),
                        tasks_completed=completed,
                        tasks_failed=failed,
                        current_task_id=stats.get("current_task_id"),
                        # Device specifications
                        cpu_cores=stats.get("cpu_cores", 1),
                        cpu_threads=stats.get("cpu_threads", 1),
                        cpu_frequency_mhz=stats.get("cpu_frequency_mhz", 1000.0),
                        cpu_usage_percent=stats.get("cpu_usage_percent", 0.0),
                        cpu_model=stats.get("cpu_model"),
                        ram_total_mb=stats.get("ram_total_mb", 1024.0),
                        ram_available_mb=stats.get("ram_available_mb", 512.0),
                        # Battery/Power
                        battery_level=stats.get("battery_level", 100.0),
                        is_charging=stats.get("is_charging", True),
                        # Network
                        network_type=stats.get("network_type", "WiFi"),
                        network_speed_mbps=stats.get("network_speed_mbps", 10.0),
                        # Performance metrics
                        avg_task_duration_sec=stats.get("avg_task_duration_sec", 0.0),
                        success_rate=success_rate,
                        # GPU
                        gpu_available=stats.get("gpu_available", False),
                        gpu_model=stats.get("gpu_model"),
                        # Storage
                        storage_available_gb=stats.get("storage_available_gb", 0.0),
                        # Device info
                        device_type=stats.get("device_type"),
                        os_type=stats.get("os_type"),
                        os_version=stats.get("os_version"),
                    )
                else:
                    # Worker exists but no stats yet - use defaults
                    workers[worker_id] = Worker(
                        id=worker_id,
                        status="online",
                        tasks_completed=0,
                        tasks_failed=0,
                        current_task_id=None,
                    )
            except Exception as e:
                print(
                    f"TaskDispatcher: Error getting stats for worker {worker_id}: {e}"
                )
                import traceback

                traceback.print_exc()
                # Create basic worker object with defaults
                workers[worker_id] = Worker(
                    id=worker_id,
                    status="online",
                    tasks_completed=0,
                    tasks_failed=0,
                    current_task_id=None,
                )

        return workers

    # ==================== Statistics ====================

    def get_scheduler_name(self) -> str:
        """Get the name of the current scheduler"""
        return self.scheduler.__class__.__name__

    def change_scheduler(self, new_scheduler: TaskScheduler) -> None:
        """
        Change the scheduling algorithm

        Args:
            new_scheduler: New scheduler instance
        """
        old_name = self.get_scheduler_name()
        self.scheduler = new_scheduler
        new_name = self.get_scheduler_name()
        print(f"TaskDispatcher: Changed scheduler from {old_name} to {new_name}")

    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"TaskDispatcher(scheduler={self.get_scheduler_name()})"
