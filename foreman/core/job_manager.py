"""
Job lifecycle management and state tracking
"""

import json
from typing import List, Optional, Dict, Any, Tuple
from .payload_store import resolve_text_ref


from .utils import (
    _create_job_in_database,
    _create_tasks_for_job,
    _get_job_by_id,
    _get_job_tasks,
    _update_job_status,
    _update_task_status,
    _record_worker_failure,
    _complete_task_if_assigned,
)
from .dependency_manager import DependencyManager, PipelineStage


class JobManager:
    """
    Manages job lifecycle and state

    Responsibilities:
    - Create and track jobs
    - Manage task state transitions
    - Track job completion
    - Cache function code for jobs
    - Retrieve job results
    """

    def __init__(self):
        # Cache function code for active jobs: job_id -> func_code
        self._job_cache: Dict[str, str] = {}

        # Track job metadata: job_id -> metadata dict
        self._job_metadata: Dict[str, Dict[str, Any]] = {}

        # Track task metadata for checkpointing: job_id -> task_metadata dict
        self._task_metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Dependency manager for pipeline jobs
        self.dependency_manager = DependencyManager()

    # ==================== Job Creation ====================

    async def create_job(
        self,
        job_id: str,
        func_code: str,
        args_list: List[Any],
        total_tasks: int,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create a new job with tasks

        Args:
            job_id: Unique job identifier
            func_code: Function code to execute
            args_list: List of arguments for each task
            total_tasks: Total number of tasks
            task_metadata: Optional checkpoint configuration from @task decorator

        Raises:
            Exception: If job creation fails
        """
        # Extract checkpoint configuration
        checkpoint_enabled = False
        checkpoint_interval = 10.0
        checkpoint_state_vars = []

        if task_metadata:
            checkpoint_enabled = task_metadata.get("checkpoint_enabled", False)
            checkpoint_interval = task_metadata.get("checkpoint_interval", 10.0)
            checkpoint_state_vars = task_metadata.get("checkpoint_state", [])

        checkpoint_info = ""
        if checkpoint_enabled:
            vars_str = (
                ", ".join(checkpoint_state_vars) if checkpoint_state_vars else "all"
            )
            checkpoint_info = f" | Checkpoint: enabled (interval={checkpoint_interval}s, vars={vars_str})"

        print(
            f"JobManager: Creating job {job_id} with {total_tasks} tasks{checkpoint_info}"
        )

        # Store func_code in cache for quick access
        self._job_cache[job_id] = func_code

        # Store task metadata for checkpoint-aware dispatching
        if task_metadata:
            self._task_metadata_cache[job_id] = task_metadata

        # Store metadata
        self._job_metadata[job_id] = {
            "total_tasks": total_tasks,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "created_at": None,  # Will be set by database
            "checkpoint_enabled": checkpoint_enabled,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_state": checkpoint_state_vars,
        }

        # Create job in database (with checkpoint support flag)
        await _create_job_in_database(
            job_id, total_tasks, supports_checkpointing=checkpoint_enabled
        )

        # Create tasks in database
        await _create_tasks_for_job(job_id, args_list)

        # Update job status to running
        await _update_job_status(job_id, "running")

        print(f"JobManager: Job {job_id} created successfully")

    # ==================== Pipeline Job Creation ====================

    async def create_pipeline_job(
        self,
        job_id: str,
        stages: List[Dict[str, Any]],
        dependency_map: Optional[Dict[str, List[str]]] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a pipeline job with dependent stages.

        Works like create_job() but creates multiple stages where later
        stages are automatically blocked until their upstream dependencies
        complete.  Uses a dependency counter per task — when all upstream
        tasks finish, the counter reaches zero and the task becomes pending.

        The first stage's tasks start as "pending" (dispatchable immediately).
        All later stages start as "blocked".

        Args:
            job_id: Unique job identifier
            stages: List of stage dicts from the protocol message, each with:
                - func_code: Serialised function code
                - args_list: List of arguments (one per task)
                - pass_upstream_results: bool
                - task_metadata: Optional per-stage checkpoint config
                - name: Optional human label
            dependency_map: Optional explicit task→[deps] for arbitrary DAGs
            task_metadata: Optional global checkpoint config

        Returns:
            Total number of tasks created
        """
        # Build PipelineStage objects
        pipeline_stages = [PipelineStage.from_dict(s) for s in stages]

        total_tasks = sum(len(s.args_list) for s in pipeline_stages)
        total_stages = len(pipeline_stages)

        print(
            f"JobManager: Creating pipeline job {job_id} with "
            f"{total_stages} stages, {total_tasks} tasks"
        )

        # Cache the first stage's func_code as the "job-level" code
        # (for backward-compatible dispatch; per-stage codes are in DependencyManager)
        self._job_cache[job_id] = pipeline_stages[0].func_code

        # Store metadata
        checkpoint_enabled = False
        if task_metadata:
            checkpoint_enabled = task_metadata.get("checkpoint_enabled", False)
            self._task_metadata_cache[job_id] = task_metadata

        self._job_metadata[job_id] = {
            "total_tasks": total_tasks,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "is_pipeline": True,
            "total_stages": total_stages,
            "checkpoint_enabled": checkpoint_enabled,
        }

        # Create the job record in DB
        from foreman.db.base import db_session
        from foreman.db.crud import create_pipeline_job as db_create_pipeline_job

        async with db_session() as session:
            await db_create_pipeline_job(
                session,
                job_id,
                total_tasks,
                total_stages,
                supports_checkpointing=checkpoint_enabled,
            )

        # Create all tasks with dependency wiring via DependencyManager
        created = await self.dependency_manager.create_pipeline_tasks(
            job_id, pipeline_stages, dependency_map
        )

        # Set job to running
        await _update_job_status(job_id, "running")

        print(
            f"JobManager: Pipeline job {job_id} created successfully ({created} tasks)"
        )
        return created

    def get_stage_func_code(self, job_id: str, stage: int) -> Optional[str]:
        """Get function code for a specific pipeline stage.

        Args:
            job_id: Job identifier
            stage: Stage index (0-based)

        Returns:
            Function code string or None
        """
        return self.dependency_manager.get_stage_func_code(job_id, stage)

    def is_pipeline_job(self, job_id: str) -> bool:
        """Check if a job is a pipeline job."""
        meta = self._job_metadata.get(job_id, {})
        return meta.get("is_pipeline", False)

    def get_func_code(self, job_id: str) -> Optional[str]:
        """
        Get cached function code for a job

        Args:
            job_id: Job identifier

        Returns:
            Function code string or None if not found
        """
        return self._job_cache.get(job_id)

    def get_task_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached task metadata for a job

        Args:
            job_id: Job identifier

        Returns:
            Task metadata dictionary or None if not found
        """
        return self._task_metadata_cache.get(job_id)

    def has_job(self, job_id: str) -> bool:
        """
        Check if job exists in cache

        Args:
            job_id: Job identifier

        Returns:
            True if job exists
        """
        return job_id in self._job_cache

    # ==================== Task State Management ====================

    async def mark_task_completed(
        self, task_id: str, job_id: str, worker_id: str, result: Any
    ) -> Tuple[bool, bool]:
        """
        Mark a task as completed

        Args:
            task_id: Task identifier
            job_id: Job identifier
            worker_id: Worker that completed the task
            result: Task result

        Returns:
            Tuple of (accepted, job_complete)
        """
        print(f"JobManager: Marking task {task_id} as completed")

        # Serialize result to JSON string for storage
        result_str = json.dumps(result) if not isinstance(result, str) else result

        accepted, _, completed_count, total_tasks = await _complete_task_if_assigned(
            task_id, worker_id, result_str
        )

        if not accepted:
            print(
                f"JobManager: Task {task_id} completion rejected (stale or already completed)"
            )
            return False, False

        # Update local metadata if we still track it
        if job_id in self._job_metadata:
            self._job_metadata[job_id]["completed_tasks"] = (
                completed_count or self._job_metadata[job_id]["completed_tasks"]
            )

        is_complete = bool(
            total_tasks is not None
            and completed_count is not None
            and completed_count >= total_tasks
        )

        if is_complete:
            print(f"JobManager: Job {job_id} is now complete")

        return True, is_complete

    async def mark_task_failed(
        self, task_id: str, job_id: str, worker_id: str, error: str
    ) -> None:
        """
        Mark a task as failed and reset to pending for retry

        Args:
            task_id: Task identifier
            job_id: Job identifier
            worker_id: Worker that failed the task
            error: Error message
        """
        print(f"JobManager: Marking task {task_id} as failed: {error}")

        await _record_worker_failure(worker_id, task_id, error, job_id)

        # Update task status back to pending for retry
        # Set worker_id to None so it can be reassigned
        await _update_task_status(task_id, "pending", error=error)

        # Update local metadata
        if job_id in self._job_metadata:
            self._job_metadata[job_id]["failed_tasks"] += 1

        print(f"JobManager: Task {task_id} reset to pending for retry")

    async def mark_task_assigned(self, task_id: str, worker_id: str) -> None:
        """
        Mark a task as assigned to a worker

        Args:
            task_id: Task identifier
            worker_id: Worker assigned to the task
        """
        await _update_task_status(task_id, "assigned")
        # Increment total_tasks_assigned for this worker
        from foreman.db.base import async_session
        from foreman.db.crud import update_worker_tasks_assigned

        async with async_session() as session:
            await update_worker_tasks_assigned(session, worker_id)

    # ==================== Job Completion ====================

    async def is_job_complete(self, job_id: str) -> bool:
        """
        Check if all tasks in a job are completed

        Args:
            job_id: Job identifier

        Returns:
            True if all tasks are completed
        """
        tasks = await _get_job_tasks(job_id)

        if not tasks:
            return False

        completed_tasks = [t for t in tasks if t.status == "completed"]
        is_complete = len(completed_tasks) == len(tasks)

        return is_complete

    async def get_job_results(self, job_id: str) -> Optional[List[Any]]:
        """
        Get ordered results for a completed job.

        For pipeline jobs, returns results from the FINAL stage only
        (matching the semantics: the pipeline produces the final stage output).

        Args:
            job_id: Job identifier

        Returns:
            List of results in task order, or None if job not complete
        """
        if not await self.is_job_complete(job_id):
            return None

        tasks = await _get_job_tasks(job_id)

        if not tasks:
            return []

        # For pipeline jobs, only return results from the last stage
        if self.is_pipeline_job(job_id):
            max_stage = max((getattr(t, "stage", 0) or 0) for t in tasks)
            final_tasks = sorted(
                [t for t in tasks if (getattr(t, "stage", 0) or 0) == max_stage],
                key=lambda t: t.id,
            )
            results = []
            for task in final_tasks:
                if task.status == "completed":
                    raw_result = resolve_text_ref(task.result)
                    try:
                        results.append(json.loads(raw_result))
                    except (json.JSONDecodeError, TypeError):
                        results.append(raw_result)
                else:
                    results.append(None)
            return results

        # Regular (non-pipeline) job — original logic
        results = []
        for i in range(len(tasks)):
            task = None
            for t in tasks:
                if t.id == f"{job_id}_task_{i}":
                    task = t
                    break

            if task and task.status == "completed":
                raw_result = resolve_text_ref(task.result)
                try:
                    results.append(json.loads(raw_result))
                except (json.JSONDecodeError, TypeError):
                    results.append(raw_result)
            else:
                results.append(None)

        return results

    async def finalize_job(self, job_id: str, completed_tasks: int = None) -> None:
        """
        Mark job as completed and clean up

        Args:
            job_id: Job identifier
            completed_tasks: Number of completed tasks (optional, only set if provided)
        """
        print(f"JobManager: Finalizing job {job_id}")

        # Update job status in database
        await _update_job_status(job_id, "completed", completed_tasks=completed_tasks)

        # Clean up cache
        if job_id in self._job_cache:
            del self._job_cache[job_id]

        # Clean up metadata
        if job_id in self._job_metadata:
            del self._job_metadata[job_id]

        # Clean up pipeline data
        self.dependency_manager.cleanup_job(job_id)

        print(f"JobManager: Job {job_id} finalized")

    # ==================== Job Information ====================

    async def get_job_info(self, job_id: str) -> Optional[Any]:
        """
        Get job information from database

        Args:
            job_id: Job identifier

        Returns:
            Job object or None if not found
        """
        return await _get_job_by_id(job_id)

    async def get_job_tasks_info(self, job_id: str) -> List[Any]:
        """
        Get all tasks for a job

        Args:
            job_id: Job identifier

        Returns:
            List of task objects
        """
        return await _get_job_tasks(job_id)

    async def get_job_progress(self, job_id: str) -> Tuple[int, int]:
        """
        Get job progress (completed tasks, total tasks)

        Args:
            job_id: Job identifier

        Returns:
            Tuple of (completed_tasks, total_tasks)
        """
        tasks = await _get_job_tasks(job_id)
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.status == "completed"])

        return completed_tasks, total_tasks

    def get_cached_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached job metadata without database query

        Args:
            job_id: Job identifier

        Returns:
            Metadata dictionary or None
        """
        return self._job_metadata.get(job_id)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get job manager statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "active_jobs": len(self._job_cache),
            "cached_jobs": len(self._job_metadata),
        }

    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"JobManager(active_jobs={len(self._job_cache)})"
