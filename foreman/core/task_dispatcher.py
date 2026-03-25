"""
Task dispatching with pluggable scheduling algorithms

Extended with mobile worker support:
- Detects worker capabilities (sys.settrace support)
- Instruments code for workers without trace support (Chaquopy)
- Transparent to developers - they write pure logic
"""

import asyncio
import json
from typing import List, Optional, Any, Dict

from .scheduling import TaskScheduler, Task as SchedulerTask, Worker
from .scheduling.model_affinity_scheduler import ModelAffinityScheduler
from .payload_store import resolve_text_ref
from .utils import (
    _get_pending_tasks,
    _get_assigned_tasks,
    _update_task_status,
    _update_worker_status,
    _claim_task_for_worker,
    _get_latest_task_failure_with_checkpoint,
)
from common.protocol import (
    create_assign_task_message,
    create_resume_task_message,
    create_assign_task_message_with_metadata,
    create_resume_task_message_with_metadata,
)
from common.serializer import get_runtime_info
from common.code_instrumenter import (
    instrument_for_mobile,
    prepare_code_for_mobile_resume,
    generate_mobile_checkpoint_wrapper,
)


class TaskDispatcher:
    """
    Dispatches tasks to workers using pluggable scheduling algorithms

    Responsibilities:
    - Use scheduler to select best worker for task
    - Use scheduler to select best task for worker
    - Send task assignments to workers
    - Update task and worker status
    """

    def __init__(
        self,
        scheduler: TaskScheduler,
        connection_manager,
        job_manager,
        model_load_tracker=None,
    ):
        """
        Initialize task dispatcher

        Args:
            scheduler: Scheduling algorithm to use
            connection_manager: ConnectionManager instance
            job_manager: JobManager instance
            model_load_tracker: Optional tracker for LOAD_MODEL readiness gating
        """
        self.scheduler = scheduler
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.model_load_tracker = model_load_tracker
        self._assignment_retry_tasks: Dict[str, asyncio.Task] = {}

        # Wrap the base scheduler with model-affinity awareness for DNN tasks
        if model_load_tracker:
            self._affinity_scheduler = ModelAffinityScheduler(
                base_scheduler=scheduler,
                model_load_tracker=model_load_tracker,
            )
        else:
            self._affinity_scheduler = None

    # ==================== Code Preparation for Mobile Workers ====================

    def _prepare_code_for_worker(
        self,
        func_code: str,
        worker_id: str,
        task_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_state: Optional[Dict[str, Any]] = None,
        is_resume: bool = False,
    ) -> str:
        """
        Prepare function code based on worker capabilities.

        For workers that don't support sys.settrace() (e.g., Chaquopy on Android),
        this method instruments the code with explicit checkpoint calls.

        This is TRANSPARENT to developers - they write pure logic, and the framework
        automatically prepares the code for each worker type.

        Args:
            func_code: Original function source code
            worker_id: Target worker identifier
            task_metadata: Checkpoint configuration from @task decorator
            checkpoint_state: Checkpoint state for resumed tasks
            is_resume: Whether this is a resumed task

        Returns:
            Prepared code (possibly instrumented for mobile workers)
        """
        # Check if worker needs code instrumentation
        if not self.connection_manager.needs_code_instrumentation(worker_id):
            # Worker supports sys.settrace() - no instrumentation needed
            return func_code

        # Worker needs instrumentation (e.g., Chaquopy)
        print(
            f"TaskDispatcher: Preparing instrumented code for mobile worker {worker_id}"
        )

        # Get checkpoint state variables from metadata
        checkpoint_state_vars = []
        if task_metadata:
            checkpoint_state_vars = task_metadata.get("checkpoint_state", [])

        if not checkpoint_state_vars:
            # No checkpoint variables declared - can't instrument effectively
            print(
                f"TaskDispatcher: No checkpoint_state vars declared, skipping instrumentation"
            )
            return func_code

        try:
            if is_resume and checkpoint_state:
                # Resumed task - apply both resume transformation and instrumentation
                prepared_code = prepare_code_for_mobile_resume(
                    func_code=func_code,
                    checkpoint_state=checkpoint_state,
                    checkpoint_state_vars=checkpoint_state_vars,
                )
            else:
                # Fresh task - just instrument for checkpoint capture
                prepared_code, num_loops = instrument_for_mobile(
                    func_code=func_code, checkpoint_state_vars=checkpoint_state_vars
                )
                print(f"TaskDispatcher: Instrumented {num_loops} loops for mobile worker")
            
            # Prepend the mobile checkpoint wrapper only if not already present.
            # SDK-side runtime wrapping may already include checkpoint support.
            if "_MobileCheckpointState" not in prepared_code:
                wrapper = generate_mobile_checkpoint_wrapper()
                prepared_code = wrapper + "\n" + prepared_code
            return prepared_code

        except Exception as e:
            print(
                f"TaskDispatcher: Code instrumentation failed: {e}, using original code"
            )
            import traceback

            traceback.print_exc()
            return func_code

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
        return len(await self.recover_orphaned_task_ids())

    async def recover_orphaned_task_ids(self) -> List[str]:
        """Recover orphaned tasks and return the recovered task IDs."""
        connected_workers = self.connection_manager.get_all_worker_ids()
        assigned_tasks = await _get_assigned_tasks()

        if not assigned_tasks:
            return []

        recovered_task_ids: List[str] = []
        for task in assigned_tasks:
            if task.worker_id and task.worker_id not in connected_workers:
                await _update_task_status(task.id, "pending", worker_id=None)
                recovered_task_ids.append(task.id)
                print(
                    f"TaskDispatcher: 🔄 Recovered orphaned task {task.id} "
                    f"from disconnected worker {task.worker_id}"
                )

        if recovered_task_ids:
            print(
                f"TaskDispatcher: Recovered {len(recovered_task_ids)} orphaned tasks for reassignment"
            )

        return recovered_task_ids

    def _get_native_model_partition_id(
        self, execution_metadata: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Return required partition id when task runs in native model mode."""
        if not execution_metadata:
            return None

        execution_mode = str(execution_metadata.get("execution_mode", "")).lower()
        if execution_mode != "native_model":
            return None

        partition_id = execution_metadata.get("model_partition_id")
        if not partition_id:
            return None
        return str(partition_id)

    def _seconds_until_native_model_ready(
        self, worker_id: str, model_partition_id: str
    ) -> float:
        """Return remaining wait time before assigning a native model task."""
        if not self.model_load_tracker or not model_partition_id:
            return 0.0

        remaining = self.model_load_tracker.seconds_until_ready(
            worker_id, model_partition_id
        )
        if remaining is None:
            return 0.0
        return max(0.0, float(remaining))

    def schedule_job_assignment_retry(
        self, job_id: str, delay_seconds: float = 1.0
    ) -> None:
        """Schedule a one-shot delayed assignment attempt for a job."""
        if not job_id:
            return

        delay = max(0.1, float(delay_seconds))
        existing = self._assignment_retry_tasks.get(job_id)
        if existing and not existing.done():
            return

        async def _retry() -> None:
            try:
                await asyncio.sleep(delay)
                func_code = self.job_manager.get_func_code(job_id)
                if not func_code:
                    return
                await self.assign_tasks_for_job(job_id, func_code, [])
            except Exception as exc:
                print(
                    f"TaskDispatcher: Deferred assignment retry failed for job {job_id}: {exc}"
                )
            finally:
                self._assignment_retry_tasks.pop(job_id, None)

        self._assignment_retry_tasks[job_id] = asyncio.create_task(_retry())

    # ==================== Task Assignment ====================

    async def assign_tasks_for_job(
        self, job_id: str, func_code: str, args_list: List[Any]
    ) -> int:
        """
        Assign available tasks for a specific job to available workers

        Uses batch_select_workers for optimized assignment:
        - If tasks >= workers: Skip ranking, assign all workers directly
        - If tasks < workers: Rank once and pick top N workers

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

        available_workers = self.connection_manager.get_available_workers()

        if not available_workers:
            print(
                f"TaskDispatcher: No available workers, {len(pending_tasks)} tasks remain"
            )
            return 0

        # Get worker objects for scheduler
        all_workers = await self._get_all_workers()

        # Convert pending tasks to scheduler tasks
        scheduler_tasks = []
        has_model_tasks = False
        for task in pending_tasks:
            st = SchedulerTask(
                id=task.id,
                job_id=task.job_id,
                args=task.args,
                priority=getattr(task, "priority", 0),
                stage_func_code=getattr(task, "stage_func_code", None),
            )
            # Attach model_partition_id so the affinity scheduler can use it
            partition_id = getattr(task, "model_partition_id", None)
            if partition_id:
                st.model_partition_id = partition_id  # type: ignore[attr-defined]
                has_model_tasks = True
            scheduler_tasks.append(st)

        # Use affinity scheduler for DNN tasks when available
        active_scheduler = (
            self._affinity_scheduler
            if has_model_tasks and self._affinity_scheduler
            else self.scheduler
        )

        # Use batch assignment - ranks workers ONCE instead of per-task
        assignments = await active_scheduler.batch_select_workers(
            scheduler_tasks, available_workers, all_workers
        )

        print(
            f"TaskDispatcher: Batch scheduler returned {len(assignments)} assignments"
        )

        tasks_assigned = 0
        used_workers = set()
        for scheduler_task, worker_id in assignments:
            task_db = next(
                (t for t in pending_tasks if t.id == scheduler_task.id), None
            )
            requirements = self._parse_device_requirements(task_db)

            if worker_id in used_workers or not self._worker_meets_requirements(
                all_workers.get(worker_id), requirements
            ):
                worker_id = self._find_requirement_compatible_worker(
                    all_workers=all_workers,
                    available_workers=available_workers,
                    used_workers=used_workers,
                    requirements=requirements,
                )
                if not worker_id:
                    print(
                        f"TaskDispatcher: No requirement-compatible worker for task {scheduler_task.id}"
                    )
                    continue

            # Gate DNN tasks: only assign to workers with the required model loaded
            partition_id = getattr(scheduler_task, "model_partition_id", None)
            if partition_id and self.model_load_tracker:
                if not self.model_load_tracker.is_model_resident(
                    worker_id, partition_id
                ):
                    # Try to find an alternative worker that has the model loaded
                    resident_workers = (
                        self.model_load_tracker.workers_with_model_resident(
                            partition_id
                        )
                    )
                    alt_worker = next(
                        (
                            w
                            for w in resident_workers
                            if w in available_workers and w not in used_workers
                        ),
                        None,
                    )
                    if alt_worker:
                        worker_id = alt_worker
                    else:
                        print(
                            f"TaskDispatcher: Skipping task {scheduler_task.id} — "
                            f"no worker has model {partition_id} loaded"
                        )
                        continue

            args_text = resolve_text_ref(scheduler_task.args)
            task_args = json.loads(args_text) if args_text else []

            # For pipeline tasks, use per-stage func_code stored on SchedulerTask
            task_func_code = (
                getattr(scheduler_task, "stage_func_code", None) or func_code
            )
            execution_metadata = self._build_execution_metadata(task_db)

            success = await self._assign_task_to_worker(
                job_id,
                scheduler_task.id,
                task_func_code,
                task_args,
                worker_id,
                execution_metadata=execution_metadata,
            )
            if success:
                tasks_assigned += 1
                used_workers.add(worker_id)

        print(f"TaskDispatcher: Assigned {tasks_assigned} tasks for job {job_id}")
        return tasks_assigned

    def _parse_device_requirements(self, task_obj) -> Dict[str, Any]:
        if not task_obj:
            return {}
        req_text = getattr(task_obj, "device_requirements", None)
        if not req_text:
            return {}
        try:
            return json.loads(req_text)
        except Exception:
            return {}

    def _parse_json_text_field(self, value: Optional[str], default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def _build_execution_metadata(self, task_obj) -> Optional[Dict[str, Any]]:
        """Build optional execution hints for runtimes that execute native models."""
        if not task_obj:
            return None

        metadata: Dict[str, Any] = {}
        model_partition_id = getattr(task_obj, "model_partition_id", None)
        if model_partition_id:
            metadata["execution_mode"] = "native_model"
            metadata["model_partition_id"] = model_partition_id

        topology_role = getattr(task_obj, "topology_role", None)
        if topology_role:
            metadata["topology_role"] = topology_role

        feature_sources = self._parse_json_text_field(
            getattr(task_obj, "feature_sources", None), []
        )
        if feature_sources:
            metadata["feature_sources"] = feature_sources

        feature_targets = self._parse_json_text_field(
            getattr(task_obj, "feature_targets", None), []
        )
        if feature_targets:
            metadata["feature_targets"] = feature_targets

        requirements = self._parse_device_requirements(task_obj)
        if requirements:
            metadata["device_requirements"] = requirements

        return metadata or None

    def _worker_meets_requirements(
        self, worker: Optional[Worker], requirements: Dict[str, Any]
    ) -> bool:
        if not worker or not requirements:
            return True

        min_ram_mb = requirements.get("min_ram_mb")
        if min_ram_mb is not None and (worker.ram_available_mb or 0.0) < float(
            min_ram_mb
        ):
            return False

        min_battery = requirements.get("min_battery")
        if min_battery is not None and (worker.battery_level or 0.0) < float(
            min_battery
        ):
            return False

        if requirements.get("require_gpu") and not worker.gpu_available:
            return False

        allowed_os_types = requirements.get("allowed_os_types")
        if allowed_os_types and worker.os_type not in allowed_os_types:
            return False

        allowed_runtimes = requirements.get("allowed_runtimes")
        if allowed_runtimes and worker.runtime not in allowed_runtimes:
            return False

        allowed_model_runtimes = requirements.get("allowed_model_runtimes")
        if (
            allowed_model_runtimes
            and worker.model_runtime not in allowed_model_runtimes
        ):
            return False

        return True

    def _find_requirement_compatible_worker(
        self,
        all_workers: Dict[str, Worker],
        available_workers,
        used_workers,
        requirements: Dict[str, Any],
    ) -> Optional[str]:
        for candidate_id in available_workers:
            if candidate_id in used_workers:
                continue
            worker = all_workers.get(candidate_id)
            if self._worker_meets_requirements(worker, requirements):
                return candidate_id
        return None

    async def assign_task_to_available_worker(
        self, worker_id: str, worker_threshold: int = 1
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
            print(
                f"TaskDispatcher: Recovered {recovered} orphaned tasks before assignment"
            )

        # Check how many workers are currently available
        available_workers = self.connection_manager.get_available_workers()
        num_available = len(available_workers)

        print(
            f"TaskDispatcher: Worker {worker_id} available. "
            f"Total available workers: {num_available}, threshold: {worker_threshold}"
        )

        # If not enough workers available, but at least one worker is present,
        # perform a per-worker assignment (assign one pending task to the worker that became available).
        if num_available < worker_threshold:
            if num_available == 0:
                print(
                    f"TaskDispatcher: Waiting for more workers "
                    f"({num_available}/{worker_threshold} available)"
                )
                return False

            # One or more workers available but below threshold - assign a single task to the available worker
            print(
                f"TaskDispatcher: Fewer workers than threshold ({num_available}/{worker_threshold}), "
                f"attempting single-worker assignment to {worker_id}"
            )

            # Get the earliest pending task
            pending_tasks = await _get_pending_tasks()
            if not pending_tasks:
                print(
                    f"TaskDispatcher: No pending tasks available for single-worker assignment"
                )
                return False

            # Find the first pending task this worker can handle (model gate)
            task = None
            for candidate in pending_tasks:
                partition_id = getattr(candidate, "model_partition_id", None)
                if partition_id and self.model_load_tracker:
                    if not self.model_load_tracker.is_model_resident(
                        worker_id, partition_id
                    ):
                        continue
                task = candidate
                break

            if task is None:
                print(
                    f"TaskDispatcher: No pending task compatible with worker "
                    f"{worker_id}'s loaded model; skipping single-worker assignment"
                )
                return False

            job_id = task.job_id

            # For pipeline tasks, use per-stage func_code if available
            func_code = self._get_func_code_for_task(task)

            if not func_code:
                print(f"TaskDispatcher: No func_code found for job {job_id}, skipping")
                return False

            args_text = resolve_text_ref(task.args)
            task_args = json.loads(args_text) if args_text else []
            execution_metadata = self._build_execution_metadata(task)
            success = await self._assign_task_to_worker(
                job_id,
                task.id,
                func_code,
                task_args,
                worker_id,
                execution_metadata=execution_metadata,
            )
            return success

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

        print(
            f"TaskDispatcher: Batch assignment completed, {total_assigned} tasks assigned"
        )
        return total_assigned > 0

    async def _assign_task_to_worker(
        self,
        job_id: str,
        task_id: str,
        func_code: str,
        task_args: Any,
        worker_id: str,
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Assign a specific task to a specific worker.

        If the task has checkpoint data from a previous failure, sends a RESUME_TASK
        message so the worker can continue from where the last worker left off.
        Otherwise, sends a regular ASSIGN_TASK message to start fresh.

        Includes task_metadata for declarative checkpointing if configured.

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
            # Get task_metadata from job_manager for declarative checkpointing
            task_metadata = self.job_manager.get_task_metadata(job_id)

            # Native runtimes may need time to finish LOAD_MODEL before task execution.
            required_partition_id = self._get_native_model_partition_id(
                execution_metadata
            )
            if required_partition_id:
                wait_seconds = self._seconds_until_native_model_ready(
                    worker_id, required_partition_id
                )
                if wait_seconds > 0.0:
                    print(
                        f"TaskDispatcher: Deferring task {task_id} for worker {worker_id} "
                        f"until model partition {required_partition_id} is ready "
                        f"(wait {wait_seconds:.2f}s)"
                    )
                    self.schedule_job_assignment_retry(
                        job_id, delay_seconds=wait_seconds
                    )
                    return False

            # Check if we have checkpoint data for this task from a previous failure
            checkpoint_data = await _get_latest_task_failure_with_checkpoint(task_id)

            # Prepare code for the target worker (handles mobile instrumentation)
            checkpoint_state = None
            is_resume = False

            if checkpoint_data and checkpoint_data.get("state"):
                is_resume = True
                checkpoint_state = checkpoint_data.get("state", {})

            # Prepare code based on worker capabilities
            prepared_code = self._prepare_code_for_worker(
                func_code=func_code,
                worker_id=worker_id,
                task_metadata=task_metadata,
                checkpoint_state=checkpoint_state,
                is_resume=is_resume,
            )

            if checkpoint_data and checkpoint_data.get("state"):
                # We have checkpoint data - send RESUME_TASK message
                if task_metadata:
                    # Use extended message with task_metadata
                    message = create_resume_task_message_with_metadata(
                        task_id=task_id,
                        job_id=job_id,
                        func_code=prepared_code,  # Use prepared code
                        checkpoint_state=checkpoint_data.get("state", {}),
                        task_args=(
                            [task_args]
                            if not isinstance(task_args, list)
                            else task_args
                        ),
                        task_kwargs={},
                        progress_percent=checkpoint_data.get("progress_percent", 0),
                        checkpoint_count=checkpoint_data.get("checkpoint_count", 0),
                        task_metadata=task_metadata,
                        execution_metadata=execution_metadata,
                    )
                else:
                    # Use standard message
                    message = create_resume_task_message(
                        task_id=task_id,
                        job_id=job_id,
                        func_code=prepared_code,  # Use prepared code
                        checkpoint_state=checkpoint_data.get("state", {}),
                        task_args=(
                            [task_args]
                            if not isinstance(task_args, list)
                            else task_args
                        ),
                        task_kwargs={},
                        progress_percent=checkpoint_data.get("progress_percent", 0),
                        checkpoint_count=checkpoint_data.get("checkpoint_count", 0),
                        execution_metadata=execution_metadata,
                    )
                print(
                    f"TaskDispatcher: 🔄 Resuming task {task_id} from checkpoint "
                    f"(progress: {checkpoint_data.get('progress_percent', 0):.1f}%)"
                )
            else:
                # No checkpoint - send regular ASSIGN_TASK message
                if task_metadata:
                    # Use extended message with task_metadata
                    message = create_assign_task_message_with_metadata(
                        func_code=prepared_code,  # Use prepared code
                        task_args=[task_args],  # Wrap in list for single task
                        task_id=task_id,
                        job_id=job_id,
                        task_metadata=task_metadata,
                        execution_metadata=execution_metadata,
                    )
                else:
                    # Use standard message
                    message = create_assign_task_message(
                        prepared_code,
                        [task_args],
                        task_id,
                        job_id,
                        execution_metadata=execution_metadata,
                    )

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

            # Increment total_tasks_assigned for this worker
            from foreman.db.base import async_session
            from foreman.db.crud import update_worker_tasks_assigned
            async with async_session() as session:
                await update_worker_tasks_assigned(session, worker_id)

            # Send to worker
            await websocket.send(message.to_json())

            # Mark worker as busy
            self.connection_manager.set_worker_active_task(worker_id, task_id, job_id)
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
            self.connection_manager.clear_worker_active_task(worker_id)
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
                        runtime=stats.get("runtime"),
                        model_runtime=stats.get("model_runtime"),
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

    def _get_func_code_for_task(self, task) -> Optional[str]:
        """
        Get the correct function code for a task.

        For pipeline tasks, each task may have its own stage_func_code
        (stored on the DB row).  Falls back to the job-level func_code.

        Args:
            task: TaskModel instance (DB row)

        Returns:
            Function code string or None
        """
        # Try per-task stage func_code first (pipeline tasks)
        stage_func = getattr(task, "stage_func_code", None)
        if stage_func:
            return stage_func

        # Fall back to job-level func_code
        job_id = task.job_id
        return self.job_manager.get_func_code(job_id)

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
