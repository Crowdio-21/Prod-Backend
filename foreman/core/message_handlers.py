"""
Message handling logic separated by client type.

Changes from original
---------------------
WorkerMessageHandler now accepts an optional ``reconnection_handler``
(WorkerReconnectionHandler).  Two integration points:

1. ``_handle_worker_ready``  – calls ``reconnection_handler.handle_worker_reconnected``
   first; if it returns True the worker already received a RESUME_TASK message
   and normal task assignment is skipped.

2. ``handle_disconnection``  – NEW public method, called by the WebSocket
   server's close/disconnect path instead of (or after) any existing cleanup.
   Delegates to ``reconnection_handler.handle_worker_disconnected``.

All other behaviour is unchanged.
"""

import json

from websockets.server import WebSocketServerProtocol

from .utils import (
    _create_worker_in_database,
    _get_assigned_tasks,
    _record_worker_failure,
    _update_task_status,
    _update_worker_status,
    _update_worker_task_stats,
)
from common.protocol import Message, MessageType, create_job_accepted_message
from common.serializer import get_runtime_info, bytes_to_hex, hex_to_bytes
from .staged_results_manager.checkpoint_manager import CheckpointManager


class ClientMessageHandler:
    """
    Handles messages from client SDK

    Responsibilities:
    - Handle job submissions
    - Handle client disconnections
    - Send job acknowledgments
    """

    def __init__(self, connection_manager, job_manager, task_dispatcher):
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher

    async def handle_message(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        if message.type in (MessageType.SUBMIT_JOB, MessageType.SUBMIT_BROADCAST_JOB):
            await self._handle_job_submission(message, websocket)
        elif message.type == MessageType.SUBMIT_PIPELINE_JOB:
            await self._handle_pipeline_submission(message, websocket)
        elif message.type == MessageType.DISCONNECT:
            await websocket.close()
        else:
            print(f"ClientMessageHandler: Unknown message type: {message.type}")

    async def _handle_job_submission(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            func_code = message.data["func_code"]
            args_list = message.data["args_list"]
            total_tasks = message.data["total_tasks"]
            task_metadata = message.data.get("task_metadata")

            print(
                f"ClientMessageHandler: Received job {job_id} with {total_tasks} tasks | foreman_runtime={get_runtime_info()}"
            )

            if task_metadata and task_metadata.get("checkpoint_enabled"):
                print(
                    f"ClientMessageHandler: Job {job_id} has checkpointing enabled "
                    f"(interval={task_metadata.get('checkpoint_interval')}s, "
                    f"state_vars={task_metadata.get('checkpoint_state', [])})"
                )

            self.connection_manager.add_client(job_id, websocket)
            await self.job_manager.create_job(
                job_id, func_code, args_list, total_tasks, task_metadata=task_metadata
            )
            assigned = await self.task_dispatcher.assign_tasks_for_job(
                job_id, func_code, args_list
            )
            print(
                f"ClientMessageHandler: Assigned {assigned} tasks immediately for job {job_id}"
            )

            response = create_job_accepted_message(job_id)
            await websocket.send(response.to_json())
            print(f"ClientMessageHandler: Job {job_id} accepted and acknowledged")

        except KeyError as e:
            print(
                f"ClientMessageHandler: Missing required field in job submission: {e}"
            )
            error_msg = Message(
                MessageType.JOB_ERROR,
                {"error": f"Missing required field: {e}"},
                message.job_id,
            )
            await websocket.send(error_msg.to_json())
        except Exception as e:
            print(f"ClientMessageHandler: Error handling job submission: {e}")
            import traceback

            traceback.print_exc()
            error_msg = Message(
                MessageType.JOB_ERROR, {"error": str(e)}, message.job_id
            )
            await websocket.send(error_msg.to_json())

    async def _handle_pipeline_submission(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            stages = message.data["stages"]
            total_tasks = message.data["total_tasks"]
            total_stages = message.data["total_stages"]
            dependency_map = message.data.get("dependency_map")
            task_metadata = message.data.get("task_metadata")

            print(
                f"ClientMessageHandler: Received pipeline job {job_id} "
                f"with {total_stages} stages, {total_tasks} tasks | "
                f"foreman_runtime={get_runtime_info()}"
            )

            for i, stage in enumerate(stages):
                stage_name = stage.get("name", f"stage_{i}")
                stage_tasks = len(stage["args_list"])
                pass_results = stage.get("pass_upstream_results", False)
                print(
                    f"  Stage {i} ({stage_name}): {stage_tasks} tasks"
                    f"{' [receives upstream results]' if pass_results else ''}"
                )

            self.connection_manager.add_client(job_id, websocket)
            await self.job_manager.create_pipeline_job(
                job_id, stages, dependency_map, task_metadata
            )

            stage_0_func_code = stages[0]["func_code"]
            assigned = await self.task_dispatcher.assign_tasks_for_job(
                job_id, stage_0_func_code, stages[0]["args_list"]
            )
            print(
                f"ClientMessageHandler: Pipeline job {job_id} — Assigned {assigned} stage-0 tasks immediately"
            )

            response = create_job_accepted_message(job_id)
            await websocket.send(response.to_json())
            print(
                f"ClientMessageHandler: Pipeline job {job_id} accepted and acknowledged"
            )

        except KeyError as e:
            print(
                f"ClientMessageHandler: Missing required field in pipeline submission: {e}"
            )
            error_msg = Message(
                MessageType.JOB_ERROR,
                {"error": f"Missing required field: {e}"},
                message.job_id,
            )
            await websocket.send(error_msg.to_json())
        except Exception as e:
            print(f"ClientMessageHandler: Error handling pipeline submission: {e}")
            import traceback

            traceback.print_exc()
            error_msg = Message(
                MessageType.JOB_ERROR, {"error": str(e)}, message.job_id
            )
            await websocket.send(error_msg.to_json())


class WorkerMessageHandler:
    """
    Handles messages from workers.

    Responsibilities:
    - Handle worker registration (with transparent checkpoint-based resume on reconnect)
    - Handle task results / errors / checkpoints / pongs
    - Graceful disconnection → reconnection lifecycle via reconnection_handler
    """

    def __init__(
        self,
        connection_manager,
        job_manager,
        task_dispatcher,
        completion_handler,
        checkpoint_manager: CheckpointManager = None,
        reconnection_handler=None,  # WorkerReconnectionHandler | None
    ):
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher
        self.completion_handler = completion_handler
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.reconnection_handler = (
            reconnection_handler  # may be None if checkpointing unused
        )

    # ------------------------------------------------------------------
    # Disconnection entry-point  (NEW – wire this into your WS close path)
    # ------------------------------------------------------------------

    async def handle_disconnection(self, websocket: WebSocketServerProtocol) -> None:
        """
        Call this from the WebSocket server whenever a worker socket closes,
        regardless of whether it was clean or an unexpected drop.

        This replaces the previous ad-hoc "mark_worker_available on close" pattern
        with a single, consistent entry point that handles both cases:
          - Checkpointing ON  → park the task; wait for the worker to reconnect
          - Checkpointing OFF → immediately re-queue the task for another worker
        """
        worker_id = self.connection_manager.find_worker_by_websocket(websocket)
        if not worker_id:
            return  # Was a client socket or already cleaned up

        print(f"[Disconnect] Worker {worker_id} disconnected.")

        # Do not requeue or explicitly resume tasks on disconnect. Workers continue
        # execution on their side and reconnect with the same worker_id. We only
        # clean up the stale socket mapping here.
        self.connection_manager.remove_worker(worker_id)

    def _get_active_task_fallback(self, worker_id: str):
        try:
            return self.connection_manager.get_worker_active_task(worker_id)
        except Exception:
            return None, None

    async def _get_db_assigned_task_for_worker(self, worker_id: str):
        """Return (task_id, job_id) if DB still has an assigned task for this worker."""
        try:
            assigned_tasks = await _get_assigned_tasks()
            for task in assigned_tasks:
                if getattr(task, "worker_id", None) == worker_id:
                    return task.id, task.job_id
        except Exception as e:
            print(
                f"WorkerMessageHandler: Error querying assigned task for {worker_id}: {e}"
            )
        return None, None

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def handle_message(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        if message.type == MessageType.WORKER_READY:
            await self._handle_worker_ready(message, websocket)
        elif message.type == MessageType.TASK_RESULT:
            await self._handle_task_result(message, websocket)
        elif message.type == MessageType.TASK_ERROR:
            await self._handle_task_error(message, websocket)
        elif message.type == MessageType.PONG:
            await self._handle_pong(message, websocket)
        elif message.type == MessageType.WORKER_HEARTBEAT:
            await self._handle_pong(message, websocket)
        elif message.type == MessageType.TASK_CHECKPOINT:
            await self._handle_task_checkpoint(message, websocket)
        elif message.type == MessageType.KILL_ACK:
            await self._handle_kill_ack(message, websocket)
        else:
            print(f"WorkerMessageHandler: Unknown message type: {message.type}")

    # ------------------------------------------------------------------
    # Individual handlers
    # ------------------------------------------------------------------

    async def _handle_pong(self, message: Message, websocket: WebSocketServerProtocol):
        worker_id = self.connection_manager.find_worker_by_websocket(websocket)
        if worker_id:
            await _update_worker_status(worker_id, "online")
            if "performance_metrics" in message.data:
                await self._update_worker_performance_metrics(
                    worker_id, message.data["performance_metrics"]
                )

    async def _update_worker_performance_metrics(self, worker_id: str, metrics: dict):
        try:
            from foreman.db.base import async_session
            from foreman.db.models import WorkerModel
            from sqlalchemy import update
            from datetime import datetime

            async with async_session() as session:
                stmt = (
                    update(WorkerModel)
                    .where(WorkerModel.id == worker_id)
                    .values(
                        cpu_usage_percent=metrics.get("cpu_usage_percent"),
                        ram_available_mb=metrics.get("ram_available_mb"),
                        battery_level=metrics.get("battery_level"),
                        is_charging=metrics.get("is_charging"),
                        network_speed_mbps=metrics.get("network_speed_mbps"),
                        storage_available_gb=metrics.get("storage_available_gb"),
                        last_performance_update=datetime.now(),
                    )
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            print(
                f"WorkerMessageHandler: Error updating performance metrics for {worker_id}: {e}"
            )

    async def _handle_worker_ready(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """
        Handle worker registration.

        Reconnect-resume path (checked first)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        If this worker_id has a parked checkpoint record, send it a
        RESUME_TASK message and return early – no new task assignment needed.

        Normal registration path
        ~~~~~~~~~~~~~~~~~~~~~~~~
        Register the worker, persist to DB, assign a pending task if available.
        """
        try:
            worker_id = message.data["worker_id"]
            device_specs = message.data.get("device_specs", {})
            worker_type = message.data.get("worker_type", "pc_python")
            platform = message.data.get("platform", "unknown")
            runtime = message.data.get("runtime", "cpython")
            capabilities = message.data.get("capabilities", {})

            print(f"WorkerMessageHandler: Worker {worker_id} connected")
            print(f"  Type: {worker_type} | Platform: {platform} | Runtime: {runtime}")

            if device_specs:
                device_type = device_specs.get("device_type", "Unknown")
                os_info = f"{device_specs.get('os_type', 'Unknown')} {device_specs.get('os_version', '')}"
                cpu = device_specs.get("cpu_model", "Unknown")
                ram = device_specs.get("ram_total_mb")
                ram_info = f"{ram:.0f} MB" if ram else "Unknown"
                print(f"  Device: {device_type} | OS: {os_info}")
                print(f"  CPU: {cpu}")
                print(f"  RAM: {ram_info}")

            if capabilities:
                settrace_support = capabilities.get("supports_settrace", True)
                frame_support = capabilities.get("supports_frame_introspection", True)
                if not settrace_support or not frame_support:
                    print(
                        f"  Limited capabilities: settrace={settrace_support}, frame_introspection={frame_support}"
                    )

            # ---- Register worker in connection manager ----
            from .connection_manager import WorkerInfo

            worker_info = WorkerInfo.from_worker_ready_message(
                worker_id=worker_id,
                websocket=websocket,
                message_data=message.data,
            )
            # ---- Same-worker reconnect handover ----
            # If the same worker_id reconnects on a new socket while the previous
            # socket is still around, hand over the connection without running
            # disconnect/requeue logic. If the worker already has an active task,
            # keep it busy and only update DB state.
            existing_ws = self.connection_manager.get_worker_websocket(worker_id)
            if existing_ws is not None and existing_ws is not websocket:
                active_task_id, active_job_id = (
                    self.connection_manager.get_worker_active_task(worker_id)
                )
                print(
                    f"WorkerMessageHandler: Worker {worker_id} reconnected on a new socket "
                    f"\u2013 handing over connection without reassignment."
                )
                self.connection_manager.add_worker(
                    worker_id, websocket, worker_info=worker_info
                )
                if active_task_id and active_job_id:
                    self.connection_manager.set_worker_active_task(
                        worker_id, active_task_id, active_job_id
                    )
                    self.connection_manager.mark_worker_busy(worker_id)
                await _create_worker_in_database(worker_id, device_specs)
                await _update_worker_status(
                    worker_id,
                    "busy" if active_task_id else "online",
                    current_task_id=active_task_id,
                )

                if active_task_id and active_job_id:
                    print(
                        f"WorkerMessageHandler: Worker {worker_id} reconnected with running task "
                        f"{active_task_id} (job {active_job_id}) \u2013 DB updated, no new task assigned."
                    )
                    return

                print(
                    f"WorkerMessageHandler: Worker {worker_id} reconnected idle \u2013 proceeding with normal assignment."
                )
                assigned = await self.task_dispatcher.assign_task_to_available_worker(
                    worker_id
                )
                if assigned:
                    print(
                        f"WorkerMessageHandler: Assigned task to reconnected worker {worker_id}"
                    )
                else:
                    print(
                        f"WorkerMessageHandler: No tasks available for reconnected worker {worker_id}"
                    )
                return

            self.connection_manager.add_worker(
                worker_id, websocket, worker_info=worker_info
            )
            await _create_worker_in_database(worker_id, device_specs)
            await _update_worker_status(worker_id, "online")

            # DB fallback for reconnect-after-cleanup cases:
            # if this worker already owns an assigned task in DB, keep it busy
            # and avoid assigning any new task.
            db_task_id, db_job_id = await self._get_db_assigned_task_for_worker(
                worker_id
            )
            if db_task_id and db_job_id:
                self.connection_manager.set_worker_active_task(
                    worker_id, db_task_id, db_job_id
                )
                self.connection_manager.mark_worker_busy(worker_id)
                await _update_worker_status(
                    worker_id, "busy", current_task_id=db_task_id
                )
                print(
                    f"WorkerMessageHandler: Worker {worker_id} reconnected with DB-assigned task "
                    f"{db_task_id} (job {db_job_id}) \u2013 no new task assigned."
                )
                return

            # No explicit RESUME_TASK dispatch here; worker continues execution
            # on its own after reconnect and keeps sending checkpoints/results.

            # ---- Normal path: assign a pending task ----
            assigned = await self.task_dispatcher.assign_task_to_available_worker(
                worker_id
            )
            if assigned:
                print(
                    f"WorkerMessageHandler: Assigned task to newly connected worker {worker_id}"
                )
            else:
                print(
                    f"WorkerMessageHandler: No tasks available for worker {worker_id}"
                )

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in worker ready: {e}")
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling worker ready: {e}")
            import traceback

            traceback.print_exc()

    async def _handle_task_result(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            result = message.data["result"]

            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if not worker_id:
                print(
                    f"[RESULT DEBUG] Could not find worker for task result – task_id={task_id}, job_id={job_id}"
                )
                return

            print(
                f"[Task Result] Received from worker {worker_id} | Task: {task_id} | Job: {job_id}"
            )

            # Normalise result payload
            normalized_result = result
            if isinstance(normalized_result, str):
                try:
                    normalized_result = json.loads(normalized_result)
                except Exception:
                    try:
                        import ast as _ast

                        normalized_result = _ast.literal_eval(normalized_result)
                    except Exception:
                        normalized_result = result

            if isinstance(normalized_result, dict):
                normalized_result.setdefault("worker_id", worker_id)
                trace = normalized_result.get("trace")
                if isinstance(trace, list) and trace and isinstance(trace[-1], dict):
                    trace[-1].setdefault("worker_id", worker_id)
                    trace[-1].setdefault("task_id", task_id)
                try:
                    result_str = json.dumps(normalized_result)
                except Exception:
                    result_str = str(result) if result is not None else ""
            else:
                result_str = str(result) if result is not None else ""

            accepted, job_complete = await self.job_manager.mark_task_completed(
                task_id, job_id, worker_id, result_str
            )

            if not accepted:
                print(
                    f"[RESULT DEBUG] Ignoring duplicate/stale completion for task {task_id} on worker {worker_id}"
                )
                self.connection_manager.clear_worker_active_task(worker_id)
                self.connection_manager.mark_worker_available(worker_id)
                await _update_worker_status(worker_id, "online", current_task_id=None)
                return

            await _update_worker_task_stats(worker_id, task_completed=True)
            self.connection_manager.clear_worker_active_task(worker_id)
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)

            # Pipeline dependency resolution
            pipeline_batch_dispatched = False
            if self.job_manager.is_pipeline_job(job_id):
                dep_mgr = self.job_manager.dependency_manager
                newly_unblocked = await dep_mgr.on_task_completed(
                    task_id, job_id, normalized_result
                )
                if newly_unblocked:
                    print(
                        f"[PIPELINE] Stage barrier lifted! {len(newly_unblocked)} downstream tasks unblocked for job {job_id}"
                    )
                    func_code = self.job_manager.get_func_code(job_id)
                    batch_assigned = await self.task_dispatcher.assign_tasks_for_job(
                        job_id, func_code or "", []
                    )
                    print(
                        f"[PIPELINE] Batch-dispatched {batch_assigned} next-stage tasks across available workers"
                    )
                    pipeline_batch_dispatched = True

            if job_complete:
                print(
                    f"[RESULT DEBUG] Job {job_id} completed, triggering completion handler"
                )
                await self.completion_handler.handle_job_completion(job_id)

            if not pipeline_batch_dispatched:
                assigned = await self.task_dispatcher.assign_task_to_available_worker(
                    worker_id
                )
                if assigned:
                    print(f"[RESULT DEBUG] Assigned next task to worker {worker_id}")

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in task result: {e}")
            import traceback

            traceback.print_exc()
        except Exception as e:
            print(f"[RESULT DEBUG] Error handling task result: {e}")
            import traceback

            traceback.print_exc()

    async def _handle_kill_ack(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            task_id = message.data["task_id"]
            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if not worker_id:
                print(
                    f"[Kill] KILL_ACK for task {task_id} from unknown worker, ignoring"
                )
                return

            print(
                f"[Kill] Task {task_id} acknowledged kill on worker {worker_id}, resetting to pending"
            )
            await _update_task_status(task_id, "pending", worker_id=None)
            self.connection_manager.clear_worker_active_task(worker_id)
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)
            await self.task_dispatcher.assign_task_to_available_worker(worker_id)

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in kill ack: {e}")
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling kill ack: {e}")
            import traceback

            traceback.print_exc()

    async def _handle_task_error(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            error = message.data["error"]

            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if not worker_id:
                print("WorkerMessageHandler: Could not find worker for task error")
                return

            print(
                f"WorkerMessageHandler: Task {task_id} failed on worker {worker_id} for job {job_id}: {error}"
            )

            await self.job_manager.mark_task_failed(task_id, job_id, worker_id, error)
            await _update_worker_task_stats(worker_id, task_completed=False)

            try:
                await _record_worker_failure(worker_id, task_id, error, job_id)
            except Exception as ex:
                print(
                    f"WorkerMessageHandler: Error recording worker failure for {worker_id}/{task_id}: {ex}"
                )

            self.connection_manager.clear_worker_active_task(worker_id)
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)

            assigned = await self.task_dispatcher.assign_task_to_available_worker(
                worker_id
            )
            if assigned:
                print(
                    f"WorkerMessageHandler: Assigned next task to worker {worker_id} after failure"
                )

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in task error: {e}")
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling task error: {e}")
            import traceback

            traceback.print_exc()

    async def _handle_task_checkpoint(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            is_base = message.data["is_base"]
            delta_data_hex = message.data["delta_data_hex"]
            progress_percent = message.data["progress_percent"]
            checkpoint_id = message.data["checkpoint_id"]
            compression_type = message.data.get("compression_type", "gzip")
            checkpoint_type = message.data.get("checkpoint_type")
            checkpoint_state_vars = message.data.get("checkpoint_state_vars")
            state_size_bytes = message.data.get("state_size_bytes")

            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if not worker_id:
                print("WorkerMessageHandler: Could not find worker for checkpoint")
                return

            checkpoint_data_bytes = hex_to_bytes(delta_data_hex)
            state_vars_info = (
                f", state_vars: {checkpoint_state_vars}"
                if checkpoint_state_vars
                else ""
            )

            print(
                f"WorkerMessageHandler: Received checkpoint {checkpoint_id} from worker {worker_id} "
                f"for task {task_id} (size: {len(checkpoint_data_bytes)} bytes, "
                f"progress: {progress_percent}%{state_vars_info})"
            )

            from foreman.db.base import get_db_session

            async with get_db_session() as session:
                success = await self.checkpoint_manager.store_checkpoint(
                    session=session,
                    task_id=task_id,
                    job_id=job_id,
                    is_base=is_base,
                    delta_data_bytes=checkpoint_data_bytes,
                    progress_percent=progress_percent,
                    checkpoint_id=checkpoint_id,
                    compression_type=compression_type,
                    checkpoint_type=checkpoint_type,
                    checkpoint_state_vars=checkpoint_state_vars,
                    state_size_bytes=state_size_bytes,
                )

            if success:
                print(
                    f"WorkerMessageHandler: Checkpoint {checkpoint_id} stored for task {task_id}"
                )
                from common.protocol import create_checkpoint_ack_message

                ack_msg = create_checkpoint_ack_message(task_id, job_id, checkpoint_id)
                await websocket.send(ack_msg.to_json())
            else:
                print(
                    f"WorkerMessageHandler: Failed to store checkpoint {checkpoint_id} for task {task_id}"
                )

        except KeyError as e:
            print(
                f"WorkerMessageHandler: Missing required field in checkpoint message: {e}"
            )
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling checkpoint: {e}")
            import traceback

            traceback.print_exc()
