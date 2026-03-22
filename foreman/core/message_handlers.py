"""
Message handling logic separated by client type
"""

import json

from websockets.server import WebSocketServerProtocol

from .utils import (
    _create_worker_in_database,
    _record_worker_failure,
    _update_worker_status,
    _update_worker_task_stats,
)
from common.protocol import (
    Message,
    MessageType,
    create_job_accepted_message,
    create_load_model_message,
)
from common.serializer import get_runtime_info, bytes_to_hex, hex_to_bytes
from .staged_results_manager.checkpoint_manager import CheckpointManager
from .model_registry import build_model_artifact_url, store_partition_blob


class ClientMessageHandler:
    """
    Handles messages from client SDK

    Responsibilities:
    - Handle job submissions
    - Handle client disconnections
    - Send job acknowledgments
    """

    def __init__(self, connection_manager, job_manager, task_dispatcher):
        """
        Initialize client message handler

        Args:
            connection_manager: ConnectionManager instance
            job_manager: JobManager instance
            task_dispatcher: TaskDispatcher instance
        """
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher

    async def _dispatch_load_model_instructions(
        self,
        job_id: str,
        model_version_id: str,
        topology_nodes,
        model_artifacts,
    ):
        """Persist model artifacts and send LOAD_MODEL messages to assigned workers."""
        if not model_artifacts:
            return

        node_to_worker = {}
        for node in topology_nodes or []:
            partition_id = node.get("model_partition_id")
            assigned_worker = node.get("assigned_device_id")
            if partition_id and assigned_worker:
                node_to_worker[partition_id] = assigned_worker

        for artifact in model_artifacts:
            partition_id = artifact.get("model_partition_id")
            content_b64 = artifact.get("content_b64")
            if not partition_id or not content_b64:
                continue

            stored = store_partition_blob(
                model_version_id=model_version_id,
                model_partition_id=partition_id,
                content_b64=content_b64,
                file_name=artifact.get("file_name"),
            )

            assigned_worker = artifact.get("assigned_device_id") or node_to_worker.get(
                partition_id
            )
            if not assigned_worker:
                print(
                    f"ClientMessageHandler: No assigned worker found for partition {partition_id}; skipping LOAD_MODEL dispatch"
                )
                continue

            worker_ws = self.connection_manager.get_worker_websocket(assigned_worker)
            if not worker_ws:
                print(
                    f"ClientMessageHandler: Assigned worker {assigned_worker} is not connected for partition {partition_id}"
                )
                continue

            model_url = build_model_artifact_url(model_version_id, stored["file_name"])
            load_msg = create_load_model_message(
                job_id=job_id,
                model_version_id=model_version_id,
                model_partition_id=partition_id,
                model_uri=model_url,
                checksum=stored["checksum"],
            )
            await worker_ws.send(load_msg.to_json())
            print(
                f"ClientMessageHandler: Sent LOAD_MODEL for partition {partition_id} to worker {assigned_worker}"
            )

    async def handle_message(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """
        Route client messages to appropriate handlers

        Args:
            message: Parsed message
            websocket: WebSocket connection
        """
        if message.type == MessageType.SUBMIT_JOB:
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
        """
        Handle a new job submission from client

        Args:
            message: Job submission message
            websocket: Client websocket connection
        """
        try:
            job_id = message.job_id
            func_code = message.data["func_code"]
            args_list = message.data["args_list"]
            total_tasks = message.data["total_tasks"]

            # Extract task metadata for declarative checkpointing
            task_metadata = message.data.get("task_metadata")

            print(
                f"ClientMessageHandler: Received job {job_id} with {total_tasks} tasks | foreman_runtime={get_runtime_info()}"
            )

            # Log checkpoint configuration if present
            if task_metadata and task_metadata.get("checkpoint_enabled"):
                print(
                    f"ClientMessageHandler: Job {job_id} has checkpointing enabled "
                    f"(interval={task_metadata.get('checkpoint_interval')}s, "
                    f"state_vars={task_metadata.get('checkpoint_state', [])})"
                )

            # Register client websocket
            self.connection_manager.add_client(job_id, websocket)

            # Create job and tasks with task_metadata
            await self.job_manager.create_job(
                job_id, func_code, args_list, total_tasks, task_metadata=task_metadata
            )

            # Try to assign tasks to available workers
            assigned = await self.task_dispatcher.assign_tasks_for_job(
                job_id, func_code, args_list
            )

            print(
                f"ClientMessageHandler: Assigned {assigned} tasks immediately for job {job_id}"
            )

            # Send job accepted message
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
        """
        Handle a pipeline job submission from client.

        A pipeline job defines multiple stages with dependency relationships.
        The system creates all tasks up-front with dependency counters:
        - Stage 0 tasks start as "pending" and are dispatched immediately
        - Later stages start as "blocked" and are unblocked automatically
          when their upstream dependencies complete

        This is the pipeline equivalent of _handle_job_submission().

        Args:
            message: Pipeline job submission message
            websocket: Client websocket connection
        """
        try:
            job_id = message.job_id
            stages = message.data["stages"]
            total_tasks = message.data["total_tasks"]
            total_stages = message.data["total_stages"]
            dependency_map = message.data.get("dependency_map")
            task_metadata = message.data.get("task_metadata")
            dnn_config = message.data.get("dnn_config")

            print(
                f"ClientMessageHandler: Received pipeline job {job_id} "
                f"with {total_stages} stages, {total_tasks} tasks | "
                f"foreman_runtime={get_runtime_info()}"
            )

            # Log stage details
            for i, stage in enumerate(stages):
                stage_name = stage.get("name", f"stage_{i}")
                stage_tasks = len(stage["args_list"])
                pass_results = stage.get("pass_upstream_results", False)
                print(
                    f"  Stage {i} ({stage_name}): {stage_tasks} tasks"
                    f"{' [receives upstream results]' if pass_results else ''}"
                )

            # Register client websocket
            self.connection_manager.add_client(job_id, websocket)

            # Create pipeline job with dependency wiring.
            # If dnn_config is provided, route through DNN-aware creation.
            if dnn_config:
                inference_graph_id = dnn_config["inference_graph_id"]
                topology_nodes = dnn_config["topology_nodes"]
                topology_edges = dnn_config["topology_edges"]
                model_version_id = dnn_config.get("model_version_id")
                aggregation_strategy = dnn_config.get("aggregation_strategy", "average")
                model_artifacts = dnn_config.get("model_artifacts", [])

                created = await self.job_manager.create_dnn_inference_job(
                    job_id=job_id,
                    stages=stages,
                    inference_graph_id=inference_graph_id,
                    topology_nodes=topology_nodes,
                    topology_edges=topology_edges,
                    model_version_id=model_version_id,
                    aggregation_strategy=aggregation_strategy,
                    dependency_map=dependency_map,
                    task_metadata=task_metadata,
                )

                if model_version_id and model_artifacts:
                    await self._dispatch_load_model_instructions(
                        job_id=job_id,
                        model_version_id=model_version_id,
                        topology_nodes=topology_nodes,
                        model_artifacts=model_artifacts,
                    )
            else:
                created = await self.job_manager.create_pipeline_job(
                    job_id, stages, dependency_map, task_metadata
                )

            # Dispatch Stage 0 tasks (they are "pending" from creation)
            stage_0_func_code = stages[0]["func_code"]
            assigned = await self.task_dispatcher.assign_tasks_for_job(
                job_id, stage_0_func_code, stages[0]["args_list"]
            )

            print(
                f"ClientMessageHandler: Pipeline job {job_id} — "
                f"Assigned {assigned} stage-0 tasks immediately"
            )

            # Send job accepted message
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
    Handles messages from workers

    Responsibilities:
    - Handle worker registration
    - Handle task results
    - Handle task errors
    - Handle pong responses
    """

    def __init__(
        self,
        connection_manager,
        job_manager,
        task_dispatcher,
        completion_handler,
        checkpoint_manager: CheckpointManager = None,
    ):
        """
        Initialize worker message handler

        Args:
            connection_manager: ConnectionManager instance
            job_manager: JobManager instance
            task_dispatcher: TaskDispatcher instance
            completion_handler: JobCompletionHandler instance
            checkpoint_manager: CheckpointManager instance for checkpoint handling
        """
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher
        self.completion_handler = completion_handler
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

    async def handle_message(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """
        Route worker messages to appropriate handlers

        Args:
            message: Parsed message
            websocket: WebSocket connection
        """
        if message.type == MessageType.WORKER_READY:
            await self._handle_worker_ready(message, websocket)
        elif message.type == MessageType.TASK_RESULT:
            await self._handle_task_result(message, websocket)
        elif message.type == MessageType.TASK_ERROR:
            await self._handle_task_error(message, websocket)
        elif message.type == MessageType.PONG:
            await self._handle_pong(message, websocket)
        elif message.type == MessageType.WORKER_HEARTBEAT:
            # Fallback for workers sending heartbeat instead of PONG
            await self._handle_pong(message, websocket)
        elif message.type == MessageType.TASK_CHECKPOINT:
            await self._handle_task_checkpoint(message, websocket)
        elif message.type == MessageType.INTERMEDIATE_FEATURE:
            await self._handle_intermediate_feature(message, websocket)
        else:
            print(f"WorkerMessageHandler: Unknown message type: {message.type}")

    async def _handle_pong(self, message: Message, websocket: WebSocketServerProtocol):
        """
        Handle pong from worker (heartbeat response + performance metrics)

        Args:
            message: Pong message
            websocket: Worker websocket connection
        """
        worker_id = self.connection_manager.find_worker_by_websocket(websocket)

        if worker_id:
            # Update worker's last seen time in database
            await _update_worker_status(worker_id, "online")

            # Update performance metrics if provided
            if "performance_metrics" in message.data:
                metrics = message.data["performance_metrics"]
                await self._update_worker_performance_metrics(worker_id, metrics)

    async def _update_worker_performance_metrics(self, worker_id: str, metrics: dict):
        """
        Update worker performance metrics in database

        Args:
            worker_id: Worker identifier
            metrics: Performance metrics dictionary
        """
        try:
            from foreman.db.base import async_session
            from foreman.db.models import WorkerModel
            from sqlalchemy import select, update
            from datetime import datetime

            async with async_session() as session:
                # Update worker performance fields
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
        Handle worker ready message (worker registration)

        Extended to extract worker capabilities for mobile support.
        Workers report their platform, runtime, and capabilities, which
        determines if code instrumentation is needed.

        Args:
            message: Worker ready message
            websocket: Worker websocket connection
        """
        try:
            worker_id = message.data["worker_id"]
            device_specs = message.data.get("device_specs", {})

            # Extract platform/runtime info for mobile worker support
            worker_type = message.data.get("worker_type", "pc_python")
            platform = message.data.get("platform", "unknown")
            runtime = message.data.get("runtime", "cpython")
            model_runtime = message.data.get("model_runtime")
            capabilities = message.data.get("capabilities", {})

            # Keep runtime metadata in device specs for DB persistence.
            if runtime and "runtime" not in device_specs:
                device_specs["runtime"] = runtime
            if model_runtime and "model_runtime" not in device_specs:
                device_specs["model_runtime"] = model_runtime

            print(f"WorkerMessageHandler: Worker {worker_id} connected")
            print(f"  Type: {worker_type} | Platform: {platform} | Runtime: {runtime}")

            # Log device specs if available
            if device_specs:
                device_type = device_specs.get("device_type", "Unknown")
                os_info = f"{device_specs.get('os_type', 'Unknown')} {device_specs.get('os_version', '')}"
                cpu = device_specs.get("cpu_model", "Unknown")
                ram = device_specs.get("ram_total_mb")
                ram_info = f"{ram:.0f} MB" if ram else "Unknown"
                print(f"  📱 Device: {device_type} | OS: {os_info}")
                print(f"  🖥️  CPU: {cpu}")
                print(f"  💾 RAM: {ram_info}")

            # Log capabilities for mobile workers
            if capabilities:
                settrace_support = capabilities.get("supports_settrace", True)
                frame_support = capabilities.get("supports_frame_introspection", True)
                if not settrace_support or not frame_support:
                    print(
                        f"  ⚠️  Limited capabilities: settrace={settrace_support}, frame_introspection={frame_support}"
                    )

            # Create WorkerInfo for extended capability tracking
            from .connection_manager import WorkerInfo

            worker_info = WorkerInfo.from_worker_ready_message(
                worker_id=worker_id, websocket=websocket, message_data=message.data
            )

            # Register worker with capabilities
            self.connection_manager.add_worker(
                worker_id, websocket, worker_info=worker_info
            )

            # Create or update worker in database with device specs
            await _create_worker_in_database(worker_id, device_specs)

            # Update worker status in database
            await _update_worker_status(worker_id, "online")

            # Assign any pending tasks to this worker
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
        """
        Handle task result from worker (successful completion)

        Args:
            message: Task result message
            websocket: Worker websocket connection
        """
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            result = message.data["result"]

            # Find worker ID
            worker_id = self.connection_manager.find_worker_by_websocket(websocket)

            if not worker_id:
                print(
                    f"⚠️ [RESULT DEBUG] Could not find worker for task result - task_id={task_id}, job_id={job_id}"
                )
                print(
                    f"⚠️ [RESULT DEBUG] This may indicate worker was disconnected before result was processed!"
                )
                return

            print(
                f"[Task Result] Received from worker {worker_id} | Task: {task_id} | Job: {job_id}"
            )

            # Normalize result payload so pipeline traces carry execution worker_id.
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
                if isinstance(trace, list) and trace:
                    if isinstance(trace[-1], dict):
                        trace[-1].setdefault("worker_id", worker_id)
                        trace[-1].setdefault("task_id", task_id)

                try:
                    result_str = json.dumps(normalized_result)
                except Exception:
                    result_str = str(result) if result is not None else ""
            else:
                # Result should already be a string from worker, but ensure it.
                result_str = str(result) if result is not None else ""

            print(
                f"[Task Result] Result preview: {result_str[:100]}..."
                if len(result_str) > 100
                else f"📊 [Task Result] Result: {result_str}"
            )

            # Mark task as completed in job manager (idempotent)
            accepted, job_complete = await self.job_manager.mark_task_completed(
                task_id, job_id, worker_id, result_str
            )

            if not accepted:
                print(
                    f"⚠️ [RESULT DEBUG] Ignoring duplicate/stale completion for task {task_id} on worker {worker_id}"
                )
                # Even if ignored, free the worker to take new tasks
                self.connection_manager.mark_worker_available(worker_id)
                await _update_worker_status(worker_id, "online", current_task_id=None)
                return

            print(
                f"[RESULT DEBUG] Task {task_id} accepted, job_complete={job_complete}"
            )

            # Update worker statistics
            await _update_worker_task_stats(worker_id, task_completed=True)

            # Mark worker as available again
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)

            # --- Pipeline dependency resolution ---
            # If this is a pipeline job, decrement dependency counters of
            # downstream tasks.  When all tasks in a stage complete (barrier
            # lifts), batch-dispatch the entire next stage to ALL available
            # workers at once – not just the worker that finished last.
            pipeline_batch_dispatched = False
            if self.job_manager.is_pipeline_job(job_id):
                dep_mgr = self.job_manager.dependency_manager
                newly_unblocked = await dep_mgr.on_task_completed(
                    task_id, job_id, normalized_result
                )
                if newly_unblocked:
                    print(
                        f"[PIPELINE] 🚀 Stage barrier lifted! {len(newly_unblocked)} "
                        f"downstream tasks unblocked for job {job_id}"
                    )
                    # Batch-dispatch ALL unblocked tasks to the best available
                    # workers (including the current worker and any idle ones).
                    func_code = self.job_manager.get_func_code(job_id)
                    batch_assigned = await self.task_dispatcher.assign_tasks_for_job(
                        job_id, func_code or "", []
                    )
                    print(
                        f"[PIPELINE] Batch-dispatched {batch_assigned} next-stage "
                        f"tasks across available workers"
                    )
                    pipeline_batch_dispatched = True

            # Check if job is complete and handle completion
            if job_complete:
                print(
                    f"[RESULT DEBUG] Job {job_id} completed, triggering completion handler"
                )
                await self.completion_handler.handle_job_completion(job_id)

            # For non-pipeline jobs, or if the pipeline batch didn't assign
            # anything to THIS worker, try assigning the next task normally.
            if not pipeline_batch_dispatched:
                assigned = await self.task_dispatcher.assign_task_to_available_worker(
                    worker_id
                )

                if assigned:
                    print(f"[RESULT DEBUG] Assigned next task to worker {worker_id}")
                else:
                    print(
                        f"[RESULT DEBUG] No more tasks to assign to worker {worker_id}"
                    )

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in task result: {e}")
            import traceback

            traceback.print_exc()
        except Exception as e:
            print(f"[RESULT DEBUG] Error handling task result: {e}")
            import traceback

            traceback.print_exc()

    async def _handle_task_error(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """
        Handle task error from worker (task failure)

        Args:
            message: Task error message
            websocket: Worker websocket connection
        """
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            error = message.data["error"]

            # Find worker ID
            worker_id = self.connection_manager.find_worker_by_websocket(websocket)

            if not worker_id:
                print(f"WorkerMessageHandler: Could not find worker for task error")
                return

            print(
                f"WorkerMessageHandler: Task {task_id} failed on worker {worker_id} for job {job_id}: {error}"
            )

            # Mark task as failed (resets to pending for retry)
            await self.job_manager.mark_task_failed(task_id, job_id, worker_id, error)

            # Update worker statistics
            await _update_worker_task_stats(worker_id, task_completed=False)

            # Record failure in history
            try:
                await _record_worker_failure(worker_id, task_id, error, job_id)
            except Exception as ex:
                print(
                    f"WorkerMessageHandler: Error recording worker failure for {worker_id}/{task_id}: {ex}"
                )

            # Mark worker as available again
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)

            # Assign next task to this worker (the failed task will be retried by another worker)
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
        """
        Handle checkpoint message from worker (incremental state save)

        Supports both legacy checkpoints and declarative checkpointing with
        additional metadata (checkpoint_type, checkpoint_state_vars, state_size_bytes).

        Args:
            message: Task checkpoint message
            websocket: Worker websocket connection
        """
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            is_base = message.data["is_base"]
            delta_data_hex = message.data["delta_data_hex"]
            progress_percent = message.data["progress_percent"]
            checkpoint_id = message.data["checkpoint_id"]
            compression_type = message.data.get("compression_type", "gzip")

            # Extract declarative checkpointing metadata (optional)
            checkpoint_type = message.data.get("checkpoint_type")
            checkpoint_state_vars = message.data.get("checkpoint_state_vars")
            state_size_bytes = message.data.get("state_size_bytes")

            # Find worker ID
            worker_id = self.connection_manager.find_worker_by_websocket(websocket)

            if not worker_id:
                print(f"WorkerMessageHandler: Could not find worker for checkpoint")
                return

            # Convert hex back to bytes
            checkpoint_data_bytes = hex_to_bytes(delta_data_hex)

            # Build log message with declarative checkpoint details if present
            state_vars_info = ""
            if checkpoint_state_vars:
                state_vars_info = f", state_vars: {checkpoint_state_vars}"

            print(
                f"WorkerMessageHandler: Received checkpoint {checkpoint_id} from worker {worker_id} "
                f"for task {task_id} (size: {len(checkpoint_data_bytes)} bytes, "
                f"progress: {progress_percent}%{state_vars_info})"
            )

            # Get database session and store checkpoint
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

                # Send acknowledgment to worker
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

    async def _handle_intermediate_feature(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """Handle an explicit intermediate feature payload from worker."""
        try:
            job_id = message.job_id
            source_task_id = message.data["task_id"]
            target_task_id = message.data["target_task_id"]
            payload = message.data.get("payload")
            payload_format = message.data.get("payload_format", "json")

            dep_mgr = self.job_manager.dependency_manager
            dep_mgr.register_intermediate_feature(
                job_id=job_id,
                source_task_id=source_task_id,
                target_task_id=target_task_id,
                payload=payload,
                payload_format=payload_format,
            )

            print(
                f"WorkerMessageHandler: Routed intermediate feature "
                f"{source_task_id} -> {target_task_id} for job {job_id}"
            )

        except KeyError as e:
            print(
                f"WorkerMessageHandler: Missing required field in intermediate feature: {e}"
            )
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling intermediate feature: {e}")
            import traceback

            traceback.print_exc()
