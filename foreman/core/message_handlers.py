"""
Message handling logic separated by client type
"""

from websockets.server import WebSocketServerProtocol

from .utils import (
    _create_worker_in_database,
    _record_worker_failure,
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

            print(
                f"ClientMessageHandler: Received job {job_id} with {total_tasks} tasks | foreman_runtime={get_runtime_info()}"
            )

            # Register client websocket
            self.connection_manager.add_client(job_id, websocket)

            # Create job and tasks
            await self.job_manager.create_job(job_id, func_code, args_list, total_tasks)

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

        Args:
            message: Worker ready message
            websocket: Worker websocket connection
        """
        try:
            worker_id = message.data["worker_id"]
            device_specs = message.data.get("device_specs", {})

            print(f"WorkerMessageHandler: Worker {worker_id} connected")

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

            # Register worker
            self.connection_manager.add_worker(worker_id, websocket)

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
                print(f"WorkerMessageHandler: Could not find worker for task result")
                return

            print(
                f"[Task Result] Received from worker {worker_id} | Task: {task_id} | Job: {job_id}"
            )

            # Result should already be a string from worker, but ensure it
            result_str = str(result) if result is not None else ""
            
            print(f"[Task Result] Result preview: {result_str[:100]}..." if len(result_str) > 100 else f"📊 [Task Result] Result: {result_str}")

            # Mark task as completed in job manager (idempotent)
            accepted, job_complete = await self.job_manager.mark_task_completed(
                task_id, job_id, worker_id, result_str
            )

            if not accepted:
                print(
                    f"WorkerMessageHandler: Ignoring duplicate/stale completion for task {task_id} on worker {worker_id}"
                )
                # Even if ignored, free the worker to take new tasks
                self.connection_manager.mark_worker_available(worker_id)
                await _update_worker_status(worker_id, "online", current_task_id=None)
                return

            # Update worker statistics
            await _update_worker_task_stats(worker_id, task_completed=True)

            # Mark worker as available again
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)

            # Check if job is complete and handle completion
            if job_complete:
                print(
                    f"WorkerMessageHandler: Job {job_id} completed, triggering completion handler"
                )
                await self.completion_handler.handle_job_completion(job_id)

            # Assign next task to this worker
            assigned = await self.task_dispatcher.assign_task_to_available_worker(
                worker_id
            )

            if assigned:
                print(f"WorkerMessageHandler: Assigned next task to worker {worker_id}")

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in task result: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling task result: {e}")
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

            # Find worker ID
            worker_id = self.connection_manager.find_worker_by_websocket(websocket)

            if not worker_id:
                print(f"WorkerMessageHandler: Could not find worker for checkpoint")
                return

            # Convert hex back to bytes
            checkpoint_data_bytes = hex_to_bytes(delta_data_hex)

            print(
                f"WorkerMessageHandler: Received checkpoint {checkpoint_id} from worker {worker_id} "
                f"for task {task_id} (size: {len(checkpoint_data_bytes)} bytes, "
                f"progress: {progress_percent}%)"
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
