"""
WebSocket manager for CrowdCompute Foreman - Refactored
"""

import asyncio
import json
import websockets
from typing import Any, Dict
from websockets.server import WebSocketServerProtocol

from .connection_manager import ConnectionManager
from .job_manager import JobManager
from .task_dispatcher import TaskDispatcher
from .message_handlers import ClientMessageHandler, WorkerMessageHandler
from .completion_handler import JobCompletionHandler
from .worker_reconnection_handler import WorkerReconnectionHandler
from .scheduling import TaskScheduler, create_scheduler
from .staged_results_manager.checkpoint_manager import CheckpointManager
from .utils import (
    _update_worker_status,
    _get_assigned_tasks,
    _record_worker_failure,
    _decode_checkpoint_blob,
    _make_json_serializable,
)
from common.protocol import Message, MessageType, create_ping_message
from foreman.db.base import db_session
from foreman.db.models import TaskModel, TaskAssignmentModel


class WebSocketManager:
    """
    Manages WebSocket connections for workers and clients with pluggable scheduling

    Usage:
        # Create with default FIFO scheduler
        manager = WebSocketManager()

        # Or create with specific scheduler
        manager = WebSocketManager(scheduler_type="performance")

        # Or inject custom scheduler
        custom_scheduler = MyCustomScheduler()
        manager = WebSocketManager(scheduler=custom_scheduler)
    """

    def __init__(self, scheduler: TaskScheduler = None, scheduler_type: str = "wrr"):
        """
        Initialize WebSocket manager

        Args:
            scheduler: Custom scheduler instance (optional)
            scheduler_type: Type of scheduler to create if scheduler not provided
                          Options: "fifo", "round_robin", "performance",
                                  "least_loaded", "priority"
        """
        # Core components
        self.connection_manager = ConnectionManager()
        self.job_manager = JobManager()

        # Scheduler
        if scheduler is None:
            scheduler = create_scheduler(scheduler_type)
        self.scheduler = scheduler

        # Task dispatcher
        self.task_dispatcher = TaskDispatcher(
            scheduler=self.scheduler,
            connection_manager=self.connection_manager,
            job_manager=self.job_manager,
        )

        self.checkpoint_manager = CheckpointManager()
        self.reconnection_handler = WorkerReconnectionHandler(
            job_manager=self.job_manager,
            task_dispatcher=self.task_dispatcher,
            checkpoint_manager=self.checkpoint_manager,
            connection_manager=self.connection_manager,
            resume_ttl_seconds=300,
        )

        # Completion handler
        self.completion_handler = JobCompletionHandler(
            connection_manager=self.connection_manager, job_manager=self.job_manager
        )

        # Message handlers
        self.client_handler = ClientMessageHandler(
            connection_manager=self.connection_manager,
            job_manager=self.job_manager,
            task_dispatcher=self.task_dispatcher,
        )

        self.worker_handler = WorkerMessageHandler(
            connection_manager=self.connection_manager,
            job_manager=self.job_manager,
            task_dispatcher=self.task_dispatcher,
            completion_handler=self.completion_handler,
            checkpoint_manager=self.checkpoint_manager,
            reconnection_handler=self.reconnection_handler,
        )

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        client_type = None
        worker_id_debug = None  # Track worker ID for debugging

        try:
            async for message in websocket:
                try:
                    msg = Message.from_json(message)

                    if client_type is None:
                        # First message determines client type
                        client_type = self._determine_client_type(msg)
                        if client_type is None:
                            print(
                                f"🔴 [DISCONNECT DEBUG] Unknown first message type: {msg.type} - breaking connection"
                            )
                            break
                        if client_type == "worker" and "worker_id" in msg.data:
                            worker_id_debug = msg.data["worker_id"]
                            print(f"🟢 [CONNECTION] Worker {worker_id_debug} connected")

                    # Route message to appropriate handler
                    if client_type == "client":
                        await self.client_handler.handle_message(msg, websocket)
                    elif client_type == "worker":
                        await self.worker_handler.handle_message(msg, websocket)

                except Exception as e:
                    print(
                        f"[DISCONNECT DEBUG] Error handling message for {client_type} {worker_id_debug}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    # Don't break - continue listening for messages

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        finally:
            print(
                f"[CLEANUP] Cleaning up connection for {client_type} {worker_id_debug}"
            )
            await self._cleanup_connection(websocket, client_type)

    def _determine_client_type(self, message: Message) -> str:
        """Determine if connection is from client or worker based on first message"""
        if message.type in (
            MessageType.SUBMIT_JOB,
            MessageType.SUBMIT_BROADCAST_JOB,
            MessageType.SUBMIT_PIPELINE_JOB,
        ):
            return "client"
        elif message.type == MessageType.WORKER_READY:
            return "worker"
        return None

    async def _cleanup_connection(
        self, websocket: WebSocketServerProtocol, client_type: str
    ):
        """Clean up when a connection is closed"""
        if client_type == "worker":
            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if worker_id:
                pending_tasks = await self._get_worker_pending_tasks(worker_id)
                print(
                    f"[CLEANUP DEBUG] Worker {worker_id} disconnecting with {len(pending_tasks)} pending tasks: {pending_tasks}"
                )
                await self.worker_handler.handle_disconnection(websocket)
                await _update_worker_status(worker_id, "offline", current_task_id=None)
            else:
                print(
                    f"[CLEANUP DEBUG] Worker websocket not found in connection manager (already cleaned up?)"
                )

        elif client_type == "client":
            job_id = self.connection_manager.find_job_by_websocket(websocket)
            if job_id:
                print(f"[CLEANUP DEBUG] Client for job {job_id} disconnected")
                self.connection_manager.remove_client(job_id)

    async def _record_worker_disconnection_failures(self, worker_id: str):
        """
        Record failures for all tasks assigned to a disconnected worker

        Extracts and decodes the latest checkpoint data from delta_checkpoint_blobs
        and stores it in the worker_failures table.

        Args:
            worker_id: ID of the disconnected worker
        """
        try:
            # Get all assigned tasks whose running assignment belongs to this worker
            assigned_tasks = await _get_assigned_tasks()
            worker_tasks = []
            async with db_session() as session:
                for task in assigned_tasks:
                    from sqlalchemy import select as sa_select, and_

                    result = await session.execute(
                        sa_select(TaskAssignmentModel).where(
                            and_(
                                TaskAssignmentModel.task_id == task.id,
                                TaskAssignmentModel.worker_id == worker_id,
                                TaskAssignmentModel.status == "running",
                            )
                        )
                    )
                    if result.scalar_one_or_none():
                        worker_tasks.append(task)

            if not worker_tasks:
                print(f"No tasks assigned to disconnected worker {worker_id}")
                return

            print(
                f"Recording {len(worker_tasks)} failure(s) for disconnected worker {worker_id}"
            )

            # Get full task data including checkpoint blobs
            async with db_session() as session:
                for task in worker_tasks:
                    # Fetch the full task with checkpoint data
                    full_task = await session.get(TaskModel, task.id)
                    if not full_task:
                        continue

                    # Check if checkpoint data exists
                    checkpoint_available = bool(full_task.base_checkpoint_data)
                    latest_checkpoint_data = None

                    # Decode the latest delta checkpoint blob
                    if full_task.delta_checkpoint_blobs:
                        try:
                            delta_blobs = json.loads(full_task.delta_checkpoint_blobs)
                            if delta_blobs:
                                # Get the latest checkpoint (highest checkpoint_id)
                                latest_checkpoint_id = max(delta_blobs.keys(), key=int)
                                latest_blob = delta_blobs[latest_checkpoint_id]

                                # Decode the blob
                                decoded_state = _decode_checkpoint_blob(latest_blob)

                                if decoded_state is not None:
                                    # Make it JSON serializable
                                    serializable_state = _make_json_serializable(
                                        decoded_state
                                    )
                                    latest_checkpoint_data = json.dumps(
                                        {
                                            "checkpoint_id": int(latest_checkpoint_id),
                                            "progress_percent": full_task.progress_percent,
                                            "checkpoint_count": full_task.checkpoint_count,
                                            "state": serializable_state,
                                        }
                                    )
                                    print(
                                        f"  Task {task.id}: Decoded checkpoint #{latest_checkpoint_id} "
                                        f"(progress: {full_task.progress_percent}%)"
                                    )
                        except Exception as e:
                            print(f"  Task {task.id}: Error decoding checkpoint: {e}")

                    # Record the failure
                    await _record_worker_failure(
                        worker_id=worker_id,
                        task_id=task.id,
                        error="Worker disconnected while task was assigned",
                        job_id=full_task.job_id,
                        checkpoint_available=checkpoint_available,
                        latest_checkpoint_data=latest_checkpoint_data,
                    )

                    print(
                        f"  Recorded failure for task {task.id} "
                        f"(checkpoint: {'yes' if checkpoint_available else 'no'})"
                    )

        except Exception as e:
            print(f"Error recording worker disconnection failures: {e}")
            import traceback

            traceback.print_exc()

    async def _get_worker_pending_tasks(self, worker_id: str) -> list:
        """Get list of pending task IDs for a worker (for debugging)"""
        try:
            from foreman.db.base import async_session
            from foreman.db.models import TaskModel
            from sqlalchemy import select

            async with async_session() as session:
                result = await session.execute(
                    select(TaskModel.id, TaskModel.status)
                    .join(
                        TaskAssignmentModel, TaskAssignmentModel.task_id == TaskModel.id
                    )
                    .where(TaskAssignmentModel.worker_id == worker_id)
                    .where(TaskAssignmentModel.status == "running")
                    .where(TaskModel.status == "assigned")
                )
                tasks = result.all()
                return [(t.id, t.status) for t in tasks]
        except Exception as e:
            print(f"⚠️ [CLEANUP DEBUG] Error getting pending tasks: {e}")
            return []

    async def ping_workers(self):
        """Periodically ping workers to keep connections alive"""
        while True:
            try:
                await asyncio.sleep(130)  # Ping every 30 seconds

                ping_message = create_ping_message()
                worker_ids = self.connection_manager.get_all_worker_ids()

                for worker_id in worker_ids:
                    websocket = self.connection_manager.get_worker_websocket(worker_id)
                    if websocket:
                        try:
                            await websocket.send(ping_message.to_json())
                        except Exception:
                            # Worker connection is dead, will be cleaned up
                            pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in ping task: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current WebSocket manager stats"""
        return self.connection_manager.get_stats()

    def change_scheduler(
        self, scheduler: TaskScheduler = None, scheduler_type: str = None
    ):
        """
        Change the scheduling algorithm at runtime

        Args:
            scheduler: Custom scheduler instance
            scheduler_type: Type of scheduler to create
        """
        if scheduler is None and scheduler_type is None:
            raise ValueError("Must provide either scheduler or scheduler_type")

        if scheduler is None:
            scheduler = create_scheduler(scheduler_type)

        self.scheduler = scheduler
        self.task_dispatcher.scheduler = scheduler
        print(f"Scheduler changed to {scheduler.__class__.__name__}")
