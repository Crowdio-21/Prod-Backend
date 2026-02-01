"""
WebSocket manager for CrowdCompute Foreman - Refactored
"""

import asyncio
import websockets
from typing import Any, Dict
from websockets.server import WebSocketServerProtocol

from .connection_manager import ConnectionManager
from .job_manager import JobManager
from .task_dispatcher import TaskDispatcher
from .message_handlers import ClientMessageHandler, WorkerMessageHandler
from .completion_handler import JobCompletionHandler
from .scheduling import TaskScheduler, create_scheduler
from .utils import _update_worker_status
from common.protocol import Message, MessageType, create_ping_message


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

    def __init__(self, scheduler: TaskScheduler = None, scheduler_type: str = "fifo"):
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
                        f"🔴 [DISCONNECT DEBUG] Error handling message for {client_type} {worker_id_debug}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    print(
                        f"🔴 [DISCONNECT DEBUG] Breaking connection due to message handling error"
                    )
                    break

            # If we exit the loop normally (no exception), it means the client/worker closed the connection
            print(
                f"🟡 [DISCONNECT DEBUG] Message loop ended for {client_type} {worker_id_debug} - connection closed by remote"
            )

        except websockets.exceptions.ConnectionClosed as e:
            print(
                f"🔴 [DISCONNECT DEBUG] WebSocket connection closed for {client_type} {worker_id_debug}: code={e.code}, reason={e.reason}"
            )
        except Exception as e:
            print(
                f"🔴 [DISCONNECT DEBUG] Unexpected exception for {client_type} {worker_id_debug}: {e}"
            )
            import traceback

            traceback.print_exc()
        finally:
            print(
                f"🔴 [CLEANUP] Cleaning up connection for {client_type} {worker_id_debug}"
            )
            await self._cleanup_connection(websocket, client_type)

    def _determine_client_type(self, message: Message) -> str:
        """Determine if connection is from client or worker based on first message"""
        if message.type == MessageType.SUBMIT_JOB:
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
                # Check if this worker has any pending tasks that haven't been received yet
                pending_tasks = await self._get_worker_pending_tasks(worker_id)
                print(
                    f"🔴 [CLEANUP DEBUG] Worker {worker_id} disconnecting with {len(pending_tasks)} pending tasks: {pending_tasks}"
                )
                self.connection_manager.remove_worker(worker_id)
                await _update_worker_status(worker_id, "offline")
            else:
                print(
                    f"🔴 [CLEANUP DEBUG] Worker websocket not found in connection manager (already cleaned up?)"
                )

        elif client_type == "client":
            job_id = self.connection_manager.find_job_by_websocket(websocket)
            if job_id:
                print(f"🔴 [CLEANUP DEBUG] Client for job {job_id} disconnected")
                self.connection_manager.remove_client(job_id)

    async def _get_worker_pending_tasks(self, worker_id: str) -> list:
        """Get list of pending task IDs for a worker (for debugging)"""
        try:
            from foreman.db.base import async_session
            from foreman.db.models import TaskModel
            from sqlalchemy import select

            async with async_session() as session:
                result = await session.execute(
                    select(TaskModel.id, TaskModel.status)
                    .where(TaskModel.worker_id == worker_id)
                    .where(TaskModel.status == "assigned")
                )
                tasks = result.all()
                return [(t.id, t.status) for t in tasks]
        except Exception as e:
            print(f"⚠️ [CLEANUP DEBUG] Error getting pending tasks: {e}")
            return []

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
