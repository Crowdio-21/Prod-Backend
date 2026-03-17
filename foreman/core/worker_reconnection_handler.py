"""
Worker reconnection handling with checkpoint-based task resumption.

Design
------
When a worker disconnects (network drop, crash, etc.) there are two cases:

Case A – Checkpointing OFF  
    The in-flight task is returned to the pending queue so any available
    worker can retry it from scratch.

Case B – Checkpointing ON  
    The last stored checkpoint is kept in the "resumable" registry.
    When the *same* worker_id reconnects it receives a RESUME_TASK message
    that includes the checkpoint payload, letting it continue where it left
    off.  If the worker never comes back (configurable TTL), the task falls
    back to Case A (fresh retry by any worker).

Integration points
------------------
- Call ``handle_worker_disconnected(worker_id)`` from your WebSocket
  disconnect handler (replace or supplement the current "mark_worker_available"
  logic on disconnect).
- Call ``handle_worker_reconnected(worker_id, websocket)`` from
  ``_handle_worker_ready`` BEFORE the existing task-assignment logic.
  If it returns ``True``, a resume message was already sent; skip the normal
  new-task assignment for this worker.

Nothing else in the existing code needs to change.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from websockets.server import WebSocketServerProtocol

from .utils import _update_worker_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DisconnectedWorkerRecord:
    """Everything we need to decide what to do when a worker comes back."""

    worker_id: str
    task_id: Optional[str]
    job_id: Optional[str]
    checkpointing_enabled: bool
    latest_checkpoint_id: Optional[str]
    disconnected_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def had_active_task(self) -> bool:
        return self.task_id is not None and self.job_id is not None

    def is_expired(self, ttl_seconds: int) -> bool:
        return (datetime.utcnow() - self.disconnected_at).total_seconds() > ttl_seconds


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

class WorkerReconnectionHandler:
    """
    Tracks disconnected workers and orchestrates checkpoint-based resumption.

    Parameters
    ----------
    job_manager        : gives us task metadata and ``mark_task_failed``
    task_dispatcher    : used to re-queue tasks that cannot be resumed
    checkpoint_manager : used to fetch the latest checkpoint bytes for a task
    connection_manager : needed to send the resume message to the socket
    resume_ttl_seconds : how long to hold a resumable slot before giving up
                         and retrying the task fresh (default 5 min)
    """

    def __init__(
        self,
        job_manager,
        task_dispatcher,
        checkpoint_manager,
        connection_manager,
        resume_ttl_seconds: int = 300,
    ):
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher
        self.checkpoint_manager = checkpoint_manager
        self.connection_manager = connection_manager
        self.resume_ttl_seconds = resume_ttl_seconds

        # worker_id -> DisconnectedWorkerRecord
        self._disconnected: dict[str, DisconnectedWorkerRecord] = {}

        # Background task that expires stale resumable slots
        self._expiry_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_worker_disconnected(self, worker_id: str) -> None:
        """
        Call this whenever a worker WebSocket closes (expected or unexpected).

        Determines whether the worker had an active task, checks if
        checkpointing was enabled, and either:
          - immediately re-queues the task (no checkpoint), or
          - parks the task in the resumable registry (checkpoint present).
        """
        # What task was this worker running?
        task_id, job_id = self._get_active_task(worker_id)

        if not task_id:
            logger.info("Worker %s disconnected with no active task – nothing to recover.", worker_id)
            self._cleanup_worker(worker_id)
            return

        # Is checkpointing enabled for this task/job?
        checkpointing_on = await self._is_checkpointing_enabled(job_id)
        latest_checkpoint_id = None

        if checkpointing_on:
            latest_checkpoint_id = await self._get_latest_checkpoint_id(task_id, job_id)

        if checkpointing_on and latest_checkpoint_id:
            # Park – wait for this worker to reconnect
            record = DisconnectedWorkerRecord(
                worker_id=worker_id,
                task_id=task_id,
                job_id=job_id,
                checkpointing_enabled=True,
                latest_checkpoint_id=latest_checkpoint_id,
            )
            self._disconnected[worker_id] = record
            logger.info(
                "Worker %s disconnected mid-task %s (job %s). "
                "Checkpoint %s parked – waiting for reconnect (TTL=%ds).",
                worker_id, task_id, job_id, latest_checkpoint_id, self.resume_ttl_seconds,
            )
        else:
            # No useful checkpoint – re-queue immediately so another worker picks it up
            logger.info(
                "Worker %s disconnected mid-task %s (job %s). "
                "No checkpoint available – re-queuing task.",
                worker_id, task_id, job_id,
            )
            await self._requeue_task(task_id, job_id, worker_id, reason="worker_disconnected_no_checkpoint")

        self._cleanup_worker(worker_id)
        self._ensure_expiry_loop()

    async def handle_worker_reconnected(
        self, worker_id: str, websocket: WebSocketServerProtocol
    ) -> bool:
        """
        Call this at the TOP of ``_handle_worker_ready`` before any other logic.

        Returns
        -------
        True  – a RESUME_TASK message was dispatched; caller should skip
                normal new-task assignment for this worker.
        False – nothing to resume; caller proceeds normally.
        """
        record = self._disconnected.pop(worker_id, None)

        if record is None:
            return False  # Not a reconnect we care about

        logger.info(
            "Worker %s reconnected. Found parked task %s (job %s); attempting resume.",
            worker_id,
            record.task_id,
            record.job_id,
        )

        if record.is_expired(self.resume_ttl_seconds):
            logger.warning(
                "Worker %s reconnected but its resumable slot for task %s expired. "
                "Re-queuing for fresh retry.",
                worker_id, record.task_id,
            )
            await self._requeue_task(record.task_id, record.job_id, worker_id, reason="resume_ttl_expired")
            return False

        # Fetch the full checkpoint payload
        checkpoint_payload = await self._fetch_checkpoint_payload(
            record.task_id, record.job_id, record.latest_checkpoint_id
        )

        if checkpoint_payload is None:
            logger.error(
                "Worker %s reconnected but checkpoint %s for task %s could not be loaded. "
                "Re-queuing for fresh retry.",
                worker_id, record.latest_checkpoint_id, record.task_id,
            )
            await self._requeue_task(record.task_id, record.job_id, worker_id, reason="checkpoint_load_failed")
            return False

        # All good – send resume message
        await self._send_resume_task(
            websocket=websocket,
            worker_id=worker_id,
            task_id=record.task_id,
            job_id=record.job_id,
            checkpoint_id=record.latest_checkpoint_id,
            checkpoint_payload=checkpoint_payload,
        )
        logger.info(
            "Worker %s resumed task %s (job %s) from checkpoint %s.",
            worker_id, record.task_id, record.job_id, record.latest_checkpoint_id,
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_active_task(self, worker_id: str) -> tuple[Optional[str], Optional[str]]:
        """Return (task_id, job_id) the worker was running, or (None, None)."""
        try:
            return self.connection_manager.get_worker_active_task(worker_id)
        except AttributeError:
            # Fallback: connection_manager may not expose this yet – see note below
            logger.debug("connection_manager.get_worker_active_task not available; no task recovered.")
            return None, None

    def _cleanup_worker(self, worker_id: str) -> None:
        """Release connection-manager state for this worker."""
        try:
            self.connection_manager.remove_worker(worker_id)
        except Exception as exc:
            logger.debug("Could not remove worker %s from connection_manager: %s", worker_id, exc)

    async def _is_checkpointing_enabled(self, job_id: str) -> bool:
        try:
            meta = self.job_manager.get_task_metadata(job_id)
            return bool(meta and meta.get("checkpoint_enabled"))
        except Exception as exc:
            logger.debug("Could not fetch task metadata for job %s: %s", job_id, exc)
            return False

    async def _get_latest_checkpoint_id(self, task_id: str, job_id: str) -> Optional[str]:
        try:
            from foreman.db.base import get_db_session
            from foreman.db.models import TaskModel
            async with get_db_session() as session:
                task = await session.get(TaskModel, task_id)
                if not task:
                    return None

                # checkpoint_count is monotonically updated on each stored checkpoint
                # and works as the latest checkpoint identifier in this codebase.
                if not task.base_checkpoint_data:
                    return None
                if not task.checkpoint_count or task.checkpoint_count <= 0:
                    return None

                return str(task.checkpoint_count)
        except Exception as exc:
            logger.warning("Could not get latest checkpoint id for task %s: %s", task_id, exc)
            return None

    async def _fetch_checkpoint_payload(
        self, task_id: str, job_id: str, checkpoint_id: str
    ) -> Optional[dict]:
        """Return the full reconstructed checkpoint dict, or None on error."""
        try:
            from foreman.db.base import get_db_session
            from common.serializer import bytes_to_hex
            async with get_db_session() as session:
                data_bytes = await self.checkpoint_manager.reconstruct_state(
                    session=session, task_id=task_id
                )
            if data_bytes is None:
                return None
            return {
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "job_id": job_id,
                "delta_data_hex": bytes_to_hex(data_bytes),
                "is_base": True,  # Reconstructed checkpoint is always a full snapshot
            }
        except Exception as exc:
            logger.error("Error reconstructing checkpoint for task %s: %s", task_id, exc, exc_info=True)
            return None

    async def _send_resume_task(
        self,
        websocket: WebSocketServerProtocol,
        worker_id: str,
        task_id: str,
        job_id: str,
        checkpoint_id: str,
        checkpoint_payload: dict,
    ) -> None:
        from common.protocol import Message, MessageType
        msg = Message(
            msg_type=MessageType.RESUME_TASK,
            data={
                "task_id": task_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_payload": checkpoint_payload,
            },
            job_id=job_id,
        )
        self.connection_manager.set_worker_active_task(worker_id, task_id, job_id)
        self.connection_manager.mark_worker_busy(worker_id)
        try:
            await websocket.send(msg.to_json())
            await _update_worker_status(worker_id, "busy", current_task_id=task_id)
        except Exception:
            self.connection_manager.clear_worker_active_task(worker_id)
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)
            raise

    async def _requeue_task(
        self, task_id: str, job_id: str, worker_id: str, reason: str
    ) -> None:
        """Mark the task failed (triggers retry) and attempt immediate re-dispatch."""
        try:
            await self.job_manager.mark_task_failed(
                task_id, job_id, worker_id, f"worker_disconnected:{reason}"
            )
            await self.task_dispatcher.assign_tasks_for_job(job_id, func_code="", args_list=[])
            logger.info("Task %s re-queued (reason=%s).", task_id, reason)
        except Exception as exc:
            logger.error("Failed to re-queue task %s: %s", task_id, exc, exc_info=True)

    # ------------------------------------------------------------------
    # TTL expiry loop
    # ------------------------------------------------------------------

    def _ensure_expiry_loop(self) -> None:
        if self._expiry_task is None or self._expiry_task.done():
            self._expiry_task = asyncio.create_task(self._expiry_loop())

    async def _expiry_loop(self) -> None:
        """Periodically expire stale reconnect slots."""
        while self._disconnected:
            await asyncio.sleep(30)
            expired = [
                w for w, r in self._disconnected.items()
                if r.is_expired(self.resume_ttl_seconds)
            ]
            for worker_id in expired:
                record = self._disconnected.pop(worker_id)
                logger.warning(
                    "Resume TTL expired for worker %s / task %s. Re-queuing.",
                    worker_id, record.task_id,
                )
                await self._requeue_task(
                    record.task_id, record.job_id, worker_id, reason="ttl_expired_background"
                )
