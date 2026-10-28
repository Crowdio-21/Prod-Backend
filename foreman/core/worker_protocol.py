"""Foreman side of the v2 worker protocol: versioned envelopes, acks, timeouts, retries.

Every command carries a ``msg_id``. The worker answers with ``msg_ack``
(``load`` is answered by ``load_result`` instead). Unacknowledged commands are
retried with the same ``msg_id`` so workers can de-duplicate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol

from foreman.planning.contract import make_envelope

logger = logging.getLogger(__name__)


class Transport(Protocol):
    async def send_worker(self, worker_id: str, message: dict) -> bool: ...
    async def send_client(self, client_key: Any, message: dict) -> bool: ...


class WebSocketTransport:
    """Live transport over the Foreman's existing WebSocket connections."""

    def __init__(self, connection_manager) -> None:
        self.connection_manager = connection_manager
        self.bytes_sent = 0

    async def send_worker(self, worker_id: str, message: dict) -> bool:
        websocket = self.connection_manager.get_worker_websocket(worker_id)
        if websocket is None:
            return False
        text = json.dumps(message)
        try:
            await websocket.send(text)
        except Exception as exc:  # connection dropped mid-send
            logger.warning("send to worker %s failed: %s", worker_id, exc)
            return False
        self.bytes_sent += len(text)
        return True

    async def send_client(self, client_key: Any, message: dict) -> bool:
        text = json.dumps(message)
        try:
            await client_key.send(text)
        except Exception as exc:
            logger.warning("send to client failed: %s", exc)
            return False
        self.bytes_sent += len(text)
        return True


@dataclass
class _Pending:
    worker_id: str
    message: dict
    attempts: int
    future: asyncio.Future
    task: Optional[asyncio.Task] = None


class ReliableSender:
    def __init__(
        self,
        transport: Transport,
        *,
        timeout_s: float = 5.0,
        retries: int = 2,
        on_failure: Optional[Callable[[str, dict], Awaitable[None]]] = None,
        on_bytes: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        self.transport = transport
        self.timeout_s = timeout_s
        self.retries = retries
        self.on_failure = on_failure
        self.on_bytes = on_bytes
        self._pending: Dict[str, _Pending] = {}
        self.control_bytes = 0
        self.control_messages = 0

    def _account(self, worker_id: str, message: dict) -> None:
        size = len(json.dumps(message))
        if message.get("type") not in ("run", "activation"):
            self.control_bytes += size
            self.control_messages += 1
        if self.on_bytes is not None:
            self.on_bytes(worker_id, size)

    async def send(self, worker_id: str, msg_type: str, body: Any, *, epoch: Optional[int] = None,
                   reliable: bool = True) -> Optional[asyncio.Future]:
        msg_id = uuid.uuid4().hex[:16]
        message = make_envelope(msg_type, body, msg_id=msg_id, epoch=epoch)
        self._account(worker_id, message)
        if not reliable:
            await self.transport.send_worker(worker_id, message)
            return None
        future = asyncio.get_running_loop().create_future()
        pending = _Pending(worker_id, message, 0, future)
        self._pending[msg_id] = pending
        pending.task = asyncio.create_task(self._drive(msg_id))
        return future

    async def _drive(self, msg_id: str) -> None:
        pending = self._pending.get(msg_id)
        while pending is not None and not pending.future.done():
            if pending.attempts > self.retries:
                self._pending.pop(msg_id, None)
                if not pending.future.done():
                    pending.future.set_result(None)
                if self.on_failure is not None:
                    await self.on_failure(pending.worker_id, pending.message)
                return
            pending.attempts += 1
            if pending.attempts > 1:
                self._account(pending.worker_id, pending.message)
            await self.transport.send_worker(pending.worker_id, pending.message)
            try:
                await asyncio.wait_for(asyncio.shield(pending.future), self.timeout_s)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

    def acknowledge(self, msg_id: Optional[str], result: Any = True) -> bool:
        pending = self._pending.pop(msg_id or "", None)
        if pending is None:
            return False
        if not pending.future.done():
            pending.future.set_result(result)
        return True

    def acknowledge_where(self, predicate: Callable[[dict], bool], result: Any = True) -> int:
        matched = [mid for mid, p in self._pending.items() if predicate(p.message)]
        for mid in matched:
            self.acknowledge(mid, result)
        return len(matched)

    def cancel_worker(self, worker_id: str) -> None:
        for mid in [m for m, p in self._pending.items() if p.worker_id == worker_id]:
            pending = self._pending.pop(mid)
            if pending.task is not None:
                pending.task.cancel()
            if not pending.future.done():
                pending.future.set_result(None)

    async def close(self) -> None:
        for pending in list(self._pending.values()):
            if pending.task is not None:
                pending.task.cancel()
        self._pending.clear()
