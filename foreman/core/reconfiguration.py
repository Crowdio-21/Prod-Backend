"""Live coordinator and WebSocket transport for the Foreman policy."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .foreman_policy import (
    Assignment, AssignmentRecord, BreakEvenPolicy, DecisionKind, LatencyPredictor,
    ReconfigurationExecutor, StageProfile, SystemState, WorkerRegistry,
    WorkerTelemetry,
)
from .reconfiguration_protocol import (
    ASSIGN, CUTOVER, DRAIN, LOAD_COMPLETE, PREFETCH, PREFETCH_DONE, UNLOAD,
    VERIFY, VERIFY_RESULT, ReconfigurationMessage,
)

logger = logging.getLogger(__name__)


class WebSocketReconfigurationTransport:
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager

    async def _send(self, worker_id: str, message: ReconfigurationMessage) -> None:
        websocket = self.connection_manager.get_worker_websocket(worker_id)
        if websocket is None:
            raise ConnectionError(f"Worker {worker_id} is not connected")
        await websocket.send(json.dumps(message.to_dict()))

    async def _request(self, worker_id: str, message: ReconfigurationMessage) -> Mapping[str, float]:
        await self._send(worker_id, message)
        return {}

    async def prefetch(self, worker_id: str, shard: str) -> None:
        await self._request(worker_id, ReconfigurationMessage(PREFETCH, {"shard": shard}))

    async def assign(self, worker_id: str, shard: str, epoch: int) -> Mapping[str, float]:
        return await self._request(
            worker_id, ReconfigurationMessage(ASSIGN, {"shard": shard}, epoch)
        )

    async def cutover(self, assignment: Assignment, epoch: int) -> None:
        # Routing is committed by the coordinator before workers are drained.
        await asyncio.gather(*(
            self._send(worker, ReconfigurationMessage(CUTOVER, {
                "assignment": dict(assignment.worker_by_partition),
                "cut": assignment.cut,
            }, epoch))
            for worker in set(assignment.worker_by_partition.values())
        ))

    async def drain(self, worker_id: str) -> None:
        await self._send(worker_id, ReconfigurationMessage(DRAIN))

    async def unload(self, worker_id: str, shard: str) -> None:
        await self._send(worker_id, ReconfigurationMessage(UNLOAD, {"shard": shard}))

    async def verify(self, assignment: Assignment, epoch: int) -> None:
        await asyncio.gather(*(
            self._send(worker, ReconfigurationMessage(VERIFY, {
                "assignment": dict(assignment.worker_by_partition),
            }, epoch))
            for worker in set(assignment.worker_by_partition.values())
        ))


@dataclass
class ReconfigurationMetrics:
    moves: int = 0
    reversals: int = 0
    last_predicted_ready: float = 0.0
    last_realized_ready: float = 0.0
    last_predicted_gain: float = 0.0
    last_realized_gain: float = 0.0
    wasted_seconds: float = 0.0
    _last_assignment: Assignment | None = None
    _last_move_at: float = 0.0


class ForemanCoordinator:
    """Clock-injected live coordinator; ``tick`` can also be called by a simulator."""

    def __init__(self, connection_manager, tick_seconds: float = 2.0):
        self.registry = WorkerRegistry()
        self.connection_manager = connection_manager
        self.transport = WebSocketReconfigurationTransport(connection_manager)
        self.executor = ReconfigurationExecutor(self.transport)
        self.tick_seconds = tick_seconds
        self.policy: BreakEvenPolicy | None = None
        self.state: SystemState | None = None
        self.metrics = ReconfigurationMetrics()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def configure(
        self,
        profiles: Mapping[str, StageProfile],
        assignment: Assignment,
        *,
        n_remaining: float = 0.0,
        arrival_rate: float = 0.0,
        stream_horizon_seconds: float = 60.0,
    ) -> None:
        predictor = LatencyPredictor(profiles)
        record = AssignmentRecord(assignment)
        self.state = SystemState(
            workers=self.registry.workers,
            profiles=profiles,
            current=record,
            n_remaining=n_remaining,
            arrival_rate=arrival_rate,
            stream_horizon_seconds=stream_horizon_seconds,
        )
        self.policy = BreakEvenPolicy(predictor)

    def record_heartbeat(self, telemetry: WorkerTelemetry, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else now
        self.registry.heartbeat(telemetry, timestamp)
        if self.state is not None:
            self.state = SystemState(**{**self.state.__dict__, "workers": self.registry.workers, "now": timestamp})

    async def tick(self, now: float | None = None) -> None:
        async with self._lock:
            if self.state is None or self.policy is None:
                return
            timestamp = time.monotonic() if now is None else now
            departed = self.registry.departed(timestamp, self.state.heartbeat_interval)
            if departed:
                workers = {key: value for key, value in self.registry.workers.items() if key not in departed}
                self.state = SystemState(**{**self.state.__dict__, "workers": workers, "now": timestamp})
            decision = self.policy.decide(self.state)
            if decision.kind != DecisionKind.REASSIGN or decision.plan is None:
                return
            started = time.monotonic()
            old = self.state.current
            try:
                updated = await self.executor.execute(old, decision.plan, self.state.workers, timestamp)
            except Exception:
                self.metrics.wasted_seconds += time.monotonic() - started
                logger.exception("Shard reconfiguration failed; retaining epoch %s", old.epoch)
                return
            realized = time.monotonic() - started
            if self.metrics._last_assignment and updated.assignment == self.metrics._last_assignment:
                self.metrics.reversals += 1
            self.metrics.moves += 1
            self.metrics.last_predicted_ready = decision.ready_time
            self.metrics.last_realized_ready = realized
            self.metrics.last_predicted_gain = decision.predicted_gain
            self.metrics._last_assignment = old.assignment
            self.metrics._last_move_at = timestamp
            self.state = SystemState(**{**self.state.__dict__, "current": updated, "now": timestamp})

    async def start(self) -> None:
        if self._task is not None:
            return
        async def loop():
            while True:
                await asyncio.sleep(self.tick_seconds)
                await self.tick()
        self._task = asyncio.create_task(loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
