"""Pure, cost-aware Foreman policy and shard reconfiguration primitives.

The policy deliberately has no clock, I/O, or WebSocket dependency.  A live
Foreman and a simulator can therefore build the same :class:`SystemState` and
call :meth:`BreakEvenPolicy.decide` unchanged.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from math import inf
from typing import Awaitable, Callable, Deque, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple


class Objective(str, Enum):
    LATENCY = "latency"
    PIPELINED_THROUGHPUT = "pipelined_throughput"


class DecisionKind(str, Enum):
    STAY = "stay"
    REASSIGN = "reassign"


@dataclass(frozen=True)
class Assignment:
    worker_by_partition: Mapping[str, str]
    cut: int = 0


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    plan: Optional[Assignment] = None
    net_benefit: float = 0.0
    predicted_gain: float = 0.0
    ready_time: float = 0.0

    @classmethod
    def stay(cls) -> "Decision":
        return cls(DecisionKind.STAY)


@dataclass
class WorkerTelemetry:
    worker_id: str
    device_class: str = "unknown"
    last_heartbeat: float = 0.0
    bandwidth_mean: float = 10.0
    bandwidth_std: float = 0.0
    rtt_seconds: float = 0.05
    throughput_share: float = 1.0
    disk_shards: set[str] = field(default_factory=set)
    stage_history: Dict[Tuple[str, str], Deque[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class StageProfile:
    shard: str
    size_bytes: int
    stage_seconds: float
    init_seconds: float = 0.0
    warmup_seconds: float = 0.0
    io_seconds: float = 0.0
    cutover_seconds: float = 0.0


@dataclass
class AssignmentRecord:
    assignment: Assignment
    last_move_by_worker: Dict[str, float] = field(default_factory=dict)
    moves_in_flight: set[str] = field(default_factory=set)
    epoch: int = 0


@dataclass(frozen=True)
class MoveCostModel:
    drain_seconds: float = 0.5
    bandwidth_min_bytes_per_second: float = 1.0
    bandwidth_k: float = 1.0
    min_bandwidth_bytes_per_second: float = 1.0
    max_concurrent_downloads: int = 1
    default_init_seconds: float = 0.0
    default_warmup_seconds: float = 0.0
    default_cutover_seconds: float = 0.0

    def ready_time(
        self,
        current: Assignment,
        candidate: Assignment,
        workers: Mapping[str, WorkerTelemetry],
        profiles: Mapping[str, StageProfile],
    ) -> float:
        durations = []
        for shard, target in candidate.worker_by_partition.items():
            if current.worker_by_partition.get(shard) == target:
                continue
            worker = workers[target]
            profile = profiles[shard]
            bandwidth = max(
                worker.bandwidth_mean - self.bandwidth_k * worker.bandwidth_std,
                self.min_bandwidth_bytes_per_second,
            ) / max(1, self.max_concurrent_downloads)
            prefetch = 0.0 if shard in worker.disk_shards else profile.size_bytes / bandwidth
            durations.append(
                self.drain_seconds
                + prefetch
                + profile.io_seconds
                + (profile.init_seconds or self.default_init_seconds)
                + (profile.warmup_seconds or self.default_warmup_seconds)
                + (profile.cutover_seconds or self.default_cutover_seconds)
                + worker.rtt_seconds
            )
        return max(durations, default=0.0)

    def downtime_loss(
        self,
        current: Assignment,
        candidate: Assignment,
        workers: Mapping[str, WorkerTelemetry],
        profiles: Mapping[str, StageProfile],
    ) -> float:
        loss = 0.0
        for shard, old_worker in current.worker_by_partition.items():
            if candidate.worker_by_partition.get(shard) == old_worker:
                continue
            profile = profiles[shard]
            worker = workers[old_worker]
            loss += (self.drain_seconds + profile.cutover_seconds) * worker.throughput_share
        return loss


@dataclass(frozen=True)
class SystemState:
    workers: Mapping[str, WorkerTelemetry]
    profiles: Mapping[str, StageProfile]
    current: AssignmentRecord
    n_remaining: float = 0.0
    arrival_rate: float = 0.0
    stream_horizon_seconds: float = 60.0
    objective: Objective = Objective.LATENCY
    now: float = 0.0
    heartbeat_interval: float = 10.0
    missed_heartbeats: int = 3
    persistence_ticks: int = 1
    persisted_ticks: int = 1
    cooldown_seconds: float = 10.0
    margin_fraction: float = 0.10
    move_cost: MoveCostModel = field(default_factory=MoveCostModel)

    @property
    def workload(self) -> float:
        return max(0.0, self.n_remaining or self.arrival_rate * self.stream_horizon_seconds)


class LatencyPredictor:
    """EWMA stage predictor, initialised from manifest values."""

    def __init__(self, profiles: Mapping[str, StageProfile], alpha: float = 0.2):
        self.alpha = alpha
        self._values = {(w, s): p.stage_seconds for w in [] for s, p in profiles.items()}
        self._profiles = dict(profiles)

    def observe(self, device_class: str, shard: str, seconds: float) -> None:
        key = (device_class, shard)
        old = self._values.get(key, self._profiles[shard].stage_seconds)
        self._values[key] = self.alpha * seconds + (1 - self.alpha) * old

    def predict(self, worker: WorkerTelemetry, shard: str) -> float:
        return self._values.get((worker.device_class, shard), self._profiles[shard].stage_seconds)


class WorkerRegistry:
    """Clock-injected heartbeat registry and departure detector."""

    def __init__(self, missed_heartbeats: int = 3):
        self.missed_heartbeats = missed_heartbeats
        self.workers: Dict[str, WorkerTelemetry] = {}

    def heartbeat(self, telemetry: WorkerTelemetry, now: float) -> None:
        telemetry.last_heartbeat = now
        self.workers[telemetry.worker_id] = telemetry

    def departed(self, now: float, interval: float) -> set[str]:
        threshold = self.missed_heartbeats * interval
        return {wid for wid, worker in self.workers.items() if now - worker.last_heartbeat > threshold}


class BreakEvenPolicy:
    """Conservative break-even policy with persistence and cooldown checks."""

    def __init__(self, predictor: LatencyPredictor):
        self.predictor = predictor

    def decide(self, state: SystemState) -> Decision:
        current = state.current.assignment
        best = current
        best_score = self._score(current, state)
        for candidate in enumerate_assignments(tuple(state.profiles), tuple(state.workers), current.cut):
            score = self._score(candidate, state)
            if score < best_score:
                best, best_score = candidate, score
        if best == current or not self._eligible(best, state):
            return Decision.stay()

        # Candidate latency is pessimistically high, current latency low.
        current_latency = self._score(current, state, pessimistic=False)
        candidate_latency = self._score(best, state, pessimistic=True)
        gain = max(0.0, current_latency - candidate_latency)
        ready = state.move_cost.ready_time(current, best, state.workers, state.profiles)
        effective_work = max(0.0, state.workload - state.arrival_rate * ready)
        benefit = gain * effective_work - state.move_cost.downtime_loss(
            current, best, state.workers, state.profiles
        )
        margin = state.margin_fraction * max(1.0, state.move_cost.downtime_loss(
            current, best, state.workers, state.profiles
        ))
        return (
            Decision(DecisionKind.REASSIGN, best, benefit, gain, ready)
            if benefit > margin
            else Decision.stay()
        )

    def _eligible(self, candidate: Assignment, state: SystemState) -> bool:
        changed = set(candidate.worker_by_partition.values()) - set(state.current.assignment.worker_by_partition.values())
        if state.current.moves_in_flight.intersection(changed):
            return False
        if state.persisted_ticks < state.persistence_ticks:
            return False
        return all(
            state.now - state.current.last_move_by_worker.get(worker, -inf) >= state.cooldown_seconds
            for worker in changed
        )

    def _score(self, assignment: Assignment, state: SystemState, pessimistic: bool = False) -> float:
        stage_times = []
        for shard, worker_id in assignment.worker_by_partition.items():
            worker = state.workers[worker_id]
            estimate = self.predictor.predict(worker, shard)
            stage_times.append(estimate * (1.10 if pessimistic else 0.90))
        if not stage_times:
            return inf
        return max(stage_times) if state.objective == Objective.PIPELINED_THROUGHPUT else sum(stage_times)


def enumerate_assignments(
    partitions: Sequence[str], workers: Sequence[str], cut: int = 0
) -> Iterable[Assignment]:
    if len(workers) < len(partitions):
        return
    for selected in itertools.permutations(workers, len(partitions)):
        yield Assignment(dict(zip(partitions, selected)), cut)


class ReconfigurationTransport(Protocol):
    async def prefetch(self, worker_id: str, shard: str) -> None: ...
    async def assign(self, worker_id: str, shard: str, epoch: int) -> Mapping[str, float]: ...
    async def cutover(self, assignment: Assignment, epoch: int) -> None: ...
    async def drain(self, worker_id: str) -> None: ...
    async def unload(self, worker_id: str, shard: str) -> None: ...
    async def verify(self, assignment: Assignment, epoch: int) -> None: ...


class ReconfigurationExecutor:
    """Shared make-before-break/break-before-make execution sequence."""

    def __init__(self, transport: ReconfigurationTransport, max_ready_seconds: float = 300.0):
        self.transport = transport
        self.max_ready_seconds = max_ready_seconds

    async def execute(
        self,
        current: AssignmentRecord,
        target: Assignment,
        workers: Mapping[str, WorkerTelemetry],
        now: Optional[float] = None,
    ) -> AssignmentRecord:
        epoch = current.epoch + 1
        changed = [(shard, current.assignment.worker_by_partition.get(shard), worker)
                   for shard, worker in target.worker_by_partition.items()
                   if current.assignment.worker_by_partition.get(shard) != worker]
        serving_targets = {
            worker for _, _, worker in changed
            if worker in current.assignment.worker_by_partition.values()
        }
        started = time.monotonic()
        try:
            # Single residency requires break-before-make when the target is
            # already serving another shard. Idle targets remain uninterrupted.
            for shard, old_worker, worker in changed:
                if worker in serving_targets and old_worker is not None:
                    await self.transport.drain(worker)
                    await self.transport.unload(worker, shard)
            await asyncio.gather(*(
                self.transport.prefetch(worker, shard)
                for shard, _, worker in changed
                if shard not in workers[worker].disk_shards
            ))
            for shard, old_worker, worker in changed:
                await asyncio.wait_for(
                    self.transport.assign(worker, shard, epoch),
                    timeout=max(0.0, self.max_ready_seconds - (time.monotonic() - started)),
                )
            await self.transport.cutover(target, epoch)
            for shard, old_worker, worker in changed:
                if old_worker is not None and worker not in serving_targets:
                    await self.transport.drain(old_worker)
                    await self.transport.unload(old_worker, shard)
            await self.transport.verify(target, epoch)
            moved = dict(current.last_move_by_worker)
            timestamp = time.monotonic() if now is None else now
            for _, old_worker, worker in changed:
                moved[worker] = timestamp
                if old_worker:
                    moved[old_worker] = timestamp
            return AssignmentRecord(target, moved, set(), epoch)
        except Exception:
            # The transport owns the atomic routing rollback; never publish a
            # partially committed assignment.
            raise
