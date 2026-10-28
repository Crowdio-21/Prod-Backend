"""Live, reconfiguration-aware plan coordinator (behind ``adaptive_planning``).

The Foreman is the only party that commits a plan. This module:

* ingests telemetry (v2 ``report`` and today's heartbeats) into the TelemetryStore;
* builds the request DAG from the chosen Plan (``submit_inference``), dispatching
  ``run`` messages and forwarding activations on the relay route;
* runs the pure controller each tick and executes switches with the epoch
  protocol. Voluntary switches **drain** (in-flight requests finish under the
  old epoch first); forced switches **resume** in-flight requests from the
  last retained stage-boundary activation when the new plan keeps that cut,
  otherwise restart them from their input. A worker never receives ``load``
  for a new shard while it still has in-flight work of the old epoch;
* handles problem reports, prefetch, peer fetch, probes and route selection.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from foreman.planning.config import RunConfig
from foreman.planning.contract import (
    URGENT_PROBLEMS,
    AckMsg,
    ActivationMsg,
    CancelPrefetchMsg,
    Envelope,
    LoadMsg,
    LoadResultMsg,
    NeedMsg,
    NextHop,
    Plan,
    PrefetchMsg,
    ProbeMsg,
    ProblemMsg,
    ProblemType,
    ReportMsg,
    RouteMode,
    RouteMsg,
    RunMsg,
    ServeAllowedMsg,
    SubmitInferenceMsg,
    WorkerState,
    make_envelope,
)
from foreman.planning.controller import Baseline, ControllerState, decide, detect_triggers
from foreman.planning.cost_model import CostOptions, c_switch, pipeline_cost
from foreman.planning.eventlog import EventLog, stable_hash
from foreman.planning.planner import Objective
from foreman.planning.policies import build_plan, current_cost, make_policy
from foreman.planning.prefetch import plan_prefetch
from foreman.planning.problems import ProblemAggregator
from foreman.planning.profiles import ProfileLearner
from foreman.planning.registry import ShardRegistry
from foreman.planning.serialize import opts_to_dict, snapshot_to_dict
from foreman.planning.routes import CapabilityTracker, RouteSelector
from foreman.planning.telemetry import TelemetryStore

from .worker_protocol import ReliableSender, Transport

logger = logging.getLogger(__name__)


@dataclass
class Request:
    request_id: str
    job_id: str
    input: Dict[str, Any]
    epoch: int
    started_at: float
    # (end_atom of the producing shard, tensor) of the latest retained boundary activation
    retained: Optional[Tuple[int, Dict[str, Any]]] = None
    computing_on: Optional[str] = None  # worker currently running a stage (relay-visible)
    computing_stage_atom: Optional[int] = None  # start atom of the stage being computed
    dispatched_at: float = 0.0
    restarts: int = 0


@dataclass
class Job:
    job_id: str
    client: Any
    request_ids: List[str]
    deadline_s: Optional[float]
    submitted_at: float
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingSwitch:
    target: Tuple[Tuple[str, str], ...]
    replicas: Tuple[Tuple[str, str], ...]
    mode: str  # drain | resume
    reason: str
    epoch: int
    started_at: float
    phase: str = "draining"  # draining -> loading -> committed
    changed: Dict[str, str] = field(default_factory=dict)  # worker -> new shard
    loads_pending: Set[str] = field(default_factory=set)
    load_started: Dict[str, float] = field(default_factory=dict)
    sources: Dict[str, Optional[str]] = field(default_factory=dict)


class _Bounded(OrderedDict):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def add(self, key) -> bool:
        if key in self:
            return False
        self[key] = True
        while len(self) > self.limit:
            self.popitem(last=False)
        return True


class PlanCoordinator:
    def __init__(
        self,
        config: RunConfig,
        registry: ShardRegistry,
        transport: Transport,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Optional[EventLog] = None,
        msg_timeout_s: float = 5.0,
        request_timeout_s: float = 120.0,
        window: Optional[int] = None,
        tick_seconds: float = 1.0,
        probe_url: Optional[str] = None,
        log_snapshots: bool = True,
    ) -> None:
        self.config = config
        self.flags = config.flags
        self.registry = registry
        self.transport = transport
        self.clock = clock
        self.log = log or EventLog(config.log_path, run_id=config.run_id, seed=config.seed,
                                   policy=config.policy, flags=config.flags.to_dict(), clock=time.time)
        self.telemetry = TelemetryStore(config.estimator, clock=clock, on_update=self._on_estimate)
        self.learner = ProfileLearner()
        self.policy = make_policy(config.policy, CostOptions.from_config(config), config.planner_mode)
        self.objective = Objective(config.objective, config.deadline_s)
        self.state = ControllerState()
        self.routes = RouteSelector(config.routes)
        self.capability = CapabilityTracker(config.routes)
        self.problems = ProblemAggregator()
        self.sender = ReliableSender(transport, timeout_s=msg_timeout_s, on_failure=self._on_send_failure)
        self.request_timeout_s = request_timeout_s
        self.window = window
        self.tick_seconds = tick_seconds
        self.probe_url = probe_url
        self.log_snapshots = log_snapshots

        self.plan: Optional[Plan] = None
        self.epoch = 0
        self.pending: Optional[PendingSwitch] = None
        self.excluded: Set[Tuple[str, str]] = set()
        self.urgent: Set[str] = set()
        self.queue: Deque[str] = deque()
        self.requests: Dict[str, Request] = {}
        self.jobs: Dict[str, Job] = {}
        self.completed = _Bounded(100_000)
        self.seen = _Bounded(100_000)
        self.connected: Set[str] = set()
        self.prefetching: Dict[Tuple[str, str], float] = {}
        self.prefetched: Set[Tuple[str, str]] = set()
        self.uploads: Dict[str, int] = {}
        self.last_decision = None
        self.counters: Dict[str, int] = {}
        self._task: Optional[asyncio.Task] = None
        self._tick_lock = asyncio.Lock()
        self.log.emit("run_start", config=config.to_dict(), flags=config.flags.to_dict())
        logger.info("Plan coordinator started; flags=%s policy=%s", config.flags.to_dict(), config.policy)

    # ------------------------------------------------------------------ util
    def _count(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def _on_estimate(self, worker_id: str, name: str, est) -> None:
        self.log.emit("estimate_update", worker=worker_id, metric=name, mean=est.mean, std=est.std, n=est.n,
                      age_s=est.age_s)

    def _opts(self, **overrides) -> CostOptions:
        critical = frozenset(self.plan.workers) if self.plan else frozenset()
        return CostOptions.from_config(self.config, critical_workers=critical, **overrides)

    def snapshot(self):
        return self.telemetry.snapshot(
            self.registry.catalog,
            self.learner.profiles(),
            excluded=frozenset(self.excluded),
            uploads_in_flight=self.uploads,
            default_compression=self.config.cost.compression_ratio,
            dnn_scope=self.config.dnn_workers,
        )

    def n_remaining(self) -> int:
        return len(self.queue) + sum(1 for r in self.requests.values() if r.request_id not in self.completed)

    def _stage_by_atom(self, plan: Plan, atom: int) -> Optional[int]:
        for stage in plan.stages:
            if self.registry.catalog.shards[stage.shard_id].start_atom == atom:
                return stage.stage_index
        return None

    def _end_atom(self, shard_id: str) -> int:
        return self.registry.catalog.shards[shard_id].end_atom

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self._task is None:
            async def loop():
                while True:
                    await asyncio.sleep(self.tick_seconds)
                    try:
                        await self.tick()
                    except Exception:
                        logger.exception("plan coordinator tick failed")
            self._task = asyncio.create_task(loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await self.sender.close()
        self.log.emit("run_end", counters=self.counters, control_bytes=self.sender.control_bytes,
                      control_messages=self.sender.control_messages, route_flips=self.routes.flips)
        self.log.close()

    def worker_connected(self, worker_id: str, device_class: Optional[str] = None,
                         kind: Optional[str] = None) -> None:
        self.connected.add(worker_id)
        if device_class:
            self.telemetry.ingest_state(WorkerState(worker_id=worker_id, device_class=device_class))
        self.telemetry.set_kind(worker_id, kind)
        self.capability.network_changed(worker_id)

    async def worker_disconnected(self, worker_id: str) -> None:
        self.connected.discard(worker_id)
        self.telemetry.mark_departed(worker_id)
        self.sender.cancel_worker(worker_id)
        self.log.emit("failure", kind="worker_lost", worker=worker_id)
        for key in [k for k in self.prefetching if k[0] == worker_id]:
            self.prefetching.pop(key, None)
        if self.plan is not None and worker_id in self.plan.workers:
            self.urgent.add("worker_lost")
            await self.tick()

    # ------------------------------------------------------------- ingestion
    def ingest_legacy_heartbeat(self, worker_id: str, data: dict) -> None:
        state = self.telemetry.ingest_legacy_heartbeat(worker_id, data)
        self.log.emit("heartbeat", worker=worker_id, source="legacy", state=state.model_dump(mode="json"))

    async def handle_envelope(self, sender: Any, raw: dict, *, worker_id: Optional[str] = None) -> None:
        """Entry point for every v2 message. ``worker_id`` is None for clients."""
        envelope = Envelope.model_validate(raw)
        body = envelope.body()
        kind = envelope.type
        if kind == "submit_inference":
            await self.submit(sender, body)
            return
        if worker_id is None:
            logger.warning("v2 %s from unknown sender ignored", kind)
            return
        if kind == "msg_ack":
            self.sender.acknowledge(body.get("ack_of"))
        elif kind == "report":
            await self._on_report(worker_id, body)
        elif kind == "load_result":
            await self._on_load_result(worker_id, body)
        elif kind == "problem":
            await self._on_problem(worker_id, body)
        elif kind == "need":
            await self._on_need(worker_id, body)
        elif kind == "activation":
            await self._on_activation(worker_id, body)
        elif kind == "ack":
            self._count("acks")
        elif kind == "probe_result":
            self._on_probe_result(worker_id, body)
        else:
            logger.warning("unexpected v2 message %s from worker %s", kind, worker_id)

    async def _on_report(self, worker_id: str, body: ReportMsg) -> None:
        state = body.state
        if state.worker_id != worker_id:
            state = state.model_copy(update={"worker_id": worker_id})
        self.connected.add(worker_id)
        self.telemetry.ingest_state(state)
        self.log.emit("heartbeat", worker=worker_id, source="v2", state=state.model_dump(mode="json"))
        pending = self.pending
        if pending is not None and pending.phase == "loading" and worker_id in pending.loads_pending                 and state.resident_shard == pending.changed.get(worker_id) and state.last_epoch == pending.epoch:
            # The worker already reports the target shard resident for the new epoch.
            pending.loads_pending.discard(worker_id)
            await self._progress_switch()
        on_disk = {d.id for d in state.disk_shards}
        for key in [k for k in self.prefetching if k[0] == worker_id and k[1] in on_disk]:
            started = self.prefetching.pop(key)
            self.prefetched.add(key)
            shard = self.registry.catalog.shards.get(key[1])
            self.log.emit("prefetch", worker=worker_id, shard=key[1], status="done",
                          bytes=shard.bytes if shard else 0, seconds=self.clock() - started)

    def _on_probe_result(self, worker_id: str, body: dict) -> None:
        size = int(body.get("size", 0))
        if body.get("peer"):
            self.capability.record(worker_id, body["peer"], bool(body.get("reachable")), self.clock())
            self.telemetry.set_capability(worker_id, body["peer"], bool(body.get("reachable")))
        else:
            if body.get("down_s"):
                self.telemetry.observe_transfer(worker_id, "down", size, float(body["down_s"]))
            if body.get("up_s"):
                self.telemetry.observe_transfer(worker_id, "up", size, float(body["up_s"]))
        self.log.emit("probe", worker=worker_id, result=body)

    # ------------------------------------------------------------- requests
    async def submit(self, client: Any, body: SubmitInferenceMsg) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = self.clock()
        ids = []
        for index, tensor in enumerate(body.inputs):
            request_id = f"{job_id}-{index}"
            self.requests[request_id] = Request(request_id, job_id, tensor, self.epoch, now)
            self.queue.append(request_id)
            ids.append(request_id)
        self.jobs[job_id] = Job(job_id, client, ids, body.deadline_s, now)
        await self.transport.send_client(client, make_envelope("inference_accepted", {"job_id": job_id, "n": len(ids)}))
        if self.plan is None and self.pending is None:
            self.urgent.add("initial")
        await self.tick()
        return job_id

    def _window(self) -> int:
        if self.window is not None:
            return self.window
        return max(2, 2 * (len(self.plan.stages) if self.plan else 1))

    async def _admit(self) -> None:
        if self.plan is None or self.pending is not None:
            return
        while self.queue and self._in_flight() < self._window():
            request_id = self.queue.popleft()
            request = self.requests[request_id]
            request.epoch = self.epoch
            self.log.emit("request_start", request_id=request_id, job_id=request.job_id)
            await self._dispatch(request, 0, request.input)

    def _in_flight(self) -> int:
        queued = set(self.queue)
        return sum(1 for r in self.requests if r not in queued and r not in self.completed)

    async def _dispatch(self, request: Request, stage_index: int, activation: Dict[str, Any]) -> None:
        stage = self.plan.stages[stage_index]
        request.computing_on = stage.worker_id
        request.computing_stage_atom = self.registry.catalog.shards[stage.shard_id].start_atom
        request.dispatched_at = self.clock()
        body = RunMsg(stage=stage_index, activation=activation, epoch=request.epoch, request_id=request.request_id)
        await self.sender.send(stage.worker_id, "run", body, epoch=request.epoch, reliable=False)
        self._count("runs_sent")

    async def _on_activation(self, worker_id: str, body: ActivationMsg) -> None:
        key = (body.request_id, body.epoch, body.stage)
        # Always ack so the producer can drop its retained copy (idempotent).
        await self.sender.send(worker_id, "ack", AckMsg(request_id=body.request_id, epoch=body.epoch,
                                                        stage=body.stage), epoch=body.epoch, reliable=False)
        if not self.seen.add(key):
            self._count("duplicates_dropped")
            self.log.emit("message", kind="duplicate_activation", worker=worker_id, key=list(key))
            return
        request = self.requests.get(body.request_id)
        if request is None or body.request_id in self.completed:
            self._count("late_activation_dropped")
            return
        if body.epoch != request.epoch:
            self._count("stale_epoch_dropped")
            self.log.emit("problem", type="WRONG_EPOCH", worker=worker_id, request_id=body.request_id,
                          got=body.epoch, expected=request.epoch)
            return
        plan = self.plan
        # Which shard produced it? Use the plan of that epoch if still current, else the worker's stage.
        producing = None
        if plan is not None and body.stage < len(plan.stages) and plan.stages[body.stage].worker_id == worker_id:
            producing = plan.stages[body.stage].shard_id
        elif request.computing_on == worker_id and request.computing_stage_atom is not None:
            producing = next((s.shard_id for s in self.registry.catalog.shards.values()
                              if s.start_atom == request.computing_stage_atom and plan and
                              any(st.shard_id == s.shard_id for st in plan.stages)), None)
        if producing is None and plan is not None:
            producing = plan.stages[min(body.stage, len(plan.stages) - 1)].shard_id
        if body.compute_ms is not None and producing is not None:
            self.telemetry.observe_compute(worker_id, producing, body.compute_ms / 1000.0)
        end_atom = self._end_atom(producing)
        request.retained = (end_atom, body.tensor)
        request.computing_on = None
        if end_atom == self.registry.catalog.n_atoms - 1:
            await self._complete(request, body.tensor)
            return
        holding = plan is None or (self.pending is not None and (
            self.pending.mode == "resume" or self.pending.phase != "draining"))
        if holding:
            return  # held at the Foreman until the switch commits
        await self._forward(request)

    async def _forward(self, request: Request) -> None:
        end_atom, tensor = request.retained
        next_index = self._stage_by_atom(self.plan, end_atom + 1)
        if next_index is None:
            # The new plan does not keep this cut: restart from the input.
            self.log.emit("message", kind="restart_from_input", request_id=request.request_id,
                          reason="cut not kept by plan")
            request.retained = None
            request.restarts += 1
            await self._dispatch(request, 0, request.input)
            return
        await self._dispatch(request, next_index, tensor)

    async def _complete(self, request: Request, tensor: Dict[str, Any]) -> None:
        if not self.completed.add(request.request_id):
            return
        now = self.clock()
        self.log.emit("request_end", request_id=request.request_id, job_id=request.job_id,
                      latency_s=now - request.started_at, restarts=request.restarts)
        job = self.jobs.get(request.job_id)
        if job is not None:
            job.results[request.request_id] = tensor
            if len(job.results) == len(job.request_ids):
                results = [{"request_id": rid, "tensor": job.results[rid]} for rid in job.request_ids]
                await self.transport.send_client(job.client, make_envelope(
                    "inference_result", {"job_id": job.job_id, "results": results}))
                for rid in job.request_ids:
                    self.requests.pop(rid, None)
                del self.jobs[job.job_id]
        await self._admit()

    # ------------------------------------------------------------- problems
    async def _on_problem(self, worker_id: str, body: ProblemMsg) -> None:
        if not self.flags.problem_reports and body.type not in URGENT_PROBLEMS:
            return
        verdict = self.problems.accept(worker_id, body, self.clock())
        self.log.emit("problem", worker=worker_id, type=body.type.value, stage=body.stage,
                      request_id=body.request_id, verdict=verdict.status, detail=body.detail)
        if verdict.status != "urgent":
            return
        if body.type is ProblemType.NEXT_HOP_DOWN:
            await self._reroute_next_hop(worker_id, body)
        elif body.type in (ProblemType.OOM, ProblemType.LOAD_FAILED):
            shard = body.detail.get("shard_id") or self._shard_of(worker_id)
            if shard:
                self.excluded.add((worker_id, shard))
            self.urgent.add("oom" if body.type is ProblemType.OOM else "load_failed")
            if self.pending is not None and worker_id in self.pending.changed:
                self._abort_switch("load failed during switch")
            await self.tick()

    def _shard_of(self, worker_id: str) -> Optional[str]:
        if self.plan is None:
            return None
        return next((s.shard_id for s in self.plan.stages if s.worker_id == worker_id), None)

    async def _reroute_next_hop(self, worker_id: str, body: ProblemMsg) -> None:
        """Peer path broke: fall back to the relay for that boundary; the worker resends."""
        if self.plan is None:
            return
        stage = next((s for s in self.plan.stages if s.worker_id == worker_id), None)
        if stage is None:
            return
        nxt = stage.next_hop.worker_id
        if nxt is not None and nxt in self.connected:
            choice = self.routes.fallback(worker_id, nxt, self.clock(), "next hop down")
            self.log.emit("route_flip", src=worker_id, dst=nxt, mode=choice.mode.value, reason=choice.reason)
            self.telemetry.set_capability(worker_id, nxt, False)
            self.capability.record(worker_id, nxt, False, self.clock())
            await self._send_route(stage.stage_index, worker_id,
                                   NextHop(worker_id=nxt, allowed_routes=[RouteMode.VIA_FOREMAN]))
        else:
            self.urgent.add("next_hop_down")
            await self.tick()

    async def _on_need(self, worker_id: str, body: NeedMsg) -> None:
        self.log.emit("problem", worker=worker_id, type="NEED", shard=body.shard_id, why=body.why)
        if self.plan is not None and self._shard_of(worker_id) == body.shard_id:
            await self.sender.send(worker_id, "load", LoadMsg(shard_id=body.shard_id, epoch=self.epoch),
                                   epoch=self.epoch)

    async def _on_send_failure(self, worker_id: str, message: dict) -> None:
        self.log.emit("failure", kind="command_timeout", worker=worker_id, type=message.get("type"))
        if message.get("type") == "load" and self.pending is not None and worker_id in self.pending.changed:
            self.excluded.add((worker_id, self.pending.changed[worker_id]))
            self._abort_switch("load command timed out")
            self.urgent.add("load_failed")

    # ------------------------------------------------------------- switching
    def _abort_switch(self, reason: str) -> None:
        if self.pending is not None:
            self.log.emit("switch_end", epoch=self.pending.epoch, ok=False, reason=reason)
            self.pending = None

    async def _start_switch(self, decision, forced: bool) -> None:
        stages = decision.target
        replicas = tuple(decision.decision.replicas) if decision.decision else ()
        epoch = self.epoch + 1
        mode = "resume" if forced else "drain"
        why = ("forced replan: in-flight work is resumed from retained boundary activations"
               if forced else "voluntary switch: in-flight requests finish under the old epoch")
        current = dict((w, s) for s, w in self.plan.assignment()) if self.plan else {}
        snap = self.snapshot()
        opts = self._opts()
        pending = PendingSwitch(stages, replicas, mode, decision.reason, epoch, self.clock())
        for shard_id, worker_id in list(stages) + list(replicas):
            if current.get(worker_id) != shard_id and snap.workers.get(worker_id) is not None \
                    and snap.workers[worker_id].resident_shard != shard_id:
                pending.changed[worker_id] = shard_id
                pending.sources[worker_id] = c_switch(snap, worker_id, shard_id, opts).source
        self.pending = pending
        self.log.epoch = self.epoch
        self.log.emit("switch_start", target=[list(s) for s in stages], mode=mode, why=why,
                      reason=decision.reason, changed=pending.changed, new_epoch=epoch,
                      predicted=self._predicted(decision))
        await self._progress_switch()

    def _predicted(self, decision) -> dict:
        cost = decision.decision.cost if decision.decision else None
        if cost is None:
            return {}
        return {"T_total": cost.T_total.mid, "T_steady": cost.T_steady.mid,
                "C_switch_total": cost.C_switch_total.mid, "C_switch_max": cost.C_switch_max.mid}

    def _busy_workers(self) -> Set[str]:
        """Workers with in-flight old-epoch work (relay-visible or reported)."""
        busy = {r.computing_on for r in self.requests.values()
                if r.computing_on and r.request_id not in self.completed}
        if self.flags.direct_path and self.plan is not None:
            # Activations may be travelling worker-to-worker: wait until reports say idle.
            snap = self.snapshot()
            for stage in self.plan.stages:
                view = snap.workers.get(stage.worker_id)
                if view is not None and view.alive and (view.queue_len > 0 or view.session_state == "SERVING"):
                    if any(r.request_id not in self.completed and r.request_id not in self.queue
                           for r in self.requests.values()):
                        busy.add(stage.worker_id)
        return busy

    async def _progress_switch(self) -> None:
        pending = self.pending
        if pending is None:
            return
        alive = self.connected
        if pending.phase == "draining":
            busy = self._busy_workers() & alive
            if pending.mode == "drain":
                outstanding = [r for r in self.requests.values()
                               if r.request_id not in self.completed and r.request_id not in self.queue]
                # Requests held at the Foreman (retained, not computing) do not block the drain.
                blocking = [r for r in outstanding if r.computing_on in alive]
                if blocking:
                    return
            else:
                if busy & set(pending.changed):
                    return  # changed workers finish old-epoch work first, the rest resumes
            pending.phase = "loading"
            for worker_id, shard_id in pending.changed.items():
                await self._send_load(pending, worker_id, shard_id)
            if not pending.loads_pending:
                await self._commit()
            return
        if pending.phase == "loading" and not pending.loads_pending:
            await self._commit()

    async def _send_load(self, pending: PendingSwitch, worker_id: str, shard_id: str) -> None:
        if any(k[0] == worker_id for k in self.prefetching):
            await self.sender.send(worker_id, "cancel_prefetch", CancelPrefetchMsg(shard_id=None))
            for key in [k for k in self.prefetching if k[0] == worker_id]:
                self.prefetching.pop(key, None)
                self.log.emit("prefetch", worker=worker_id, shard=key[1], status="cancelled_by_load")
        source = pending.sources.get(worker_id)
        if source is not None and self.flags.peer_fetch:
            self.uploads[source] = self.uploads.get(source, 0) + 1
            await self.sender.send(source, "serve_allowed", ServeAllowedMsg(shard_id=shard_id, to=worker_id))
            self.log.emit("peer_fetch", donor=source, worker=worker_id, shard=shard_id, status="brokered")
        if (worker_id, shard_id) in self.prefetched:
            self.log.emit("prefetch_used", worker=worker_id, shard=shard_id)
        pending.loads_pending.add(worker_id)
        pending.load_started[worker_id] = self.clock()
        stage = next((i for i, (s, w) in enumerate(pending.target) if w == worker_id), None)
        await self.sender.send(worker_id, "load", LoadMsg(shard_id=shard_id, epoch=pending.epoch, stage=stage,
                                                          source=source if self.flags.peer_fetch else None),
                               epoch=pending.epoch)

    async def _on_load_result(self, worker_id: str, body: LoadResultMsg) -> None:
        self.sender.acknowledge_where(lambda m: m.get("type") == "load" and m.get("data", {}).get("shard_id") == body.shard_id
                                      and m.get("data", {}).get("epoch") == body.epoch)
        shard = self.registry.catalog.shards.get(body.shard_id)
        size = shard.bytes if shard else 0
        t = body.timings
        snap = self.snapshot()
        device_class = snap.workers[worker_id].device_class if worker_id in snap.workers else "unknown"
        if body.ok:
            self.learner.observe(device_class, size, load=t.load_s, init=t.init_s, write=t.write_s,
                                 unload=t.unload_s, peak_mem=t.peak_mem)
            if t.fetch_s > 0 and t.bytes_fetched:
                self.telemetry.observe_transfer(worker_id, "down", t.bytes_fetched, t.fetch_s)
            self.registry.touch(worker_id, body.shard_id, self.clock())
        pending = self.pending
        if pending is not None:
            source = pending.sources.get(worker_id)
            if source is not None and self.uploads.get(source):
                self.uploads[source] -= 1
        self.log.emit("message", kind="load_result", worker=worker_id,
                      shard=body.shard_id, ok=body.ok, timings=t.model_dump(), error=body.error)
        if pending is None or body.epoch != pending.epoch:
            return
        if not body.ok:
            self.excluded.add((worker_id, body.shard_id))
            self._abort_switch(f"load failed on {worker_id}: {body.error}")
            self.urgent.add("load_failed")
            await self.tick()
            return
        pending.loads_pending.discard(worker_id)
        # Reflect the new residency immediately (the next report confirms it).
        self.telemetry.note_resident(worker_id, body.shard_id, size)
        await self._progress_switch()

    async def _commit(self) -> None:
        pending = self.pending
        snap = self.snapshot()
        opts = self._opts()
        old_plan = self.plan
        self.plan = build_plan(snap, pending.target, epoch=pending.epoch, reason=pending.reason, opts=opts,
                               replicas=pending.replicas)
        self.epoch = pending.epoch
        self.log.epoch = self.epoch
        self.pending = None
        self.state = replace(self.state, baseline=Baseline.capture(snap))
        self.problems.reset_epoch()
        self.log.emit("switch_end", ok=True, mode=pending.mode, seconds=self.clock() - pending.started_at,
                      plan=self.plan.model_dump(mode="json"),
                      load_seconds={w: self.clock() - s for w, s in pending.load_started.items()})
        for stage in self.plan.stages:
            await self._send_route(stage.stage_index, stage.worker_id, self._next_hop(stage, snap, opts))
        # Re-tag and re-route held or lost requests under the new epoch.
        for request in list(self.requests.values()):
            if request.request_id in self.completed or request.request_id in self.queue:
                continue
            lost = request.computing_on is not None and request.computing_on not in self.connected
            if request.computing_on is not None and not lost:
                continue  # still running on a surviving worker; forwarded when it returns
            request.epoch = self.epoch
            if request.retained is not None:
                kept = self._stage_by_atom(self.plan, request.retained[0] + 1) is not None
                self.log.emit("message", kind="resume" if kept else "restart_from_input",
                              request_id=request.request_id,
                              reason="retained boundary kept by new plan" if kept else "cut changed")
                await self._forward(request)
            else:
                request.restarts += 1
                self.log.emit("message", kind="restart_from_input", request_id=request.request_id,
                              reason="no retained activation")
                await self._dispatch(request, 0, request.input)
        await self._admit()

    def _next_hop(self, stage, snap, opts) -> NextHop:
        nxt = stage.next_hop.worker_id
        routes = [RouteMode.VIA_FOREMAN]
        address = None
        if nxt is not None and self.flags.direct_path:
            shard = self.registry.catalog.shards[stage.shard_id]
            choice = self.routes.choose(snap, stage.worker_id, nxt,
                                        shard.out_act_bytes_per_token * opts.tokens, opts, self.clock())
            self.log.emit("route_flip" if choice.flipped else "route_choice", src=stage.worker_id, dst=nxt,
                          mode=choice.mode.value, reason=choice.reason, direct_s=choice.direct_s,
                          relay_s=choice.relay_s)
            if choice.mode is RouteMode.DIRECT:
                routes = [RouteMode.DIRECT, RouteMode.VIA_FOREMAN]
                record = self.telemetry._workers.get(nxt)
                address = record.state.peer_endpoint if record and record.state else None
        return NextHop(worker_id=nxt, allowed_routes=routes, address=address)

    async def _send_route(self, stage_index: int, worker_id: str, next_hop: NextHop) -> None:
        await self.sender.send(worker_id, "route", RouteMsg(epoch=self.epoch, stage=stage_index, next_hop=next_hop),
                               epoch=self.epoch)

    # ------------------------------------------------------------- tick
    async def tick(self, now: Optional[float] = None) -> None:
        async with self._tick_lock:
            await self._tick(self.clock() if now is None else now)

    async def _tick(self, now: float) -> None:
        if self.pending is not None:
            lost = set(self.pending.changed) - self.connected
            if lost:
                for worker_id in lost:
                    self.excluded.add((worker_id, self.pending.changed[worker_id]))
                self._abort_switch(f"worker(s) lost during switch: {sorted(lost)}")
                self.urgent.add("worker_lost")
            else:
                await self._progress_switch()
                await self._timeouts(now)
                return
        snap = self.snapshot()
        opts = self._opts()
        self.policy.opts = opts
        n = max(1, self.n_remaining())
        if self.plan is None and not self.queue:
            return
        job_deadline = self._deadline_remaining(now)
        current = current_cost(snap, self.plan, n, opts) if self.plan else None
        triggers = detect_triggers(
            snap, self.plan, self.state.baseline, self.config.controller,
            problem_count=self.problems.recent_count(now), urgent_problems=frozenset(self.urgent),
            deadline_remaining_s=job_deadline, j_current_hi=current.T_total.hi if current else None,
        )
        if self.plan is None:
            triggers = tuple(sorted(set(triggers) | {"initial"}))
        self.urgent.clear()
        started = time.perf_counter()
        decision, self.state = decide(snap, self.plan, n, self.policy, self.objective, self.state,
                                      self.config.controller, triggers, now)
        planner_s = time.perf_counter() - started
        self.last_decision = decision
        if decision.planner_called:
            snapshot_dict = snapshot_to_dict(snap)
            current_stages = [list(s) for s in self.plan.assignment()] if self.plan else None
            self.log.emit(
                "planner_call", inputs_hash=stable_hash({"snapshot": snapshot_dict, "n": n, "current": current_stages}),
                n=n, policy=self.policy.name, planner_s=planner_s,
                chosen=[list(s) for s in (decision.target or ())],
                scores=decision.decision.scores() if decision.decision else [],
                evaluated=decision.decision.search.evaluated if decision.decision and decision.decision.search else None,
                current=current_stages, objective=[self.objective.kind, self.objective.deadline_s],
                opts=opts_to_dict(opts), snapshot=snapshot_dict if self.log_snapshots else None,
                triggers=list(triggers), controller=dict(self.config.controller.__dict__),
            )
        self.log.emit("controller_decision", action=decision.action, reason=decision.reason,
                      triggers=list(decision.triggers), j_current=decision.j_current, j_new=decision.j_new,
                      margin=decision.margin, break_even=decision.break_even)
        if decision.switches:
            await self._start_switch(decision, forced=decision.action in ("forced_switch",))
        elif decision.action == "stall":
            self.log.emit("failure", kind="no_feasible_plan", triggers=list(triggers))
        if self.plan is not None and self.flags.direct_path and self.pending is None:
            await self._refresh_routes(snap, opts)
        if self.flags.prefetch and self.pending is None:
            await self._prefetch(snap, decision)
        await self._probes(now)
        await self._timeouts(now)
        await self._admit()

    def _deadline_remaining(self, now: float) -> Optional[float]:
        deadlines = [job.submitted_at + job.deadline_s - now for job in self.jobs.values() if job.deadline_s]
        return min(deadlines) if deadlines else None

    async def _refresh_routes(self, snap, opts) -> None:
        for stage in self.plan.stages:
            nxt = stage.next_hop.worker_id
            if nxt is None:
                continue
            before = self.routes.current(stage.worker_id, nxt)
            hop = self._next_hop(stage, snap, opts)
            if self.routes.current(stage.worker_id, nxt) is not before:
                await self._send_route(stage.stage_index, stage.worker_id, hop)

    async def _prefetch(self, snap, decision) -> None:
        likely = []
        if decision.decision is not None and decision.decision.search is not None:
            likely = [c.stages for c in decision.decision.search.ranked[1:]]
        utilisation = {}
        for worker_id, view in snap.workers.items():
            utilisation[worker_id] = min(1.0, view.queue_len / 4.0)
        last_used = {k: v for k, v in self.registry._last_used.items()}
        orders = plan_prefetch(snap, self.plan, likely, self.config.prefetch, link_utilisation=utilisation,
                               in_flight=self.prefetching.keys(), last_used=last_used)
        opts = self._opts()
        for order in orders:
            source = None
            if self.flags.peer_fetch:
                source = c_switch(snap, order.worker_id, order.shard_id, opts).source
            self.prefetching[(order.worker_id, order.shard_id)] = self.clock()
            await self.sender.send(order.worker_id, "prefetch", PrefetchMsg(shard_id=order.shard_id, source=source))
            if source is not None:
                await self.sender.send(source, "serve_allowed", ServeAllowedMsg(shard_id=order.shard_id,
                                                                                to=order.worker_id))
            self.log.emit("prefetch", worker=order.worker_id, shard=order.shard_id, status="issued",
                          evict=list(order.evict), reason=order.reason, source=source)

    async def _probes(self, now: float) -> None:
        for worker_id in sorted(self.connected):
            if self.telemetry.needs_probe(worker_id, now):
                self.telemetry.mark_probed(worker_id, now)
                await self.sender.send(worker_id, "probe", ProbeMsg(size=self.config.estimator.probe_bytes,
                                                                    url=self.probe_url))
                self.log.emit("probe", worker=worker_id, kind="bandwidth", size=self.config.estimator.probe_bytes)
        if self.flags.direct_path or self.flags.peer_fetch:
            for a, b in self.capability.pairs_to_test(self.connected, now):
                self.capability.record(a, b, self.capability.reachable(a, b), now)  # rate-limit re-tests
                await self.sender.send(a, "probe", ProbeMsg(size=16 * 1024, peer=b))
                self.log.emit("probe", worker=a, kind="capability", peer=b)

    async def _timeouts(self, now: float) -> None:
        for request in list(self.requests.values()):
            if request.request_id in self.completed or request.request_id in self.queue:
                continue
            if request.dispatched_at and now - request.dispatched_at > self.request_timeout_s and self.pending is None \
                    and self.plan is not None:
                self.log.emit("failure", kind="request_timeout", request_id=request.request_id)
                request.restarts += 1
                request.epoch = self.epoch
                request.retained = None
                await self._dispatch(request, 0, request.input)

    # ------------------------------------------------------------- views
    def status(self) -> dict:
        return {
            "epoch": self.epoch,
            "policy": self.policy.name,
            "flags": self.flags.to_dict(),
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "pending_switch": None if self.pending is None else {
                "phase": self.pending.phase, "mode": self.pending.mode, "changed": self.pending.changed},
            "queued": len(self.queue),
            "in_flight": self._in_flight(),
            "counters": dict(self.counters),
            "control_bytes": self.sender.control_bytes,
            "route_flips": self.routes.flips,
            "last_decision": None if self.last_decision is None else {
                "action": self.last_decision.action, "reason": self.last_decision.reason,
                "triggers": list(self.last_decision.triggers)},
        }
