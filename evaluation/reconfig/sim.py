"""Trace-driven simulator that runs the *same* pure planner and controller as the Foreman.

Ground truth (worker links, compute, phase times) evolves with the scenario.
The controller only sees noisy telemetry through the real TelemetryStore and
ProfileLearner. Execution uses a fluid pipeline model consistent with the cost
model: a switch stalls for the slowest changed stage's true C_switch (single
residency; B5b overlaps it with the old plan), after which the pipeline refills
(first completion after the true single-request latency) and then completes one
request every true period P.

An ``oracle`` run decides on ground truth with zero margin and no hysteresis;
regret = T_policy - T_oracle.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from foreman.planning.config import ControllerConfig, RunConfig
from foreman.planning.contract import WorkerState
from foreman.planning.controller import ControllerState, decide, detect_triggers
from foreman.planning.cost_model import CostOptions, c_switch, pipeline_cost, stage_bytes, t_comm
from foreman.planning.eventlog import EventLog, stable_hash
from foreman.planning.planner import Objective
from foreman.planning.policies import build_plan, make_policy
from foreman.planning.prefetch import plan_prefetch
from foreman.planning.profiles import ProfileLearner
from foreman.planning.routes import RouteSelector
from foreman.planning.scope import infer_kind, kind_allowed
from foreman.planning.telemetry import TelemetryStore
from foreman.planning.types import DeviceProfile, Est, LinkView, ShardCatalog, Snapshot, WorkerView

from .scenario import ScenarioEvent
from foreman.planning.serialize import opts_to_dict, snapshot_to_dict

HEARTBEAT_LEGACY_BYTES = 420  # typical JSON size of today's heartbeat
REPORT_V2_BYTES = 900  # typical JSON size of a v2 report with stage_stats and peer links
COMMAND_BYTES = 220  # load/route/prefetch/probe envelope


@dataclass
class SimWorker:
    worker_id: str
    device_class: str
    bw: float  # bytes/s, both directions to the Foreman
    rtt: float = 0.02
    cpu: float = 0.0
    temp: Optional[float] = None
    resident: Optional[str] = None
    disk: set = field(default_factory=set)
    ram: Optional[int] = None
    battery: Optional[float] = 90.0
    alive: bool = True


@dataclass
class SimSpec:
    catalog: ShardCatalog
    workers: List[SimWorker]
    profiles: Mapping[str, DeviceProfile]  # ground truth (exact fits)
    peer_links: Dict[Tuple[str, str], Tuple[float, float]] = field(default_factory=dict)  # (bw, rtt)


@dataclass
class SimResult:
    policy: str
    total_time: float
    completed: int
    switches: int
    stall_time: float
    wasted_switches: int
    bytes_moved: int
    control_bytes: int
    planner_time_s: float
    deadline_hit_rate: Optional[float]
    route_shares: Dict[str, float]
    route_flips: int
    log: EventLog


class Simulator:
    def __init__(self, spec: SimSpec, config: RunConfig, events: Sequence[ScenarioEvent] = (), *,
                 dt: float = 1.0, noise: float = 0.05, oracle: bool = False, max_time: float = 1e6,
                 log_path: Optional[str] = None, log_snapshots: bool = True) -> None:
        self.spec = spec
        self.config = config
        self.events = sorted(events, key=lambda e: e.at)
        self.dt = dt
        self.noise = noise
        self.oracle = oracle
        self.max_time = max_time
        self.log_snapshots = log_snapshots
        self.rng = random.Random(config.seed)
        self.t = 0.0
        self.workers = {w.worker_id: replace(w, disk=set(w.disk) | ({w.resident} if w.resident else set()))
                        for w in spec.workers}
        policy_name = "S-aware" if oracle else config.policy
        self.log = EventLog(log_path, run_id=config.run_id, seed=config.seed,
                            policy=("oracle" if oracle else config.policy), flags=config.flags.to_dict(),
                            clock=lambda: self.t, keep_in_memory=True)
        self.telemetry = TelemetryStore(config.estimator, clock=lambda: self.t)
        self.learner = ProfileLearner(spec.profiles)
        self.policy_name = policy_name
        self.objective = Objective(config.objective, config.deadline_s)
        self.cfg = (ControllerConfig(persist_k=1, cooldown_s=0, max_switches=10**9, margin_abs_s=0,
                                     margin_frac=0) if oracle else config.controller)
        self.routes = RouteSelector(config.routes)

    # ------------------------------------------------------------ truth
    def truth(self) -> Snapshot:
        exact = Est.known
        workers = {
            w.worker_id: WorkerView(
                worker_id=w.worker_id, device_class=w.device_class, alive=w.alive, cpu=w.cpu, ram_free=w.ram,
                battery=w.battery, temp=w.temp, resident_shard=w.resident, disk_shards=frozenset(w.disk),
                up=exact(w.bw), down=exact(w.bw), rtt=exact(w.rtt),
                eligible=kind_allowed(infer_kind(w.device_class), self.config.dnn_workers),
            )
            for w in self.workers.values()
        }
        links = {pair: LinkView(True, exact(bw), exact(rtt)) for pair, (bw, rtt) in self.spec.peer_links.items()
                 if self.workers[pair[0]].alive and self.workers[pair[1]].alive}
        capability = {pair: True for pair in links}
        return Snapshot(now=self.t, workers=workers, catalog=self.spec.catalog, profiles=self.spec.profiles,
                        links=links, capability=capability, compression_ratio=self.config.cost.compression_ratio)

    def _opts(self, plan=None, **overrides) -> CostOptions:
        critical = frozenset(plan.workers) if plan else frozenset()
        return CostOptions.from_config(self.config, critical_workers=critical, **overrides)

    def _noisy(self, value: float) -> float:
        return max(1e-9, value * (1.0 + self.rng.gauss(0.0, self.noise)))

    def _heartbeats(self, truth: Snapshot, opts: CostOptions) -> None:
        for w in self.workers.values():
            if not w.alive:
                self.telemetry.mark_departed(w.worker_id)
                continue
            stats = None
            if w.resident:
                from foreman.planning.cost_model import t_comp
                seconds = t_comp(truth, w.worker_id, w.resident, opts).mid
                stats = {"compute_ms_p50": self._noisy(seconds) * 1000, "samples": 5, "shard_id": w.resident}
            peer_links = [
                {"peer": b, "reachable": True, "rtt_ms": rtt * 1000, "bw_est": self._noisy(bw), "age_s": 0}
                for (a, b), (bw, rtt) in self.spec.peer_links.items()
                if a == w.worker_id and self.workers[b].alive
            ]
            self.telemetry.ingest_state(WorkerState(
                worker_id=w.worker_id, device_class=w.device_class, cpu=w.cpu, ram_free=w.ram, battery=w.battery,
                temp=w.temp, resident_shard=w.resident,
                disk_shards=[{"id": s, "bytes": self.spec.catalog.shards[s].bytes} for s in sorted(w.disk)],
                observed_up_bps=self._noisy(w.bw), observed_down_bps=self._noisy(w.bw),
                rtt_to_foreman_ms=self._noisy(w.rtt) * 1000, stage_stats=stats, peer_links=peer_links,
            ))

    def _apply_events(self) -> None:
        while self.events and self.events[0].at <= self.t:
            event = self.events.pop(0)
            worker = self.workers.get(event.worker) if event.worker else None
            if event.action == "network" and worker:
                worker.bw = event.bw if event.bw is not None else worker.bw
                worker.rtt = event.rtt if event.rtt is not None else worker.rtt
            elif event.action == "load" and worker:
                worker.cpu = event.cpu if event.cpu is not None else worker.cpu
                worker.temp = event.temp if event.temp is not None else worker.temp
            elif event.action == "drop_worker" and worker:
                worker.alive = False
                worker.resident = None
            elif event.action == "join_worker" and worker:
                worker.alive = True
            self.log.emit("scenario", action=event.action, worker=event.worker, at=event.at)

    # ------------------------------------------------------------ run
    def run(self, n: Optional[int] = None) -> SimResult:
        n = n or self.config.n_requests
        cfg = self.config
        policy = make_policy(self.policy_name, self._opts(), cfg.planner_mode)
        state = ControllerState()
        plan = None
        epoch = 0
        remaining = n
        completed_times: List[float] = []
        next_done: Optional[float] = None
        period = math.inf
        stall_until = 0.0
        first_done: Optional[float] = None
        stall_time = 0.0
        switches = 0
        wasted = 0
        bytes_moved = 0
        control_bytes = 0
        planner_time = 0.0
        history: List[Tuple[float, tuple]] = []
        route_counts: Dict[str, int] = {}
        prefetched: set = set()
        self.log.emit("run_start", config=cfg.to_dict(), n=n, oracle=self.oracle)
        heartbeat_bytes = REPORT_V2_BYTES if cfg.flags.adaptive_planning else HEARTBEAT_LEGACY_BYTES

        while remaining > 0 and self.t < self.max_time:
            self._apply_events()
            truth = self.truth()
            opts = self._opts(plan)
            self._heartbeats(truth, opts)
            control_bytes += heartbeat_bytes * sum(1 for w in self.workers.values() if w.alive)
            obs = truth if self.oracle else self.telemetry.snapshot(
                self.spec.catalog, self.learner.profiles(), default_compression=cfg.cost.compression_ratio,
                dnn_scope=cfg.dnn_workers)
            policy.opts = opts
            triggers = detect_triggers(obs, plan, state.baseline, self.cfg)
            if self.oracle and plan is not None and "initial" not in triggers:
                triggers = tuple(sorted(set(triggers) | {"oracle_tick"}))
            started = time.perf_counter()
            decision, state = decide(obs, plan, remaining, policy, self.objective, state, self.cfg, triggers, self.t)
            spent = time.perf_counter() - started
            planner_time += spent
            if decision.planner_called:
                record = {"n": remaining, "policy": policy.name, "planner_s": spent,
                          "chosen": [list(s) for s in (decision.target or ())],
                          "scores": decision.decision.scores() if decision.decision else [],
                          "triggers": list(triggers), "action": decision.action,
                          "current": [list(s) for s in plan.assignment()] if plan else None,
                          "objective": [self.objective.kind, self.objective.deadline_s],
                          "opts": opts_to_dict(opts), "controller": dict(self.cfg.__dict__)}
                if self.log_snapshots:
                    record["snapshot"] = snapshot_to_dict(obs)
                    record["controller_state"] = {"last_switch_at": state.last_switch_at}
                record["inputs_hash"] = stable_hash({k: record.get(k) for k in ("snapshot", "n", "current", "opts")})
                self.log.emit("planner_call", **record)
            self.log.emit("controller_decision", action=decision.action, reason=decision.reason,
                          triggers=list(decision.triggers), j_current=decision.j_current, j_new=decision.j_new,
                          margin=decision.margin, break_even=decision.break_even)

            if decision.switches and decision.target:
                stages = decision.target
                truth_opts = self._opts(plan)
                changed = [(s, w) for s, w in stages if self.workers[w].resident != s]
                phases = {w: c_switch(truth, w, s, truth_opts) for s, w in changed}
                # Execution follows the cost model: ready_i = true C_switch_i, so the
                # pipeline starts filling before the slowest stage finishes loading.
                refill = pipeline_cost(truth, stages, max(1, remaining), truth_opts)
                duration = max((p.total.mid for p in phases.values()), default=0.0)
                asynchronous = policy.name == "B5b" and plan is not None
                epoch += 1
                self.log.epoch = epoch
                self.log.emit("switch_start", target=[list(s) for s in stages], changed=[list(c) for c in changed],
                              reason=decision.reason, mode="resume" if decision.action == "forced_switch" else "drain",
                              predicted=decision.decision.cost.C_switch_max.mid if decision.decision and decision.decision.cost else None)
                for s, w in changed:
                    worker = self.workers[w]
                    p = phases[w]
                    if s not in worker.disk:
                        bytes_moved += self.spec.catalog.shards[s].bytes
                    if (w, s) in prefetched:
                        self.log.emit("prefetch_used", worker=w, shard=s)
                    worker.resident = s
                    worker.disk.add(s)
                    phase = dict(p.phases)
                    self.learner.observe(worker.device_class, self.spec.catalog.shards[s].bytes,
                                         **{k: self._noisy(v.mid) for k, v in phase.items() if k != "fetch"})
                    control_bytes += COMMAND_BYTES
                self.log.emit("switch_end", seconds=duration, asynchronous=asynchronous,
                              phases={w: {k: v.mid for k, v in p.phases} for w, p in phases.items()})
                switches += 1 if decision.action != "initial" else 0
                key = tuple(stages)
                if history and len(history) >= 2 and history[-2][1] == key and self.t - history[-2][0] <= self.cfg.switch_window_s:
                    wasted += 1
                history.append((self.t, key))
                plan = build_plan(obs, stages, epoch=epoch, reason=decision.reason, opts=truth_opts,
                                  replicas=decision.decision.replicas if decision.decision else ())
                control_bytes += COMMAND_BYTES * len(stages)  # route messages
                if not asynchronous:
                    stall_until = self.t + duration
                    stall_time += duration if decision.action != "initial" else 0.0
                    next_done = None
                    first_done = self.t + refill.finish_K.mid + refill.out_leg.mid
            elif decision.action == "stall":
                self.log.emit("failure", kind="no_feasible_plan")

            if plan is not None and cfg.flags.prefetch:
                likely = [c.stages for c in decision.decision.search.ranked[1:]] if decision.decision and decision.decision.search else []
                for order in plan_prefetch(truth, plan, likely, cfg.prefetch, link_utilisation={}):
                    worker = self.workers[order.worker_id]
                    for victim in order.evict:
                        worker.disk.discard(victim)
                    worker.disk.add(order.shard_id)
                    prefetched.add((order.worker_id, order.shard_id))
                    bytes_moved += self.spec.catalog.shards[order.shard_id].bytes
                    control_bytes += COMMAND_BYTES
                    self.log.emit("prefetch", worker=order.worker_id, shard=order.shard_id, status="done",
                                  bytes=self.spec.catalog.shards[order.shard_id].bytes, reason=order.reason)

            # ------------------------------------------------ execute dt
            t_end = self.t + self.dt
            runnable = plan is not None and all(self.workers[w].alive for w in plan.workers)
            if runnable:
                truth = self.truth()
                exec_opts = self._opts(plan, zero_switch=True)
                cost = pipeline_cost(truth, plan.assignment(), max(1, remaining), exec_opts)
                period = cost.T_steady.mid
                if cfg.flags.direct_path:
                    _, boundaries, _ = stage_bytes(truth, [s for s, _ in plan.assignment()], exec_opts)
                    for index, (stage, nxt) in enumerate(zip(plan.stages, plan.stages[1:])):
                        choice = self.routes.choose(obs, stage.worker_id, nxt.worker_id, boundaries[index], opts, self.t)
                        if choice.flipped:
                            self.log.emit("route_flip", src=stage.worker_id, dst=nxt.worker_id, mode=choice.mode.value,
                                          reason=choice.reason)
                if next_done is None:
                    next_done = first_done if first_done is not None else                         max(self.t, stall_until) + cost.finish_K.mid + cost.out_leg.mid
                    first_done = None
                while next_done <= t_end and remaining > 0:
                    completed_times.append(next_done)
                    self.log.emit("request_end", at_ts=next_done, epoch=epoch)
                    for index in range(len(plan.stages) - 1):
                        route = (self.routes.current(plan.stages[index].worker_id, plan.stages[index + 1].worker_id).value
                                 if cfg.flags.direct_path else "via_foreman")
                        route_counts[route] = route_counts.get(route, 0) + 1
                    remaining -= 1
                    next_done += period
            else:
                next_done = None
            self.t = t_end

        total = completed_times[-1] if completed_times else math.inf
        hit = None
        if cfg.deadline_s is not None and completed_times:
            hit = sum(1 for c in completed_times if c <= cfg.deadline_s) / n
        shares_total = sum(route_counts.values()) or 1
        result = SimResult(
            policy=("oracle" if self.oracle else policy.name), total_time=total, completed=len(completed_times),
            switches=switches, stall_time=stall_time, wasted_switches=wasted, bytes_moved=bytes_moved,
            control_bytes=control_bytes, planner_time_s=planner_time, deadline_hit_rate=hit,
            route_shares={k: v / shares_total for k, v in route_counts.items()}, route_flips=self.routes.flips,
            log=self.log,
        )
        self.log.emit("run_end", **{k: v for k, v in result.__dict__.items() if k != "log"})
        self.log.close()
        return result
