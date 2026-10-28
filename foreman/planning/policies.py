"""Planning policies behind one interface (the evaluation baselines).

``Policy.plan(snapshot, current_plan, n_pending, objective)`` returns a
:class:`PolicyDecision`. Each policy also declares its ``switch_rule``, which
the controller applies:

* ``static``: never switches on its own; only forced replans (B0, B1, B3).
* ``always``: switches whenever its planner prefers another plan (B2).
* ``break_even``: switches only when J(new) + margin < J(current) (B5a, B5b, S-aware).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .contract import NextHop, Plan, PlanStage, Replica, RouteMode
from .cost_model import (
    CostOptions,
    PipelineCost,
    StageCost,
    c_switch,
    memory_feasible,
    pipeline_cost,
    replica_cost,
    residency,
    Residency,
)
from .planner import Candidate, Objective, SearchResult, Stages, exhaustive_search, search
from .types import ZERO, Q, Snapshot


@dataclass(frozen=True)
class PolicyDecision:
    policy: str
    stages: Optional[Stages]
    cost: Optional[PipelineCost]
    replicas: Tuple[Tuple[str, str], ...] = ()
    search: Optional[SearchResult] = None
    reason: str = ""
    feasible: bool = True

    def scores(self) -> List[dict]:
        if self.search is None:
            return []
        return [
            {"stages": list(c.stages), "T_total": c.cost.T_total.mid, "T_steady": c.cost.T_steady.mid,
             "C_switch_total": c.cost.C_switch_total.mid}
            for c in self.search.ranked
        ]


def current_cost(snap: Snapshot, plan: Optional[Plan], n: int, opts: CostOptions) -> Optional[PipelineCost]:
    """J(current): cost of keeping the current plan (None if it is no longer runnable)."""
    if plan is None or not plan.stages:
        return None
    for stage in plan.stages:
        worker = snap.workers.get(stage.worker_id)
        if worker is None or not worker.alive or (stage.worker_id, stage.shard_id) in snap.excluded:
            return None
    return pipeline_cost(snap, plan.assignment(), n, opts)


def composition_of(plan: Optional[Plan]) -> Optional[Tuple[str, ...]]:
    return tuple(s.shard_id for s in plan.stages) if plan and plan.stages else None


class Policy:
    name = "base"
    switch_rule = "static"

    def __init__(self, opts: CostOptions, mode: str = "exhaustive") -> None:
        self.opts = opts
        self.mode = mode

    def cost_options(self) -> CostOptions:
        return self.opts

    def plan(self, snap: Snapshot, current: Optional[Plan], n: int, objective: Objective) -> PolicyDecision:
        raise NotImplementedError

    def evaluate(self, snap: Snapshot, current: Optional[Plan], n: int) -> Optional[PipelineCost]:
        return current_cost(snap, current, n, self.cost_options())


# ---------------------------------------------------------------------------
# B0 / B1: fixed cut, tiered affinity
# ---------------------------------------------------------------------------


TIER_NAMES = ("resident", "cached", "idle", "evict")


def affinity_tier(snap: Snapshot, worker_id: str, shard_id: str) -> int:
    worker = snap.workers[worker_id]
    if worker.resident_shard == shard_id:
        return 0
    if shard_id in worker.disk_shards:
        return 1
    if worker.resident_shard is None:
        return 2
    return 3


class B0Policy(Policy):
    """Current behaviour: the SDK's fixed cut and 4-tier affinity
    (resident > cached on disk > idle > evict), ties broken by worker id."""

    name = "B0"
    switch_rule = "static"

    def _rank(self, snap: Snapshot, shard_id: str, candidates: Sequence[str]) -> List[str]:
        return sorted(candidates, key=lambda w: (affinity_tier(snap, w, shard_id), w))

    def plan(self, snap, current, n, objective):
        composition = snap.catalog.default_composition
        chosen: List[str] = []
        for shard_id in composition:
            candidates = [
                w for w in snap.alive_workers()
                if w not in chosen and memory_feasible(snap, w, shard_id, self.opts)
            ]
            if not candidates:
                return PolicyDecision(self.name, None, None, reason="no feasible worker", feasible=False)
            chosen.append(self._rank(snap, shard_id, candidates)[0])
        stages = tuple(zip(composition, chosen))
        return PolicyDecision(self.name, stages, pipeline_cost(snap, stages, n, self.opts), reason="fixed cut, affinity")


class B1Policy(B0Policy):
    """B0 plus a network-aware tier: within an affinity tier, rank by bandwidth and RTT."""

    name = "B1"

    def __init__(self, opts: CostOptions, mode: str = "exhaustive", bw_weight: float = 1.0, rtt_weight: float = 1.0):
        super().__init__(opts, mode)
        self.bw_weight = bw_weight
        self.rtt_weight = rtt_weight

    def _rank(self, snap, shard_id, candidates):
        top_bw = max((min(snap.workers[w].up.mean, snap.workers[w].down.mean) for w in candidates), default=1.0) or 1.0
        top_rtt = max((snap.workers[w].rtt.mean for w in candidates), default=1.0) or 1.0

        def score(w: str) -> float:
            worker = snap.workers[w]
            bw = min(worker.up.mean, worker.down.mean) / top_bw
            return -self.bw_weight * bw + self.rtt_weight * worker.rtt.mean / top_rtt

        return sorted(candidates, key=lambda w: (affinity_tier(snap, w, shard_id), score(w), w))


# ---------------------------------------------------------------------------
# Planner-based policies
# ---------------------------------------------------------------------------


class SAwarePolicy(Policy):
    """Ours: full switching cost from the current residency state."""

    name = "S-aware"
    switch_rule = "break_even"

    def __init__(self, opts: CostOptions, mode: str = "exhaustive", recut: bool = True):
        super().__init__(opts, mode)
        self.recut = recut
        if not recut:
            self.name = "S-aware-placement"

    def _compositions(self, snap: Snapshot, current: Optional[Plan]):
        if self.recut:
            return None
        return [composition_of(current) or snap.catalog.default_composition]

    def plan(self, snap, current, n, objective):
        opts = self.cost_options()
        result = search(snap, n, opts, objective, mode=self.mode, compositions=self._compositions(snap, current))
        if not result.found:
            return PolicyDecision(self.name, None, None, search=result, reason="no feasible plan", feasible=False)
        return PolicyDecision(self.name, result.best.stages, result.best.cost, search=result, reason="min J")


class B2Policy(SAwarePolicy):
    """EdgeShard/PipeEdge style at planner level: placement and cuts by predicted
    latency, ignoring switching cost, re-run every epoch."""

    name = "B2"
    switch_rule = "always"

    def __init__(self, opts: CostOptions, mode: str = "exhaustive"):
        super().__init__(opts, mode, recut=True)
        self.name = "B2"

    def cost_options(self):
        return replace(self.opts, zero_switch=True)


class B3Policy(SAwarePolicy):
    """CSGO style: load time in the objective, but no current-residency state; one shot."""

    name = "B3"
    switch_rule = "static"

    def __init__(self, opts: CostOptions, mode: str = "exhaustive"):
        super().__init__(opts, mode, recut=True)
        self.name = "B3"

    def cost_options(self):
        return replace(self.opts, ignore_residency=True)


class B5Policy(SAwarePolicy):
    """Dora-style runtime adapter with a per-plan switching overhead.

    B5a: synchronous (respects single residency; the stage stalls).
    B5b: asynchronous switching allowed (needs spare memory for two sessions);
    the unconstrained reference showing what single residency costs.
    """

    switch_rule = "break_even"

    def __init__(self, opts: CostOptions, mode: str = "exhaustive", overhead_s: float = 6.0, asynchronous: bool = False):
        super().__init__(opts, mode, recut=True)
        self.overhead_s = overhead_s
        self.asynchronous = asynchronous
        self.name = "B5b" if asynchronous else "B5a"

    def cost_options(self):
        return replace(self.opts, fixed_switch_overhead=self.overhead_s)

    def plan(self, snap, current, n, objective):
        opts = self.cost_options()
        if not self.asynchronous:
            return super().plan(snap, current, n, objective)
        old = current_cost(snap, current, n, opts)
        result = exhaustive_search(snap, n, replace(opts, zero_switch=True), objective, keep_resident=True)
        if not result.found:
            return PolicyDecision(self.name, None, None, search=result, reason="no feasible plan", feasible=False)
        best: Optional[Tuple[float, Candidate, PipelineCost]] = None
        for candidate in result.ranked:
            cost = self._async_cost(snap, candidate.stages, current, old, n, opts)
            if best is None or cost.T_total.mid < best[0]:
                best = (cost.T_total.mid, candidate, cost)
        return PolicyDecision(self.name, best[1].stages, best[2], search=result, reason="async switch")

    def _async_cost(self, snap, stages, current, old, n, opts) -> PipelineCost:
        """Old plan keeps serving while the new shards load next to it."""
        changed = current is None or tuple(stages) != current.assignment()
        switch = self.overhead_s if changed else 0.0
        served = 0
        if old is not None and changed and old.T_steady.mid > 0:
            served = min(n, int(switch / old.T_steady.mid))
        new = pipeline_cost(snap, stages, max(1, n - served), replace(opts, zero_switch=True))
        shift = Q.exact(switch)
        return replace(new, T_total=new.T_total + shift, C_switch_total=shift, C_switch_max=shift, n=n)


class B4Policy(Policy):
    """Replicate where it fits: data-parallel full-model replicas on every worker
    that can hold the full shard; pipeline only over the remaining workers."""

    name = "B4"
    switch_rule = "static"

    def plan(self, snap, current, n, objective):
        opts = self.opts
        full = snap.catalog.full_shard_id
        replicas: List[str] = []
        if full is not None:
            replicas = [w for w in snap.alive_workers() if memory_feasible(snap, w, full, opts)]
        rest = [w for w in snap.alive_workers() if w not in replicas]
        pipeline: Optional[Candidate] = None
        if rest:
            sub = replace(snap, workers={w: v for w, v in snap.workers.items() if w in rest})
            compositions = [c for c in snap.catalog.compositions() if c != (full,)]
            result = exhaustive_search(sub, n, opts, objective, compositions=compositions)
            pipeline = result.best
        if not replicas and pipeline is None:
            return PolicyDecision(self.name, None, None, reason="no feasible plan", feasible=False)
        groups = []  # (ready, latency, period)
        details: List[StageCost] = []
        for w in replicas:
            ready, latency, period = replica_cost(snap, w, full, opts)
            groups.append((ready, latency, period))
            details.append(StageCost(full, w, ready, latency, ZERO, "via_foreman", residency(snap.workers[w], full, opts), ready + latency))
        if pipeline is not None:
            cost = pipeline.cost
            groups.append((cost.C_switch_max, cost.finish_K + cost.out_leg, cost.T_steady))
            details.extend(cost.stages)
        rate = sum(1.0 / max(1e-9, g[2].mid) for g in groups)
        ready_max = Q.max(*(g[0] for g in groups))
        latency_min = Q.min(*(g[1] for g in groups))
        period = Q.exact(1.0 / rate)
        total = ready_max + latency_min + period.scale(max(0, n - 1))
        switch_total = ZERO
        for g in groups:
            switch_total = switch_total + g[0]
        cost = PipelineCost(total, period, switch_total, ready_max, latency_min, ZERO, n, tuple(details))
        if pipeline is not None:
            stages = pipeline.stages
            replica_pairs = tuple((full, w) for w in replicas)
        else:
            stages = ((full, replicas[0]),)
            replica_pairs = tuple((full, w) for w in replicas[1:])
        return PolicyDecision(self.name, stages, cost, replicas=replica_pairs, reason="replicate where it fits")


def make_policy(name: str, opts: CostOptions, mode: str = "exhaustive", **kwargs) -> Policy:
    if name == "B0":
        return B0Policy(opts, mode)
    if name == "B1":
        return B1Policy(opts, mode, **kwargs)
    if name == "B2":
        return B2Policy(opts, mode)
    if name == "B3":
        return B3Policy(opts, mode)
    if name == "B4":
        return B4Policy(opts, mode)
    if name == "B5a":
        return B5Policy(opts, mode, asynchronous=False, **kwargs)
    if name == "B5b":
        return B5Policy(opts, mode, asynchronous=True, **kwargs)
    if name == "S-aware":
        return SAwarePolicy(opts, mode, recut=True)
    if name == "S-aware-placement":
        return SAwarePolicy(opts, mode, recut=False)
    raise ValueError(f"unknown policy {name!r}")


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def build_plan(
    snap: Snapshot,
    stages: Stages,
    *,
    epoch: int,
    reason: str,
    opts: CostOptions,
    replicas: Sequence[Tuple[str, str]] = (),
    n_backups: int = 2,
) -> Plan:
    plan_workers = {w for _, w in stages} | {w for _, w in replicas}
    plan_stages: List[PlanStage] = []
    for index, (shard_id, worker_id) in enumerate(stages):
        next_worker = stages[index + 1][1] if index + 1 < len(stages) else None
        routes = [RouteMode.VIA_FOREMAN]
        if opts.direct_path and next_worker is not None and snap.capability.get((worker_id, next_worker), False):
            routes.append(RouteMode.DIRECT)
        backups = sorted(
            (w for w in snap.alive_workers() if w not in plan_workers and memory_feasible(snap, w, shard_id, opts)),
            key=lambda w: (c_switch(snap, w, shard_id, opts).total.mid, w),
        )[:n_backups]
        plan_stages.append(PlanStage(
            stage_index=index, shard_id=shard_id, worker_id=worker_id,
            next_hop=NextHop(worker_id=next_worker, allowed_routes=routes), backups=backups,
        ))
    return Plan(
        epoch=epoch, stages=plan_stages, created_at=snap.now, reason=reason,
        replicas=[Replica(shard_id=s, worker_id=w) for s, w in replicas],
    )
