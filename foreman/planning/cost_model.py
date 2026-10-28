"""Pure cost model that prices switching under single residency.

No I/O, no clocks, no globals: every function takes a :class:`Snapshot` and
:class:`CostOptions` and returns interval quantities (:class:`Q`).

Formulas (seconds):

* C_switch(w, s) = 0 if s is resident on w
  = t_unload(w) [if another shard is resident] + t_load(w, s) + t_init(w, s) if on disk
  = the above + C_fetch(w, s) + t_write(w, s) if absent
* C_fetch(w, s) = bytes(s) / best_bw + rtt, best over the Foreman and (with
  peer_fetch) eligible donors holding s on disk.
* t_comm(i -> j) = act_bytes * compression_ratio / bw + rtt, where the route is
  min(direct, relay) when a direct link is allowed and fresh, else relay with
  bw = min(up_i, down_j) and rtt = rtt_i + rtt_j.
* Pipeline for N requests: finish_0 = 0,
  finish_i = max(ready_i, finish_{i-1} + t_comm_{i-1,i}) + t_comp_i with ready_i = C_switch_i,
  P = max_i(t_comp_i, t_comm_{i,i+1}),
  T_total(N) = finish_K + t_out + (N - 1) * P.
  ``t_out`` (the stage K -> Foreman leg) and t_comm_{0,1} (Foreman -> stage 1) are
  the I/O legs; set ``include_io_legs=False`` for the brief's formula verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .config import RunConfig
from .profiles import profile_for
from .types import ZERO, Est, Q, ShardInfo, Snapshot, WorkerView

FOREMAN = None  # endpoint id used for the Foreman in t_comm


class Residency(str, Enum):
    RESIDENT = "resident"
    ON_DISK = "on_disk"
    ABSENT = "absent"


@dataclass(frozen=True)
class CostOptions:
    k: float = 2.0
    age_growth: float = 0.05
    max_age_s: float = 30.0
    min_samples: int = 2
    min_bw: float = 1_000.0
    seq_len: int = 64
    batch: int = 1
    include_io_legs: bool = True
    runtime_overhead_bytes: int = 64 * 1024 * 1024
    peer_fetch: bool = False
    direct_path: bool = False
    donor_min_battery: float = 30.0
    donor_max_thermal: int = 2
    donor_max_uploads: int = 1
    temp_slowdown_per_c: float = 0.02
    temp_ref_c: float = 38.0
    allow_time_multiplexing: bool = False
    critical_workers: FrozenSet[str] = frozenset()
    # Policy switches (baselines):
    ignore_residency: bool = False  # B3: plan as if nothing is resident or on disk
    zero_switch: bool = False  # B2: ignore switching cost
    fixed_switch_overhead: Optional[float] = None  # B5a/B5b: per-shard constant overhead

    @classmethod
    def from_config(cls, config: RunConfig, **overrides) -> "CostOptions":
        est, cost = config.estimator, config.cost
        values = dict(
            k=est.uncertainty_k,
            age_growth=est.age_std_growth,
            max_age_s=est.max_age_s,
            min_samples=est.min_samples,
            min_bw=cost.min_bandwidth_bps,
            seq_len=cost.seq_len,
            batch=cost.batch,
            include_io_legs=cost.include_io_legs,
            runtime_overhead_bytes=cost.runtime_overhead_bytes,
            peer_fetch=config.flags.peer_fetch,
            direct_path=config.flags.direct_path,
            donor_min_battery=cost.donor_min_battery,
            donor_max_thermal=cost.donor_max_thermal,
            donor_max_uploads=cost.donor_max_uploads,
            temp_slowdown_per_c=cost.temp_slowdown_per_c,
            temp_ref_c=cost.temp_ref_c,
            allow_time_multiplexing=cost.allow_time_multiplexing,
        )
        values.update(overrides)
        return cls(**values)

    @property
    def tokens(self) -> int:
        return self.seq_len * self.batch


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _q(est: Est, opts: CostOptions, floor: float = 0.0) -> Q:
    return Q.from_est(est, opts.k, opts.age_growth, floor)


def residency(worker: WorkerView, shard_id: str, opts: Optional[CostOptions] = None) -> Residency:
    if opts is not None and opts.ignore_residency:
        return Residency.ABSENT
    if worker.resident_shard == shard_id:
        return Residency.RESIDENT
    if shard_id in worker.disk_shards:
        return Residency.ON_DISK
    return Residency.ABSENT


def phase_time(snap: Snapshot, worker_id: str, phase: str, n_bytes: int, opts: CostOptions) -> Q:
    worker = snap.workers[worker_id]
    fit = getattr(profile_for(snap.profiles, worker.device_class), phase)
    return Q.from_est(fit.predict(n_bytes), opts.k, 0.0, 0.0)


def is_stale(est: Est, opts: CostOptions) -> bool:
    return est.is_stale(opts.max_age_s, opts.min_samples)


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkCost:
    time: Q
    route: str  # "via_foreman" | "direct" | "foreman_leg"
    stale: bool = False


def _direct_link(snap: Snapshot, i: str, j: str, opts: CostOptions):
    if not opts.direct_path or not snap.capability.get((i, j), False):
        return None
    link = snap.links.get((i, j))
    if link is None or not link.reachable or is_stale(link.bw, opts):
        return None
    return link


def t_comm(snap: Snapshot, i: Optional[str], j: Optional[str], n_bytes: float, opts: CostOptions) -> LinkCost:
    """Transfer time of an activation from i to j (None = Foreman)."""
    wire = n_bytes * snap.compression_ratio
    if i is None and j is None:
        return LinkCost(ZERO, "foreman_leg")
    if i is None or j is None:
        worker = snap.workers[j if i is None else i]
        bw = worker.down if i is None else worker.up
        time = Q.divide(wire, _q(bw, opts), opts.min_bw) + _q(worker.rtt, opts)
        return LinkCost(time, "foreman_leg", is_stale(bw, opts))
    if i == j:
        return LinkCost(ZERO, "local")
    wi, wj = snap.workers[i], snap.workers[j]
    relay_bw = Q.min(_q(wi.up, opts), _q(wj.down, opts))
    relay = Q.divide(wire, relay_bw, opts.min_bw) + _q(wi.rtt, opts) + _q(wj.rtt, opts)
    relay_stale = is_stale(wi.up, opts) or is_stale(wj.down, opts)
    direct_link = _direct_link(snap, i, j, opts)
    if direct_link is not None:
        direct = Q.divide(wire, _q(direct_link.bw, opts), opts.min_bw) + _q(direct_link.rtt, opts)
        if direct.mid < relay.mid:
            return LinkCost(direct, "direct", False)
    return LinkCost(relay, "via_foreman", relay_stale)


# ---------------------------------------------------------------------------
# Fetch and switch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchCost:
    time: Q
    source: Optional[str]  # None = Foreman
    stale: bool


def eligible_donors(snap: Snapshot, worker_id: str, shard_id: str, opts: CostOptions) -> List[str]:
    donors = []
    for donor_id in sorted(snap.holders(shard_id)):
        if donor_id == worker_id:
            continue
        donor = snap.workers[donor_id]
        if donor.battery is not None and not donor.charging and donor.battery < opts.donor_min_battery:
            continue
        if donor.thermal is not None and donor.thermal > opts.donor_max_thermal:
            continue
        if snap.uploads_in_flight.get(donor_id, 0) >= opts.donor_max_uploads:
            continue
        if donor_id in opts.critical_workers:
            continue
        if _peer_link(snap, donor_id, worker_id, opts) is None:
            continue
        donors.append(donor_id)
    return donors


def _peer_link(snap: Snapshot, donor: str, worker: str, opts: CostOptions):
    """Peer fetch needs a fresh, reachable donor->worker link (independent of direct_path)."""
    if not snap.capability.get((donor, worker), False):
        return None
    link = snap.links.get((donor, worker))
    if link is None or not link.reachable or is_stale(link.bw, opts):
        return None
    return link


def c_fetch(snap: Snapshot, worker_id: str, shard: ShardInfo, opts: CostOptions) -> FetchCost:
    worker = snap.workers[worker_id]
    best = FetchCost(
        Q.divide(shard.bytes, _q(worker.down, opts), opts.min_bw) + _q(worker.rtt, opts),
        None,
        is_stale(worker.down, opts),
    )
    if not opts.peer_fetch:
        return best
    for donor in eligible_donors(snap, worker_id, shard.shard_id, opts):
        link = _peer_link(snap, donor, worker_id, opts)
        time = Q.divide(shard.bytes, _q(link.bw, opts), opts.min_bw) + _q(link.rtt, opts)
        if time.mid < best.time.mid:
            best = FetchCost(time, donor, False)
    return best


@dataclass(frozen=True)
class SwitchCost:
    total: Q
    residency: Residency
    phases: Tuple[Tuple[str, Q], ...] = ()
    source: Optional[str] = None
    stale: bool = False


def c_switch(snap: Snapshot, worker_id: str, shard_id: str, opts: CostOptions) -> SwitchCost:
    worker = snap.workers[worker_id]
    shard = snap.catalog.shards[shard_id]
    state = residency(worker, shard_id, opts)
    if state is Residency.RESIDENT:
        return SwitchCost(ZERO, state)
    if opts.zero_switch:
        return SwitchCost(ZERO, state)
    if opts.fixed_switch_overhead is not None:
        return SwitchCost(Q.exact(opts.fixed_switch_overhead), state)
    phases: List[Tuple[str, Q]] = []
    stale = False
    other_resident = worker.resident_shard is not None and not opts.ignore_residency
    if other_resident:
        resident = snap.catalog.shards.get(worker.resident_shard)
        phases.append(("unload", phase_time(snap, worker_id, "unload", resident.bytes if resident else 0, opts)))
    source = None
    if state is Residency.ABSENT:
        fetch = c_fetch(snap, worker_id, shard, opts)
        phases.append(("fetch", fetch.time))
        phases.append(("write", phase_time(snap, worker_id, "write", shard.bytes, opts)))
        source, stale = fetch.source, fetch.stale
    phases.append(("load", phase_time(snap, worker_id, "load", shard.bytes, opts)))
    phases.append(("init", phase_time(snap, worker_id, "init", shard.bytes, opts)))
    total = ZERO
    for _, value in phases:
        total = total + value
    return SwitchCost(total, state, tuple(phases), source, stale)


# ---------------------------------------------------------------------------
# Compute and memory
# ---------------------------------------------------------------------------


def t_comp(snap: Snapshot, worker_id: str, shard_id: str, opts: CostOptions) -> Q:
    worker = snap.workers[worker_id]
    profile = profile_for(snap.profiles, worker.device_class)
    shard = snap.catalog.shards[shard_id]
    measured = profile.compute_by_shard.get(shard_id)
    if measured is not None and measured.n > 0:
        base = _q(measured, opts).scale(opts.tokens / max(1, profile.ref_tokens))
    else:
        flops = shard.flops_per_token * opts.tokens
        base = Q.divide(flops, _q(profile.flops_per_s, opts), 1.0)
    factor = 1.0 + max(0.0, worker.cpu)
    if worker.temp is not None:
        factor *= 1.0 + opts.temp_slowdown_per_c * max(0.0, worker.temp - opts.temp_ref_c)
    return base.scale(factor)


def compute_stale(snap: Snapshot, worker_id: str, shard_id: str, opts: CostOptions) -> bool:
    worker = snap.workers[worker_id]
    profile = profile_for(snap.profiles, worker.device_class)
    measured = profile.compute_by_shard.get(shard_id)
    if measured is not None and measured.n > 0:
        return is_stale(measured, opts)
    # No per-shard measurement: the FLOPs model is only trusted with a fresh throughput estimate.
    return is_stale(profile.flops_per_s, opts)


def memory_need(snap: Snapshot, worker_id: str, shard_id: str, opts: CostOptions) -> float:
    worker = snap.workers[worker_id]
    shard = snap.catalog.shards[shard_id]
    peak = profile_for(snap.profiles, worker.device_class).peak_mem.predict(shard.bytes).mean
    return max(peak, shard.bytes) + opts.runtime_overhead_bytes


def memory_feasible(snap: Snapshot, worker_id: str, shard_id: str, opts: CostOptions,
                    keep_resident: bool = False) -> bool:
    """Shard bytes + learned load peak + runtime overhead must fit in free RAM.

    Single residency: switching frees the resident shard first, so its bytes
    count as available unless ``keep_resident`` (asynchronous switching, B5b).
    Unknown ``ram_free`` is treated as feasible (and is visible in the logs).
    """
    worker = snap.workers[worker_id]
    if not worker.alive or not worker.eligible or (worker_id, shard_id) in snap.excluded:
        return False
    if worker.resident_shard == shard_id and not opts.ignore_residency:
        return True
    if worker.ram_free is None:
        return True
    available = float(worker.ram_free)
    if worker.resident_shard and not keep_resident:
        resident = snap.catalog.shards.get(worker.resident_shard)
        if resident is not None:
            available += resident.bytes
    return memory_need(snap, worker_id, shard_id, opts) <= available


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageCost:
    shard_id: str
    worker_id: str
    ready: Q
    comp: Q
    comm_in: Q
    route_in: str
    residency: Residency
    finish: Q


@dataclass(frozen=True)
class PipelineCost:
    T_total: Q
    T_steady: Q  # the period P
    C_switch_total: Q
    C_switch_max: Q
    finish_K: Q
    out_leg: Q
    n: int
    stages: Tuple[StageCost, ...]
    stale: bool = False

    @property
    def score(self) -> float:
        return self.T_total.mid


def stage_bytes(snap: Snapshot, shard_ids: Sequence[str], opts: CostOptions) -> Tuple[float, List[float], float]:
    """(input bytes, boundary bytes between consecutive stages, output bytes)."""
    shards = [snap.catalog.shards[s] for s in shard_ids]
    tokens = opts.tokens
    inputs = shards[0].in_act_bytes_per_token * tokens
    boundaries = [shards[i].out_act_bytes_per_token * tokens for i in range(len(shards) - 1)]
    output = shards[-1].out_act_bytes_per_token * tokens
    return inputs, boundaries, output


def pipeline_cost(
    snap: Snapshot,
    stages: Sequence[Tuple[str, str]],
    n: int,
    opts: CostOptions,
    *,
    ready_override: Optional[Sequence[Q]] = None,
) -> PipelineCost:
    """Cost of running ``n`` pending requests through (shard, worker) stages."""
    if not stages:
        raise ValueError("a plan needs at least one stage")
    shard_ids = [s for s, _ in stages]
    inputs, boundaries, output = stage_bytes(snap, shard_ids, opts)
    finish = ZERO
    period = ZERO
    switch_total = ZERO
    switch_max = ZERO
    stale = False
    details: List[StageCost] = []
    seen_workers: Dict[str, List[str]] = {}
    previous: Optional[str] = None
    for index, (shard_id, worker_id) in enumerate(stages):
        if index == 0:
            link = t_comm(snap, FOREMAN, worker_id, inputs, opts) if opts.include_io_legs else LinkCost(ZERO, "none")
        else:
            link = t_comm(snap, previous, worker_id, boundaries[index - 1], opts)
        switch = c_switch(snap, worker_id, shard_id, opts)
        ready = ready_override[index] if ready_override is not None else switch.total
        comp = t_comp(snap, worker_id, shard_id, opts)
        stale = stale or link.stale or switch.stale or compute_stale(snap, worker_id, shard_id, opts)
        if worker_id in seen_workers and opts.allow_time_multiplexing:
            ready = ZERO  # the worker is already busy with an earlier stage
        finish = Q.max(ready, finish + link.time) + comp
        period = Q.max(period, comp, link.time)
        switch_total = switch_total + ready
        switch_max = Q.max(switch_max, ready)
        seen_workers.setdefault(worker_id, []).append(shard_id)
        details.append(StageCost(shard_id, worker_id, ready, comp, link.time, link.route, switch.residency, finish))
        previous = worker_id
    out_leg = ZERO
    if opts.include_io_legs:
        out_link = t_comm(snap, previous, FOREMAN, output, opts)
        out_leg = out_link.time
        period = Q.max(period, out_leg)
        stale = stale or out_link.stale
    if opts.allow_time_multiplexing:
        # A worker serving several stages runs them back to back and, under
        # single residency, swaps shards for every request.
        for worker_id, shards in seen_workers.items():
            if len(shards) < 2:
                continue
            busy = ZERO
            for shard_id in shards:
                busy = busy + t_comp(snap, worker_id, shard_id, opts)
                shard = snap.catalog.shards[shard_id]
                busy = busy + phase_time(snap, worker_id, "unload", shard.bytes, opts)
                busy = busy + phase_time(snap, worker_id, "load", shard.bytes, opts)
                busy = busy + phase_time(snap, worker_id, "init", shard.bytes, opts)
            period = Q.max(period, busy)
    total = finish + out_leg + period.scale(max(0, n - 1))
    return PipelineCost(total, period, switch_total, switch_max, finish, out_leg, n, tuple(details), stale)


def replica_cost(snap: Snapshot, worker_id: str, shard_id: str, opts: CostOptions) -> Tuple[Q, Q, Q]:
    """(ready, single-request latency, period) for a data-parallel replica."""
    shard = snap.catalog.shards[shard_id]
    inputs = shard.in_act_bytes_per_token * opts.tokens
    output = shard.out_act_bytes_per_token * opts.tokens
    comm_in = t_comm(snap, FOREMAN, worker_id, inputs, opts).time
    comm_out = t_comm(snap, worker_id, FOREMAN, output, opts).time
    comp = t_comp(snap, worker_id, shard_id, opts)
    ready = c_switch(snap, worker_id, shard_id, opts).total
    return ready, comm_in + comp + comm_out, Q.max(comm_in, comp, comm_out)
