"""Pure prefetch planning: download shard files to standby disks, never load them.

Candidates are (a) the stages of the planner's next-best plans (a likely
switch) and (b) each current stage's backup workers. A prefetch is only issued
when the worker's link is quiet, its battery and heat allow it, it is not
loading, and the file fits under its disk cap after LRU eviction (the resident
shard's file is never evicted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import PrefetchConfig
from .contract import Plan
from .registry import DiskEntry, plan_evictions
from .types import Snapshot


@dataclass(frozen=True)
class PrefetchOrder:
    worker_id: str
    shard_id: str
    evict: Tuple[str, ...]
    reason: str


def plan_prefetch(
    snap: Snapshot,
    plan: Optional[Plan],
    likely: Sequence[Sequence[Tuple[str, str]]],
    cfg: PrefetchConfig,
    *,
    link_utilisation: Mapping[str, float],
    in_flight: Iterable[Tuple[str, str]] = (),
    last_used: Optional[Mapping[Tuple[str, str], float]] = None,
) -> List[PrefetchOrder]:
    last_used = last_used or {}
    in_flight = set(in_flight)
    wanted: List[Tuple[str, str, str]] = []
    for rank, stages in enumerate(likely[: cfg.top_candidates]):
        for shard_id, worker_id in stages:
            wanted.append((worker_id, shard_id, f"candidate#{rank}"))
    if plan is not None:
        for stage in plan.stages:
            for backup in stage.backups:
                wanted.append((backup, stage.shard_id, f"backup:stage{stage.stage_index}"))
    orders: List[PrefetchOrder] = []
    seen = set()
    for worker_id, shard_id, reason in wanted:
        if (worker_id, shard_id) in seen or (worker_id, shard_id) in in_flight:
            continue
        seen.add((worker_id, shard_id))
        worker = snap.workers.get(worker_id)
        shard = snap.catalog.shards.get(shard_id)
        if worker is None or shard is None or not worker.alive:
            continue
        if shard_id in worker.disk_shards or worker.resident_shard == shard_id:
            continue
        if worker.session_state in ("LOADING", "UNLOADING"):
            continue
        if worker.battery is not None and not worker.charging and worker.battery < cfg.min_battery:
            continue
        if worker.thermal is not None and worker.thermal > cfg.max_thermal:
            continue
        if link_utilisation.get(worker_id, 0.0) > cfg.max_link_utilisation:
            continue
        entries = [
            DiskEntry(s, snap.catalog.shards[s].bytes if s in snap.catalog.shards else 0,
                      last_used.get((worker_id, s), 0.0))
            for s in worker.disk_shards
        ]
        protected = {worker.resident_shard} if worker.resident_shard else set()
        protected |= {s for w, s in in_flight if w == worker_id}
        evict = plan_evictions(entries, worker.disk_used, worker.disk_cap, shard.bytes, protected)
        if evict is None:
            continue
        orders.append(PrefetchOrder(worker_id, shard_id, tuple(evict), reason))
    return orders
