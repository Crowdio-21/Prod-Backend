"""Builders for small, exactly-known planning instances."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, Iterable, Optional, Sequence

from foreman.planning.cost_model import CostOptions
from foreman.planning.types import (
    DeviceProfile,
    Est,
    LinearFit,
    LinkView,
    ShardCatalog,
    ShardInfo,
    Snapshot,
    WorkerView,
)

MB = 1_000_000


def exact(value: float) -> Est:
    return Est.known(value)


def fit(seconds: float, per_byte: float = 0.0) -> LinearFit:
    return LinearFit(seconds, per_byte, 0.0, 100)


def profile(device_class: str = "phone", *, load=1.0, init=0.5, write=0.2, unload=0.3,
            peak_mult=1.0, flops_per_s=1e9, compute: Optional[Dict[str, float]] = None) -> DeviceProfile:
    return DeviceProfile(
        device_class=device_class,
        load=fit(load), init=fit(init), write=fit(write), unload=fit(unload),
        peak_mem=LinearFit(0.0, peak_mult, 0.0, 100),
        flops_per_s=exact(flops_per_s),
        compute_by_shard={k: exact(v) for k, v in (compute or {}).items()},
        ref_tokens=1,
    )


def shard(shard_id: str, start: int, end: int, *, size: int = 100 * MB, act: float = 1000.0,
          pinned: bool = False, role: str = "encoder", in_act: Optional[float] = None) -> ShardInfo:
    return ShardInfo(shard_id, start, end, size, 1e6, act if in_act is None else in_act, act, pinned, role)


def three_stage_catalog(*, with_full: bool = False, layer_shards: bool = False) -> ShardCatalog:
    """Atoms 0..3: a=[0], b=[1,2], c=[3]; optional finer cuts and a full shard."""
    shards = {
        "a": shard("a", 0, 0),
        "b": shard("b", 1, 2),
        "c": shard("c", 3, 3, size=10 * MB),
    }
    if layer_shards:
        shards["b1"] = shard("b1", 1, 1, size=50 * MB)
        shards["b2"] = shard("b2", 2, 2, size=50 * MB)
        shards["ab1"] = shard("ab1", 0, 1, size=150 * MB)
    if with_full:
        shards["full"] = shard("full", 0, 3, size=210 * MB, role="full")
    return ShardCatalog("m", 4, shards, ("a", "b", "c"), "full" if with_full else None)


def worker(worker_id: str, *, device_class: str = "phone", bw: float = 10 * MB, rtt: float = 0.1,
           resident: Optional[str] = None, disk: Iterable[str] = (), ram: Optional[int] = None,
           battery: Optional[float] = 80.0, thermal: Optional[int] = 0, cpu: float = 0.0,
           alive: bool = True) -> WorkerView:
    return WorkerView(
        worker_id=worker_id, device_class=device_class, alive=alive, cpu=cpu, ram_free=ram,
        battery=battery, charging=False, temp=None, thermal=thermal, resident_shard=resident,
        disk_shards=frozenset(disk), up=exact(bw), down=exact(bw), rtt=exact(rtt),
    )


def snapshot(workers: Sequence[WorkerView], catalog: Optional[ShardCatalog] = None,
             profiles: Optional[Dict[str, DeviceProfile]] = None, links=None, capability=None,
             compression: float = 1.0, now: float = 0.0, **kw) -> Snapshot:
    return Snapshot(
        now=now,
        workers={w.worker_id: w for w in workers},
        catalog=catalog or three_stage_catalog(),
        profiles=profiles or {"phone": profile(), "default": profile("default")},
        links=links or {},
        capability=capability or {},
        compression_ratio=compression,
        **kw,
    )


def link(bw: float, rtt: float, reachable: bool = True) -> LinkView:
    return LinkView(reachable, exact(bw), exact(rtt))


def opts(**kw) -> CostOptions:
    defaults = dict(k=2.0, age_growth=0.0, seq_len=1, batch=1, runtime_overhead_bytes=0,
                    min_samples=1, max_age_s=1e9)
    defaults.update(kw)
    return CostOptions(**defaults)


def random_snapshot(rng: random.Random, n_workers: int = 4, *, layer_shards: bool = True,
                    with_full: bool = True) -> Snapshot:
    catalog = three_stage_catalog(with_full=with_full, layer_shards=layer_shards)
    classes = {f"c{i}": profile(f"c{i}", load=rng.uniform(0.5, 5), init=rng.uniform(0.1, 2),
                                write=rng.uniform(0.1, 1), unload=rng.uniform(0.05, 0.5),
                                flops_per_s=rng.uniform(2e8, 5e9)) for i in range(3)}
    workers = []
    for i in range(n_workers):
        resident = rng.choice([None] + list(catalog.shards))
        disk = {s for s in catalog.shards if rng.random() < 0.3}
        ram = rng.choice([None, 120 * MB, 260 * MB, 400 * MB])
        workers.append(worker(
            f"w{i}", device_class=f"c{i % 3}", bw=rng.uniform(1, 50) * MB, rtt=rng.uniform(0.005, 0.2),
            resident=resident, disk=disk, ram=ram,
        ))
    return snapshot(workers, catalog, classes)
