"""Immutable value types shared by the pure cost model, planner and controller.

Nothing in this module performs I/O, reads a clock or touches global state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Est:
    """An estimate with its spread, sample count and age (seconds)."""

    mean: float
    std: float = 0.0
    n: int = 0
    age_s: float = math.inf

    @classmethod
    def known(cls, value: float) -> "Est":
        """An exact, fresh value (used by tests and fixed priors)."""
        return cls(value, 0.0, 1_000_000, 0.0)

    @classmethod
    def prior(cls, value: float, rel_std: float = 0.5) -> "Est":
        """A prior: counts as zero samples, so it is stale for decisions."""
        return cls(value, abs(value) * rel_std, 0, math.inf)

    def is_stale(self, max_age_s: float, min_samples: int) -> bool:
        return self.age_s > max_age_s or self.n < min_samples

    def effective_std(self, age_growth: float) -> float:
        """Spread grows with age so old estimates carry wider bounds."""
        age = self.age_s if math.isfinite(self.age_s) else 1e3
        return self.std + abs(self.mean) * age_growth * age


@dataclass(frozen=True)
class Q:
    """An interval quantity (lo <= mid <= hi) propagated through the cost model."""

    lo: float
    mid: float
    hi: float

    @classmethod
    def exact(cls, value: float) -> "Q":
        return cls(value, value, value)

    @classmethod
    def from_est(cls, est: Est, k: float, age_growth: float = 0.0, floor: float = 0.0) -> "Q":
        spread = k * est.effective_std(age_growth)
        return cls(max(floor, est.mean - spread), max(floor, est.mean), max(floor, est.mean + spread))

    def __add__(self, other: "Q") -> "Q":
        return Q(self.lo + other.lo, self.mid + other.mid, self.hi + other.hi)

    def scale(self, factor: float) -> "Q":
        return Q(self.lo * factor, self.mid * factor, self.hi * factor) if factor >= 0 else Q(
            self.hi * factor, self.mid * factor, self.lo * factor)

    @staticmethod
    def max(*values: "Q") -> "Q":
        return Q(max(v.lo for v in values), max(v.mid for v in values), max(v.hi for v in values))

    @staticmethod
    def min(*values: "Q") -> "Q":
        return Q(min(v.lo for v in values), min(v.mid for v in values), min(v.hi for v in values))

    @staticmethod
    def divide(size: float, rate: "Q", floor: float) -> "Q":
        """Time to move ``size`` at an uncertain ``rate``."""
        return Q(size / max(floor, rate.hi), size / max(floor, rate.mid), size / max(floor, rate.lo))

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0


ZERO = Q.exact(0.0)


@dataclass(frozen=True)
class LinearFit:
    """y = a + b * x with residual spread and sample count (x = shard bytes)."""

    a: float
    b: float
    resid_std: float = 0.0
    n: int = 0

    def predict(self, x: float) -> Est:
        value = max(0.0, self.a + self.b * x)
        return Est(value, self.resid_std + 0.0, self.n, 0.0 if self.n else math.inf)


@dataclass(frozen=True)
class DeviceProfile:
    """Learned per-device-class phase models. Times in seconds, memory in bytes."""

    device_class: str
    load: LinearFit
    init: LinearFit
    write: LinearFit
    unload: LinearFit
    peak_mem: LinearFit  # peak RAM during load as a function of shard bytes
    flops_per_s: Est  # compute throughput used when no per-shard measurement exists
    # (shard_id) -> seconds per request at the reference batch/seq
    compute_by_shard: Mapping[str, Est] = field(default_factory=dict)
    ref_tokens: int = 64


@dataclass(frozen=True)
class ShardInfo:
    shard_id: str
    start_atom: int
    end_atom: int
    bytes: int
    flops_per_token: float
    in_act_bytes_per_token: float
    out_act_bytes_per_token: float
    pinned: bool = False
    role: str = "encoder"


@dataclass(frozen=True)
class ShardCatalog:
    model_id: str
    n_atoms: int
    shards: Mapping[str, ShardInfo]
    default_composition: Tuple[str, ...]
    full_shard_id: Optional[str] = None

    def compositions(self) -> Tuple[Tuple[str, ...], ...]:
        """Every ordered tiling of the atoms by exported shards (pinned shards never merged)."""
        starts: Dict[int, list] = {}
        for shard in self.shards.values():
            starts.setdefault(shard.start_atom, []).append(shard)
        pinned_cuts = set()
        for shard in self.shards.values():
            if shard.pinned:
                pinned_cuts.add(shard.start_atom)
                pinned_cuts.add(shard.end_atom + 1)
        results = []

        def walk(atom: int, path: Tuple[str, ...]) -> None:
            if atom == self.n_atoms:
                results.append(path)
                return
            for shard in sorted(starts.get(atom, []), key=lambda s: (s.end_atom, s.shard_id)):
                inner_cuts = {c for c in pinned_cuts if shard.start_atom < c <= shard.end_atom}
                if inner_cuts:
                    continue
                walk(shard.end_atom + 1, path + (shard.shard_id,))

        walk(0, ())
        if self.full_shard_id and (self.full_shard_id,) not in results and self.full_shard_id in self.shards:
            # A full shard always counts as a composition even if it spans pinned cuts.
            results.append((self.full_shard_id,))
        return tuple(results)


@dataclass(frozen=True)
class WorkerView:
    worker_id: str
    device_class: str = "unknown"
    alive: bool = True
    cpu: float = 0.0
    ram_free: Optional[int] = None
    battery: Optional[float] = None
    charging: Optional[bool] = None
    temp: Optional[float] = None
    thermal: Optional[int] = None
    resident_shard: Optional[str] = None
    session_state: str = "IDLE"
    disk_shards: FrozenSet[str] = frozenset()
    disk_used: int = 0
    disk_cap: Optional[int] = None
    up: Est = Est(1e6)  # worker -> Foreman, bytes/s
    down: Est = Est(1e6)  # Foreman -> worker, bytes/s
    rtt: Est = Est(0.05)  # seconds to Foreman
    queue_len: int = 0
    eligible: bool = True  # allowed to run shards under the DNN worker scope


@dataclass(frozen=True)
class LinkView:
    reachable: bool
    bw: Est
    rtt: Est


@dataclass(frozen=True)
class Snapshot:
    """Read-only view handed to pure planners. ``now`` is only used for ages."""

    now: float
    workers: Mapping[str, WorkerView]
    catalog: ShardCatalog
    profiles: Mapping[str, DeviceProfile]
    links: Mapping[Tuple[str, str], LinkView] = field(default_factory=dict)
    capability: Mapping[Tuple[str, str], bool] = field(default_factory=dict)
    excluded: FrozenSet[Tuple[str, str]] = frozenset()  # (worker, shard) after LOAD_FAILED/OOM
    uploads_in_flight: Mapping[str, int] = field(default_factory=dict)
    compression_ratio: float = 0.9

    def holders(self, shard_id: str) -> FrozenSet[str]:
        return frozenset(w.worker_id for w in self.workers.values() if shard_id in w.disk_shards and w.alive)

    def alive_workers(self) -> Tuple[str, ...]:
        """Workers that are connected and allowed to run shards (DNN worker scope)."""
        return tuple(sorted(w for w, v in self.workers.items() if v.alive and v.eligible))

    def with_workers(self, **changes: WorkerView) -> "Snapshot":
        workers = dict(self.workers)
        workers.update(changes)
        return replace(self, workers=workers)
