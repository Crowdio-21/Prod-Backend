"""Adaptive model sizing: look at the memory the phones have free, then pick the largest model that fits.

The planner already adapts the *cut* to free RAM (a phone with less memory gets a smaller shard). That
stops working once the whole model no longer fits across the phones, however finely it is cut. This
module adds the step before it: a ladder of model variants (fp32 > fp16 > int8, or a different model
family such as BERT-base > DistilBERT), ordered by quality. For each planning call it

1. measures the usable capacity: every eligible phone's free RAM minus the headroom,
2. skips variants that cannot fit at all (a cheap lower bound: the smallest composition must fit in the
   combined capacity and every atom must have some shard that fits on some phone),
3. runs the normal planner on the best remaining variant, and walks down the ladder until a plan is
   feasible (and, when a deadline is set, meets it).

To stop the model size flapping as free RAM moves, a running variant is only replaced by a *larger* one
when that larger one still fits with ``upgrade_headroom`` extra memory held back. Moving down is
immediate, because the current variant no longer fits.

Variants come either from real exports (:func:`variant_from_manifest`) or are derived from one export
by precision (:func:`derive_precision_variant`): shard sizes are scaled by the bytes per weight. A
derived variant is a planning estimate; running it needs the matching quantised files on disk.

Everything here is pure: no I/O except :func:`variant_from_manifest`, no clock, no global state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .cost_model import CostOptions, memory_need
from .planner import Objective
from .policies import PolicyDecision, SAwarePolicy
from .types import ShardCatalog, ShardInfo, Snapshot, WorkerView

# Weight bytes relative to fp32. Activations between stages stay fp32 (dynamic quantisation).
PRECISION_BYTES = {"fp32": 1.0, "fp16": 0.5, "int8": 0.25}


@dataclass(frozen=True)
class ModelVariant:
    """One rung of the ladder.

    ``quality`` is the task metric of this variant (e.g. SST-2 accuracy); only its order matters here.
    ``precision`` is informational. ``catalog`` shard ids must be unique across the ladder.
    """

    name: str
    catalog: ShardCatalog
    quality: float
    precision: str = "fp32"
    derived: bool = False  # True when sizes are estimated from another export, not measured

    @property
    def min_bytes(self) -> int:
        """Bytes of the smallest composition (what the phones must hold together)."""
        return min(sum(self.catalog.shards[s].bytes for s in comp) for comp in self.catalog.compositions())

    @property
    def full_bytes(self) -> int:
        return max(sum(self.catalog.shards[s].bytes for s in comp) for comp in self.catalog.compositions())


def derive_precision_variant(base: ModelVariant, precision: str, quality: float, *,
                             compute_factor: float = 1.0, overhead_frac: float = 0.02) -> ModelVariant:
    """A planning variant of ``base`` at another precision.

    Shard bytes scale by the weight width (plus ``overhead_frac`` for scales and zero points);
    ``compute_factor`` scales FLOPs (e.g. 0.6 if int8 kernels run about 1.7x faster on the phones).
    Shard ids get an ``@precision`` suffix so residency of different variants is never confused.
    """
    if precision not in PRECISION_BYTES:
        raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(PRECISION_BYTES)}")
    ratio = PRECISION_BYTES[precision] / PRECISION_BYTES.get(base.precision, 1.0)
    if precision != base.precision:
        ratio *= 1.0 + overhead_frac
    suffix = f"@{precision}"
    rename = {sid: f"{sid}{suffix}" for sid in base.catalog.shards}
    shards = {
        rename[sid]: replace(info, shard_id=rename[sid], bytes=int(math.ceil(info.bytes * ratio)),
                             flops_per_token=info.flops_per_token * compute_factor)
        for sid, info in base.catalog.shards.items()
    }
    catalog = ShardCatalog(
        model_id=f"{base.catalog.model_id}{suffix}",
        n_atoms=base.catalog.n_atoms,
        shards=shards,
        default_composition=tuple(rename[s] for s in base.catalog.default_composition),
        full_shard_id=rename.get(base.catalog.full_shard_id) if base.catalog.full_shard_id else None,
    )
    return ModelVariant(f"{base.name}{suffix}", catalog, quality, precision, derived=True)


def variant_from_manifest(path: Path, quality: float, *, name: Optional[str] = None) -> ModelVariant:
    """A variant from a real export directory (``manifest.json`` or legacy ``partition_metadata.json``)."""
    from .registry import catalog_from_manifest, load_manifest

    manifest, _ = load_manifest(Path(path), hash_files=False)
    precisions = {s.precision for s in manifest.shards}
    precision = precisions.pop() if len(precisions) == 1 else "mixed"
    return ModelVariant(name or manifest.model_id, catalog_from_manifest(manifest), quality, precision)


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capacity:
    """Usable RAM per eligible phone (bytes, after headroom). ``None`` = the phone did not report."""

    per_worker: Mapping[str, Optional[float]]

    @property
    def known(self) -> Dict[str, float]:
        return {w: c for w, c in self.per_worker.items() if c is not None}

    @property
    def all_known(self) -> bool:
        return all(c is not None for c in self.per_worker.values())

    @property
    def total(self) -> float:
        return math.inf if not self.all_known else sum(self.known.values())

    @property
    def largest(self) -> float:
        if not self.per_worker:
            return 0.0
        return math.inf if not self.all_known else max(self.known.values(), default=0.0)

    def as_dict(self) -> dict:
        finite = lambda x: None if math.isinf(x) else x  # noqa: E731
        return {"per_worker": dict(self.per_worker), "total": finite(self.total), "largest": finite(self.largest)}


def _usable(worker: WorkerView, catalogs: Iterable[ShardCatalog], opts: CostOptions) -> Optional[float]:
    if worker.ram_free is None:
        return None
    free = float(worker.ram_free)
    if worker.resident_shard:
        for catalog in catalogs:
            info = catalog.shards.get(worker.resident_shard)
            if info is not None:
                free += info.bytes  # switching unloads it first (single residency)
                break
    return free * (1.0 - opts.ram_headroom_frac)


def measure_capacity(snap: Snapshot, opts: CostOptions, catalogs: Sequence[ShardCatalog] = ()) -> Capacity:
    catalogs = list(catalogs) or [snap.catalog]
    return Capacity({w: _usable(snap.workers[w], catalogs, opts) for w in snap.alive_workers()})


def quick_fit(variant: ModelVariant, snap: Snapshot, capacity: Capacity, opts: CostOptions) -> Tuple[bool, str]:
    """Necessary (not sufficient) conditions for ``variant`` to fit; cheap, so it runs before the search."""
    if not capacity.per_worker:
        return False, "no eligible workers"
    if capacity.all_known and variant.min_bytes > capacity.total:
        return False, f"needs {variant.min_bytes} B, phones have {int(capacity.total)} B usable in total"
    vsnap = snapshot_for(snap, variant, ())
    for atom in range(variant.catalog.n_atoms):
        covering = [s for s in variant.catalog.shards.values() if s.start_atom <= atom <= s.end_atom]
        if not covering:
            return False, f"no shard covers atom {atom}"
        smallest = min(covering, key=lambda s: s.bytes)
        if capacity.all_known and not any(
                memory_need(vsnap, w, smallest.shard_id, opts) <= c for w, c in capacity.known.items()):
            return False, f"atom {atom}: smallest shard {smallest.shard_id} fits on no phone"
    return True, "ok"


def snapshot_for(snap: Snapshot, variant: ModelVariant, ladder: Sequence[ModelVariant]) -> Snapshot:
    """``snap`` re-targeted at ``variant``: a shard of another variant that is resident is treated as
    unloaded (its RAM credited back), and only this variant's files count as cached on disk."""
    others = [v.catalog for v in ladder if v.catalog is not variant.catalog]
    workers = {}
    for wid, view in snap.workers.items():
        resident = view.resident_shard
        ram = view.ram_free
        if resident and resident not in variant.catalog.shards:
            info = next((c.shards[resident] for c in others if resident in c.shards), None)
            if info is not None and ram is not None:
                ram += info.bytes
            resident = None
        workers[wid] = replace(view, resident_shard=resident, ram_free=ram,
                               disk_shards=frozenset(s for s in view.disk_shards if s in variant.catalog.shards))
    return replace(snap, catalog=variant.catalog, workers=workers,
                   excluded=frozenset(e for e in snap.excluded if e[1] in variant.catalog.shards))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantAttempt:
    variant: str
    outcome: str  # chosen | too_big | infeasible | misses_deadline | upgrade_blocked
    detail: str = ""
    T_total: Optional[float] = None


@dataclass(frozen=True)
class AdaptiveDecision:
    variant: Optional[ModelVariant]
    decision: Optional[PolicyDecision]
    capacity: Capacity
    attempts: Tuple[VariantAttempt, ...] = ()
    reason: str = ""

    @property
    def feasible(self) -> bool:
        return self.decision is not None and self.decision.feasible

    @property
    def stages(self):
        return self.decision.stages if self.decision else None

    def summary(self) -> dict:
        return {
            "variant": self.variant.name if self.variant else None,
            "precision": self.variant.precision if self.variant else None,
            "quality": self.variant.quality if self.variant else None,
            "derived": self.variant.derived if self.variant else None,
            "stages": [list(s) for s in (self.stages or ())],
            "T_total": self.decision.cost.T_total.mid if self.feasible and self.decision.cost else None,
            "capacity": self.capacity.as_dict(),
            "attempts": [a.__dict__ for a in self.attempts],
            "reason": self.reason,
        }


@dataclass
class AdaptivePlanner:
    """Chooses the model variant from the phones' free memory, then plans it with ``SAwarePolicy``."""

    ladder: Sequence[ModelVariant]
    opts: CostOptions
    mode: str = "auto"
    # Extra free-RAM fraction a larger variant must leave before moving up. 20% because free RAM on the
    # real phones dipped 7-23% below its mean during runs; a smaller margin invites upgrade-then-downgrade.
    upgrade_headroom: float = 0.20
    current: Optional[str] = None  # name of the variant now running (set by the caller or by plan())
    _policies: Dict[str, SAwarePolicy] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.ladder:
            raise ValueError("the ladder needs at least one variant")
        names = [v.name for v in self.ladder]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        ids = [sid for v in self.ladder for sid in v.catalog.shards]
        if len(set(ids)) != len(ids):
            raise ValueError("shard ids must be unique across variants (derive_precision_variant suffixes them)")
        self.ladder = tuple(sorted(self.ladder, key=lambda v: (-v.quality, v.full_bytes, v.name)))

    def _policy(self, variant: ModelVariant, opts: CostOptions) -> SAwarePolicy:
        key = f"{variant.name}|{opts.ram_headroom_frac}"
        if key not in self._policies:
            self._policies[key] = SAwarePolicy(opts, self.mode, recut=True)
        return self._policies[key]

    def _rank(self, name: Optional[str]) -> Optional[int]:
        return next((i for i, v in enumerate(self.ladder) if v.name == name), None)

    def plan(self, snap: Snapshot, n: int, objective: Objective = Objective(), *, commit: bool = True) -> AdaptiveDecision:
        capacity = measure_capacity(snap, self.opts, [v.catalog for v in self.ladder])
        current_rank = self._rank(self.current)
        strict = replace(self.opts, ram_headroom_frac=min(0.95, self.opts.ram_headroom_frac + self.upgrade_headroom))
        attempts: List[VariantAttempt] = []
        fallback: Optional[Tuple[ModelVariant, PolicyDecision]] = None
        for rank, variant in enumerate(self.ladder):
            ok, why = quick_fit(variant, snap, capacity, self.opts)
            if not ok:
                attempts.append(VariantAttempt(variant.name, "too_big", why))
                continue
            vsnap = snapshot_for(snap, variant, self.ladder)
            upgrading = current_rank is not None and rank < current_rank
            decision = self._policy(variant, strict if upgrading else self.opts).plan(vsnap, None, n, objective)
            if not decision.feasible:
                if upgrading and self._policy(variant, self.opts).plan(vsnap, None, n, objective).feasible:
                    attempts.append(VariantAttempt(variant.name, "upgrade_blocked",
                                                   f"fits, but not with {self.upgrade_headroom:.0%} extra headroom"))
                else:
                    attempts.append(VariantAttempt(variant.name, "infeasible", decision.reason))
                continue
            total = decision.cost.T_total.mid if decision.cost else None
            if objective.deadline_s is not None and decision.cost is not None \
                    and decision.cost.T_total.hi > objective.deadline_s:
                attempts.append(VariantAttempt(variant.name, "misses_deadline",
                                               f"T_total.hi {decision.cost.T_total.hi:.2f}s > {objective.deadline_s}s",
                                               total))
                if fallback is None or total < fallback[1].cost.T_total.mid:
                    fallback = (variant, decision)
                continue
            attempts.append(VariantAttempt(variant.name, "chosen", "", total))
            reason = "largest variant fits" if rank == 0 else f"stepped down {rank} rung(s): larger variants do not fit"
            return self._done(variant, decision, capacity, attempts, commit, reason)
        if fallback is not None:
            variant, decision = fallback
            return self._done(variant, decision, capacity, attempts, commit, "no variant meets the deadline; fastest feasible")
        return AdaptiveDecision(None, None, capacity, tuple(attempts), "no variant fits the free memory")

    def _done(self, variant, decision, capacity, attempts, commit, reason) -> AdaptiveDecision:
        if commit:
            self.current = variant.name
        return AdaptiveDecision(variant, decision, capacity, tuple(attempts), reason)


def precision_ladder(base: ModelVariant, qualities: Mapping[str, float], *,
                     compute_factors: Optional[Mapping[str, float]] = None) -> List[ModelVariant]:
    """``base`` plus derived variants for each precision in ``qualities`` (other than the base's own)."""
    compute_factors = compute_factors or {}
    ladder = [base]
    for precision, quality in qualities.items():
        if precision == base.precision:
            continue
        ladder.append(derive_precision_variant(base, precision, quality,
                                               compute_factor=compute_factors.get(precision, 1.0)))
    return ladder
