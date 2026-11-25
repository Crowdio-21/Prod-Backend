"""Pure planner: ordered assignments of distinct workers to stages, and
compositions of atoms into stages restricted to exported shards.

* :func:`exhaustive_search` is the exact oracle. Memory feasibility prunes
  first; a branch-and-bound cut on T_total (monotone in the stages added) prunes
  next.
* :func:`pareto_dp_search` keeps a Pareto set of (finish, period, switch) per
  state (boundary atom, last shard, last worker, used workers). It is exact for
  the mid-point objective because the objective mixes a sum (finish) and a max
  (period) and both are monotone.
* :func:`scalar_dp_search` drops the used-worker set and keeps one scalarised
  label per state; it scales better but is *not* exact. Use
  :func:`dp_gap` to report its gap against exhaustive.
* :func:`local_search` is incremental repartitioning: best-improvement hill
  climbing from the current plan over small moves (replace a worker with a
  standby, swap two stages' workers, shift a cut by one atom, merge or split a
  stage) plus one larger move per exported composition (re-tile: take that
  cut and assign workers greedily, keeping residency where it can). The small
  moves change one or two shards, so most of the plan keeps its residency and
  its switch cost stays zero. Not exact; pair it with an occasional full search
  and report :func:`local_gap`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .cost_model import (
    FOREMAN,
    CostOptions,
    PipelineCost,
    c_switch,
    memory_feasible,
    pipeline_cost,
    t_comm,
    t_comp,
)
from .types import ZERO, Q, Snapshot

Stages = Tuple[Tuple[str, str], ...]  # ((shard_id, worker_id), ...)


@dataclass(frozen=True)
class Objective:
    kind: str = "min_total"  # min_total | min_steady_deadline
    deadline_s: Optional[float] = None

    def key(self, cost: PipelineCost) -> Tuple[float, ...]:
        if self.kind == "min_steady_deadline":
            met = self.deadline_s is not None and cost.T_total.hi <= self.deadline_s
            # Deadline-feasible plans first, then by period, then by total.
            return (0.0 if met else 1.0, cost.T_steady.mid if met else cost.T_total.mid, cost.T_total.mid)
        return (cost.T_total.mid, cost.T_steady.mid)

    def bound(self, partial_total: float) -> Tuple[float, ...]:
        # Only a valid lower bound for min_total.
        return (partial_total, 0.0)


@dataclass(frozen=True)
class Candidate:
    stages: Stages
    cost: PipelineCost


@dataclass(frozen=True)
class SearchResult:
    best: Optional[Candidate]
    ranked: Tuple[Candidate, ...]
    evaluated: int
    pruned_memory: int
    pruned_bound: int
    method: str
    compositions: int = 0

    @property
    def found(self) -> bool:
        return self.best is not None


def _feasible_workers(snap: Snapshot, shard_id: str, opts: CostOptions, keep_resident: bool) -> List[str]:
    return [w for w in snap.alive_workers() if memory_feasible(snap, w, shard_id, opts, keep_resident)]


def exhaustive_search(
    snap: Snapshot,
    n: int,
    opts: CostOptions,
    objective: Objective = Objective(),
    *,
    compositions: Optional[Sequence[Tuple[str, ...]]] = None,
    keep_top: int = 5,
    keep_resident: bool = False,
    use_bound: bool = True,
) -> SearchResult:
    compositions = tuple(compositions) if compositions is not None else snap.catalog.compositions()
    ranked: List[Candidate] = []
    evaluated = pruned_memory = pruned_bound = 0
    best_key: Tuple[float, ...] = (math.inf,)
    can_bound = use_bound and objective.kind == "min_total" and not opts.allow_time_multiplexing
    for composition in compositions:
        feasible = [_feasible_workers(snap, s, opts, keep_resident) for s in composition]
        total_pairs = len(composition) * len(snap.alive_workers())
        pruned_memory += total_pairs - sum(len(f) for f in feasible)
        if any(not f for f in feasible):
            continue

        def dfs(index: int, chosen: List[str]) -> None:
            nonlocal evaluated, pruned_bound, best_key
            if index == len(composition):
                stages = tuple(zip(composition, chosen))
                cost = pipeline_cost(snap, stages, n, opts)
                evaluated += 1
                key = objective.key(cost)
                ranked.append(Candidate(stages, cost))
                if key < best_key:
                    best_key = key
                return
            for worker in feasible[index]:
                if worker in chosen and not opts.allow_time_multiplexing:
                    continue
                prefix = tuple(zip(composition[: index + 1], chosen + [worker]))
                if can_bound and index + 1 < len(composition):
                    partial = _partial_total(snap, prefix, n, opts)
                    if (partial,) >= best_key[:1]:
                        pruned_bound += 1
                        continue
                dfs(index + 1, chosen + [worker])

        dfs(0, [])
    ranked.sort(key=lambda c: (objective.key(c.cost), c.stages))
    top = tuple(ranked[:keep_top])
    return SearchResult(top[0] if top else None, top, evaluated, pruned_memory, pruned_bound,
                        "exhaustive", len(compositions))


def _partial_total(snap: Snapshot, prefix: Stages, n: int, opts: CostOptions) -> float:
    """Lower bound on T_total for any completion of ``prefix``.

    finish and period only grow as stages are appended, so
    finish_prefix + (n - 1) * period_prefix is a valid lower bound. The input
    leg is included; the output leg is not (it is added later).
    """
    finish = 0.0
    period = 0.0
    previous = None
    from .cost_model import stage_bytes

    shard_ids = [s for s, _ in prefix]
    inputs, boundaries, _ = stage_bytes(snap, shard_ids, opts)
    for index, (shard_id, worker_id) in enumerate(prefix):
        if index == 0:
            comm = t_comm(snap, FOREMAN, worker_id, inputs, opts).time.mid if opts.include_io_legs else 0.0
        else:
            comm = t_comm(snap, previous, worker_id, boundaries[index - 1], opts).time.mid
        ready = c_switch(snap, worker_id, shard_id, opts).total.mid
        comp = t_comp(snap, worker_id, shard_id, opts).mid
        finish = max(ready, finish + comm) + comp
        period = max(period, comp, comm)
        previous = worker_id
    return finish + max(0, n - 1) * period


# ---------------------------------------------------------------------------
# Dynamic programmes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Label:
    finish: float
    period: float
    path: Stages


def _transitions(snap: Snapshot, boundary: int, allowed: Optional[FrozenSet[Tuple[int, str]]]):
    for shard in snap.catalog.shards.values():
        if shard.start_atom != boundary:
            continue
        if allowed is not None and (boundary, shard.shard_id) not in allowed:
            continue
        yield shard


def _allowed_steps(compositions: Optional[Sequence[Tuple[str, ...]]], snap: Snapshot) -> Optional[FrozenSet[Tuple[int, str]]]:
    if compositions is None:
        compositions = snap.catalog.compositions()
    steps = set()
    for composition in compositions:
        for shard_id in composition:
            steps.add((snap.catalog.shards[shard_id].start_atom, shard_id))
    return frozenset(steps)


def _step(snap: Snapshot, opts: CostOptions, label: _Label, prev: Optional[Tuple[str, str]], shard, worker: str) -> _Label:
    if prev is None:
        n_bytes = shard.in_act_bytes_per_token * opts.tokens
        comm = t_comm(snap, FOREMAN, worker, n_bytes, opts).time.mid if opts.include_io_legs else 0.0
    else:
        prev_shard = snap.catalog.shards[prev[0]]
        comm = t_comm(snap, prev[1], worker, prev_shard.out_act_bytes_per_token * opts.tokens, opts).time.mid
    ready = c_switch(snap, worker, shard.shard_id, opts).total.mid
    comp = t_comp(snap, worker, shard.shard_id, opts).mid
    return _Label(max(ready, label.finish + comm) + comp, max(label.period, comp, comm),
                  label.path + ((shard.shard_id, worker),))


def _close(snap: Snapshot, opts: CostOptions, label: _Label, n: int) -> float:
    last_shard, last_worker = label.path[-1]
    out = 0.0
    period = label.period
    if opts.include_io_legs:
        shard = snap.catalog.shards[last_shard]
        out = t_comm(snap, last_worker, FOREMAN, shard.out_act_bytes_per_token * opts.tokens, opts).time.mid
        period = max(period, out)
    return label.finish + out + max(0, n - 1) * period


def pareto_dp_search(
    snap: Snapshot,
    n: int,
    opts: CostOptions,
    objective: Objective = Objective(),
    *,
    compositions: Optional[Sequence[Tuple[str, ...]]] = None,
    keep_resident: bool = False,
) -> SearchResult:
    """Exact for the mid-point min_total objective (distinct workers)."""
    allowed = _allowed_steps(compositions, snap)
    n_atoms = snap.catalog.n_atoms
    # state: (boundary, last shard, last worker, used) -> Pareto labels
    frontier: Dict[Tuple[int, Optional[str], Optional[str], FrozenSet[str]], List[_Label]] = {
        (0, None, None, frozenset()): [_Label(0.0, 0.0, ())]
    }
    finals: List[_Label] = []
    evaluated = pruned_memory = 0
    for boundary in range(n_atoms):
        keys = [k for k in frontier if k[0] == boundary]
        for key in keys:
            labels = frontier.pop(key)
            _, last_shard, last_worker, used = key
            prev = (last_shard, last_worker) if last_shard is not None else None
            for shard in _transitions(snap, boundary, allowed):
                for worker in snap.alive_workers():
                    if worker in used:
                        continue
                    if not memory_feasible(snap, worker, shard.shard_id, opts, keep_resident):
                        pruned_memory += 1
                        continue
                    for label in labels:
                        new = _step(snap, opts, label, prev, shard, worker)
                        evaluated += 1
                        target = (shard.end_atom + 1, shard.shard_id, worker, used | {worker})
                        if target[0] == n_atoms:
                            finals.append(new)
                        else:
                            frontier[target] = _pareto_insert(frontier.get(target, []), new)
    return _finish_dp(snap, n, opts, objective, finals, evaluated, pruned_memory, "pareto_dp")


def _pareto_insert(labels: List[_Label], new: _Label) -> List[_Label]:
    for label in labels:
        if label.finish <= new.finish and label.period <= new.period:
            return labels
    return [l for l in labels if not (new.finish <= l.finish and new.period <= l.period)] + [new]


def scalar_dp_search(
    snap: Snapshot,
    n: int,
    opts: CostOptions,
    objective: Objective = Objective(),
    *,
    compositions: Optional[Sequence[Tuple[str, ...]]] = None,
    keep_resident: bool = False,
) -> SearchResult:
    """Scalarised DP without the used-worker set (heuristic; report its gap)."""
    allowed = _allowed_steps(compositions, snap)
    n_atoms = snap.catalog.n_atoms
    best: Dict[Tuple[int, Optional[str], Optional[str]], _Label] = {(0, None, None): _Label(0.0, 0.0, ())}
    finals: List[_Label] = []
    evaluated = pruned_memory = 0
    scal = lambda l: l.finish + max(0, n - 1) * l.period  # noqa: E731
    for boundary in range(n_atoms):
        for key in [k for k in best if k[0] == boundary]:
            label = best.pop(key)
            _, last_shard, last_worker = key
            prev = (last_shard, last_worker) if last_shard is not None else None
            used = {w for _, w in label.path}
            for shard in _transitions(snap, boundary, allowed):
                for worker in snap.alive_workers():
                    if worker in used:
                        continue
                    if not memory_feasible(snap, worker, shard.shard_id, opts, keep_resident):
                        pruned_memory += 1
                        continue
                    new = _step(snap, opts, label, prev, shard, worker)
                    evaluated += 1
                    target = (shard.end_atom + 1, shard.shard_id, worker)
                    if target[0] == n_atoms:
                        finals.append(new)
                    elif target not in best or scal(new) < scal(best[target]):
                        best[target] = new
    return _finish_dp(snap, n, opts, objective, finals, evaluated, pruned_memory, "scalar_dp")


def _finish_dp(snap, n, opts, objective, finals, evaluated, pruned_memory, method) -> SearchResult:
    finals.sort(key=lambda l: (_close(snap, opts, l, n), l.path))
    # Re-cost the best few with the full interval model for reporting.
    ranked = [Candidate(l.path, pipeline_cost(snap, l.path, n, opts)) for l in finals[:20]]
    ranked.sort(key=lambda c: (objective.key(c.cost), c.stages))
    ranked = ranked[:5]
    return SearchResult(ranked[0] if ranked else None, tuple(ranked), evaluated, pruned_memory, 0, method)


# ---------------------------------------------------------------------------
# Incremental repartitioning (local search)
# ---------------------------------------------------------------------------

LOCAL_MOVES = ("replace", "swap", "shift", "merge", "split", "retile")


class _Neighbourhood:
    """Feasibility and shard lookups shared by every move of one local search."""

    def __init__(self, snap: Snapshot, opts: CostOptions, keep_resident: bool) -> None:
        self.snap = snap
        self.opts = opts
        self.keep_resident = keep_resident
        self.alive = list(snap.alive_workers())
        self.valid = frozenset(snap.catalog.compositions())
        self.by_range: Dict[Tuple[int, int], str] = {}
        for shard in sorted(snap.catalog.shards.values(), key=lambda s: s.shard_id):
            self.by_range.setdefault((shard.start_atom, shard.end_atom), shard.shard_id)
        self._fits: Dict[Tuple[str, str], bool] = {}

    def fits(self, worker: str, shard_id: str) -> bool:
        key = (worker, shard_id)
        if key not in self._fits:
            self._fits[key] = memory_feasible(self.snap, worker, shard_id, self.opts, self.keep_resident)
        return self._fits[key]

    def feasible(self, stages: Stages) -> bool:
        if tuple(s for s, _ in stages) not in self.valid:
            return False
        workers = [w for _, w in stages]
        if not self.opts.allow_time_multiplexing and len(set(workers)) != len(workers):
            return False
        alive = set(self.alive)
        return all(w in alive and self.fits(w, s) for s, w in stages)

    def span(self, shard_id: str) -> Tuple[int, int]:
        shard = self.snap.catalog.shards[shard_id]
        return shard.start_atom, shard.end_atom

    def moves(self, stages: Stages):
        """Yield (move, stages) for every plan one small move away from ``stages``."""
        used = {w for _, w in stages}
        spare = [w for w in self.alive if w not in used or self.opts.allow_time_multiplexing]
        k = len(stages)
        for i, (shard_id, worker) in enumerate(stages):
            for other in spare:
                if other != worker:
                    yield "replace", stages[:i] + ((shard_id, other),) + stages[i + 1:]
        for i in range(k):
            for j in range(i + 1, k):
                swapped = list(stages)
                swapped[i] = (stages[i][0], stages[j][1])
                swapped[j] = (stages[j][0], stages[i][1])
                yield "swap", tuple(swapped)
        for i in range(k - 1):
            (left, wl), (right, wr) = stages[i], stages[i + 1]
            start, end = self.span(left)
            _, right_end = self.span(right)
            for new_end in (end - 1, end + 1):
                if new_end < start or new_end >= right_end:
                    continue
                a = self.by_range.get((start, new_end))
                b = self.by_range.get((new_end + 1, right_end))
                if a and b:
                    yield "shift", stages[:i] + ((a, wl), (b, wr)) + stages[i + 2:]
            merged = self.by_range.get((start, right_end))
            if merged:
                for w in (wl, wr):
                    yield "merge", stages[:i] + ((merged, w),) + stages[i + 2:]
        for i, (shard_id, worker) in enumerate(stages):
            start, end = self.span(shard_id)
            for cut in range(start + 1, end + 1):
                a = self.by_range.get((start, cut - 1))
                b = self.by_range.get((cut, end))
                if not (a and b):
                    continue
                for other in spare:
                    if other == worker:
                        continue
                    yield "split", stages[:i] + ((a, worker), (b, other)) + stages[i + 1:]
                    yield "split", stages[:i] + ((a, other), (b, worker)) + stages[i + 1:]

        current = tuple(s for s, _ in stages)
        for composition in sorted(self.valid):
            if composition != current:
                retiled = self.assign(composition, keep=dict((s, w) for s, w in stages))
                if retiled is not None:
                    yield "retile", retiled

    def assign(self, composition: Tuple[str, ...], keep: Dict[str, str]) -> Optional[Stages]:
        """Greedy workers for ``composition``: a shard keeps its current worker,
        otherwise takes the free worker with the cheapest switch."""
        alive = set(self.alive)
        chosen: List[Optional[str]] = [None] * len(composition)
        for i, shard_id in enumerate(composition):
            worker = keep.get(shard_id)
            if worker in alive and self.fits(worker, shard_id) and worker not in chosen:
                chosen[i] = worker
        for i, shard_id in enumerate(composition):
            if chosen[i] is not None:
                continue
            options = [w for w in self.alive if (w not in chosen or self.opts.allow_time_multiplexing)
                       and self.fits(w, shard_id)]
            if not options:
                return None
            chosen[i] = min(options, key=lambda w: (c_switch(self.snap, w, shard_id, self.opts).total.mid, w))
        return tuple(zip(composition, chosen))

    def repair(self, stages: Stages) -> Optional[Stages]:
        """Move each stage whose worker is dead or cannot hold its shard to the
        cheapest-to-switch standby. Keeps the cut. None if no standby fits."""
        alive = set(self.alive)
        fixed = list(stages)
        for i, (shard_id, worker) in enumerate(fixed):
            if worker in alive and self.fits(worker, shard_id):
                continue
            used = {w for _, w in fixed}
            options = [w for w in self.alive if (w not in used or self.opts.allow_time_multiplexing)
                       and self.fits(w, shard_id)]
            if not options:
                return None
            best = min(options, key=lambda w: (c_switch(self.snap, w, shard_id, self.opts).total.mid, w))
            fixed[i] = (shard_id, best)
        fixed_stages = tuple(fixed)
        return fixed_stages if self.feasible(fixed_stages) else None


def local_search(
    snap: Snapshot,
    n: int,
    opts: CostOptions,
    objective: Objective = Objective(),
    *,
    start: Stages,
    keep_top: int = 5,
    keep_resident: bool = False,
    max_rounds: int = 50,
) -> SearchResult:
    """Best-improvement hill climbing from ``start`` (usually the current plan).

    A start whose workers died or no longer fit is first repaired by moving the
    broken stages to standbys. The result is a local optimum: no single move
    improves the objective. The full search remains the reference.
    """
    hood = _Neighbourhood(snap, opts, keep_resident)
    current = hood.repair(tuple(start)) if start else None
    if current is None:
        return SearchResult(None, (), 0, 0, 0, "local")
    seen: Dict[Stages, Candidate] = {}

    def price(stages: Stages) -> Candidate:
        if stages not in seen:
            seen[stages] = Candidate(stages, pipeline_cost(snap, stages, n, opts))
        return seen[stages]

    best = price(current)
    for _ in range(max_rounds):
        step: Optional[Candidate] = None
        for _, stages in hood.moves(best.stages):
            if stages in seen or not hood.feasible(stages):
                continue
            candidate = price(stages)
            if step is None or (objective.key(candidate.cost), candidate.stages) < (objective.key(step.cost), step.stages):
                step = candidate
        if step is None or objective.key(step.cost) >= objective.key(best.cost):
            break
        best = step
    ranked = sorted(seen.values(), key=lambda c: (objective.key(c.cost), c.stages))[:keep_top]
    return SearchResult(best, tuple(ranked), len(seen), 0, 0, "local", len({tuple(s for s, _ in c) for c in seen}))


def local_gap(snap: Snapshot, n: int, opts: CostOptions, start: Stages,
              objective: Objective = Objective()) -> Tuple[float, float, float]:
    """(exhaustive T_total, local-search T_total, relative gap) for one instance."""
    exact = exhaustive_search(snap, n, opts, objective)
    approx = local_search(snap, n, opts, objective, start=start)
    if not exact.found or not approx.found:
        return (math.inf, math.inf, 0.0 if exact.found == approx.found else math.inf)
    a, b = exact.best.cost.T_total.mid, approx.best.cost.T_total.mid
    return a, b, (b - a) / a if a > 0 else 0.0


# "auto" search limits: exhaustive search tries every assignment of workers to every composition, which is
# fine for a handful of phones and takes minutes for ten (BERT-base, 24 shards: 0.9 s at 5 phones, 24 s at 7).
AUTO_EXHAUSTIVE_LIMIT = 20_000  # (composition, assignment) pairs
AUTO_PARETO_MAX_WORKERS = 7  # exact DP, about a second at 7 phones and growing fast


def exhaustive_work(snap: Snapshot, compositions=None) -> int:
    """How many (composition, worker assignment) pairs an exhaustive search would price."""
    workers = len(list(snap.alive_workers()))
    total = 0
    for composition in (compositions if compositions is not None else snap.catalog.compositions()):
        k = len(composition)
        if k > workers:
            continue
        ways = 1
        for i in range(k):
            ways *= workers - i
        total += ways
    return total


def auto_search(snap: Snapshot, n: int, opts: CostOptions, objective: Objective, *, compositions=None,
                keep_resident: bool = False) -> SearchResult:
    """Pick the search by instance size so planning never stalls the Foreman: exhaustive when small, the exact
    Pareto DP up to a few phones, then the scalar DP polished by local search (always feasible, near-optimal)."""
    if objective.kind != "min_total" or exhaustive_work(snap, compositions) <= AUTO_EXHAUSTIVE_LIMIT:
        return exhaustive_search(snap, n, opts, objective, compositions=compositions, keep_resident=keep_resident)
    if len(list(snap.alive_workers())) <= AUTO_PARETO_MAX_WORKERS:
        return pareto_dp_search(snap, n, opts, objective, compositions=compositions, keep_resident=keep_resident)
    result = scalar_dp_search(snap, n, opts, objective, compositions=compositions, keep_resident=keep_resident)
    if result.found and compositions is None:
        polished = local_search(snap, n, opts, objective, start=result.best.stages, keep_resident=keep_resident)
        if polished.found and objective.key(polished.best.cost) < objective.key(result.best.cost):
            return polished
    return result


def search(snap: Snapshot, n: int, opts: CostOptions, objective: Objective, *, mode: str = "exhaustive",
           compositions=None, keep_resident: bool = False) -> SearchResult:
    if mode == "auto":
        return auto_search(snap, n, opts, objective, compositions=compositions, keep_resident=keep_resident)
    if mode == "pareto_dp" and objective.kind == "min_total":
        return pareto_dp_search(snap, n, opts, objective, compositions=compositions, keep_resident=keep_resident)
    if mode == "scalar_dp" and objective.kind == "min_total":
        return scalar_dp_search(snap, n, opts, objective, compositions=compositions, keep_resident=keep_resident)
    return exhaustive_search(snap, n, opts, objective, compositions=compositions, keep_resident=keep_resident)


def dp_gap(snap: Snapshot, n: int, opts: CostOptions, method: str = "scalar_dp") -> Tuple[float, float, float]:
    """(exhaustive T_total, DP T_total, relative gap) for one instance."""
    exact = exhaustive_search(snap, n, opts)
    approx = (scalar_dp_search if method == "scalar_dp" else pareto_dp_search)(snap, n, opts)
    if not exact.found or not approx.found:
        return (math.inf, math.inf, 0.0 if exact.found == approx.found else math.inf)
    a, b = exact.best.cost.T_total.mid, approx.best.cost.T_total.mid
    return a, b, (b - a) / a if a > 0 else 0.0
