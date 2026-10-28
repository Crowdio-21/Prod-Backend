"""Planner and policy properties on random small instances (seeded, no hypothesis dep)."""

import random
from dataclasses import replace

import pytest

from foreman.planning.cost_model import memory_feasible
from foreman.planning.planner import (
    Objective,
    dp_gap,
    exhaustive_search,
    pareto_dp_search,
    scalar_dp_search,
)
from foreman.planning.policies import (
    B0Policy,
    B1Policy,
    B2Policy,
    B4Policy,
    SAwarePolicy,
    affinity_tier,
    build_plan,
)
from tests.planning.fixtures import MB, opts, random_snapshot, snapshot, three_stage_catalog, worker

SEEDS = range(40)


def _check_feasible(snap, stages, o):
    workers = [w for _, w in stages]
    assert len(set(workers)) == len(workers), "distinct workers per stage (single residency)"
    covered = []
    for shard_id, worker_id in stages:
        assert memory_feasible(snap, worker_id, shard_id, o)
        shard = snap.catalog.shards[shard_id]
        covered.extend(range(shard.start_atom, shard.end_atom + 1))
    assert covered == list(range(snap.catalog.n_atoms)), "stages tile the atoms in order"


@pytest.mark.parametrize("seed", SEEDS)
def test_exhaustive_plans_are_feasible(seed):
    rng = random.Random(seed)
    snap = random_snapshot(rng, n_workers=rng.randint(2, 5))
    o = opts()
    result = exhaustive_search(snap, rng.randint(1, 50), o)
    for candidate in result.ranked:
        _check_feasible(snap, candidate.stages, o)


@pytest.mark.parametrize("seed", SEEDS)
def test_bound_pruning_does_not_change_the_optimum(seed):
    rng = random.Random(seed)
    snap = random_snapshot(rng, n_workers=4)
    n = rng.randint(1, 30)
    a = exhaustive_search(snap, n, opts(), use_bound=True)
    b = exhaustive_search(snap, n, opts(), use_bound=False)
    assert a.found == b.found
    if a.found:
        assert a.best.cost.T_total.mid == pytest.approx(b.best.cost.T_total.mid)


@pytest.mark.parametrize("seed", SEEDS)
def test_pareto_dp_matches_exhaustive(seed):
    rng = random.Random(seed)
    snap = random_snapshot(rng, n_workers=rng.randint(2, 5))
    n = rng.randint(1, 40)
    exact = exhaustive_search(snap, n, opts())
    dp = pareto_dp_search(snap, n, opts())
    assert exact.found == dp.found
    if exact.found:
        assert dp.best.cost.T_total.mid == pytest.approx(exact.best.cost.T_total.mid, rel=1e-9)
        _check_feasible(snap, dp.best.stages, opts())


def test_scalar_dp_gap_is_reported_not_assumed_zero():
    gaps = []
    for seed in SEEDS:
        rng = random.Random(seed)
        snap = random_snapshot(rng, n_workers=4)
        exact, approx, gap = dp_gap(snap, rng.randint(1, 40), opts(), "scalar_dp")
        if exact != float("inf"):
            assert gap >= -1e-9  # never better than the oracle
            gaps.append(gap)
    assert gaps  # the harness reports max/mean gap; exactness is not claimed


def test_memory_filter_prunes_before_scoring():
    snap = snapshot([worker("big", ram=500 * MB), worker("tiny", ram=1 * MB), worker("mid", ram=500 * MB)])
    result = exhaustive_search(snap, 10, opts())
    assert result.pruned_memory > 0
    assert all("tiny" not in [w for _, w in c.stages] for c in result.ranked)


def test_switching_cost_zero_saware_equals_b2():
    for seed in SEEDS:
        rng = random.Random(seed)
        snap = random_snapshot(rng, n_workers=4)
        o = opts(zero_switch=True)
        n = rng.randint(1, 40)
        ours = SAwarePolicy(o).plan(snap, None, n, Objective())
        b2 = B2Policy(opts()).plan(snap, None, n, Objective())
        assert ours.feasible == b2.feasible
        if ours.feasible:
            assert ours.cost.T_total.mid == pytest.approx(b2.cost.T_total.mid)


def test_bandwidth_terms_zero_b1_ranks_like_b0():
    for seed in SEEDS:
        rng = random.Random(seed)
        snap = random_snapshot(rng, n_workers=5)
        b0 = B0Policy(opts()).plan(snap, None, 10, Objective())
        b1 = B1Policy(opts(), bw_weight=0.0, rtt_weight=0.0).plan(snap, None, 10, Objective())
        assert b0.stages == b1.stages


def test_b0_four_tier_affinity():
    snap = snapshot([
        worker("evict", resident="c"),
        worker("idle"),
        worker("cached", disk={"a"}),
        worker("resident", resident="a"),
    ])
    ranked = sorted(snap.workers, key=lambda w: (affinity_tier(snap, w, "a"), w))
    assert ranked == ["resident", "cached", "idle", "evict"]
    decision = B0Policy(opts()).plan(snap, None, 5, Objective())
    assert decision.stages[0] == ("a", "resident")


def test_b1_prefers_faster_worker_within_tier():
    snap = snapshot([worker("slow", bw=1 * MB), worker("fast", bw=50 * MB), worker("x", bw=5 * MB)])
    decision = B1Policy(opts()).plan(snap, None, 5, Objective())
    assert decision.stages[0][1] == "fast"


def test_b4_replicates_full_model_where_it_fits():
    catalog = three_stage_catalog(with_full=True)
    snap = snapshot(
        [worker("r1", ram=1000 * MB), worker("r2", ram=1000 * MB), worker("p1", ram=120 * MB),
         worker("p2", ram=120 * MB), worker("p3", ram=120 * MB)],
        catalog,
    )
    decision = B4Policy(opts()).plan(snap, None, 20, Objective())
    replica_workers = {w for _, w in decision.replicas}
    assert replica_workers == {"r1", "r2"}
    assert {w for _, w in decision.stages} == {"p1", "p2", "p3"}


def test_recut_considers_finer_compositions():
    snap = snapshot([worker(f"w{i}") for i in range(4)], three_stage_catalog(layer_shards=True))
    placement = SAwarePolicy(opts(), recut=False).plan(snap, None, 50, Objective())
    recut = SAwarePolicy(opts(), recut=True).plan(snap, None, 50, Objective())
    assert recut.search.compositions > placement.search.compositions
    assert recut.cost.T_total.mid <= placement.cost.T_total.mid + 1e-9


def test_deadline_objective_minimises_period_among_feasible():
    rng = random.Random(3)
    snap = random_snapshot(rng, n_workers=4)
    loose = exhaustive_search(snap, 10, opts(), Objective("min_steady_deadline", 1e9))
    assert loose.found
    best_period = min(c.cost.T_steady.mid for c in loose.ranked)
    assert loose.best.cost.T_steady.mid == pytest.approx(best_period)


def test_build_plan_sets_next_hops_and_backups():
    snap = snapshot([worker(f"w{i}") for i in range(5)], capability={("w0", "w1"): True})
    stages = (("a", "w0"), ("b", "w1"), ("c", "w2"))
    plan = build_plan(snap, stages, epoch=3, reason="t", opts=opts(direct_path=True))
    assert [s.next_hop.worker_id for s in plan.stages] == ["w1", "w2", None]
    assert [r.value for r in plan.stages[0].next_hop.allowed_routes] == ["via_foreman", "direct"]
    assert all(set(s.backups) <= {"w3", "w4"} for s in plan.stages)
