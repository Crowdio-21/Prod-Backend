"""Incremental repartitioning (local search) and the regret-counter switch rule."""

import random
from dataclasses import replace

import pytest

from foreman.planning.config import POLICIES, ControllerConfig, RunConfig
from foreman.planning.controller import ControllerState, decide
from foreman.planning.planner import (
    Objective,
    _Neighbourhood,
    exhaustive_search,
    local_gap,
    local_search,
)
from foreman.planning.policies import B0Policy, SAwarePolicy, build_plan, make_policy
from tests.planning.fixtures import MB, opts, profile, random_snapshot, snapshot, three_stage_catalog, worker
from tests.planning.test_planner import _check_feasible

SEEDS = range(40)


def _start(snap, o):
    """A feasible starting plan: the residency-preserving affinity plan."""
    decision = B0Policy(o).plan(snap, None, 10, Objective())
    return decision.stages if decision.feasible else None


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_local_search_is_feasible_and_never_worse_than_start(seed):
    rng = random.Random(seed)
    snap = random_snapshot(rng, n_workers=rng.randint(3, 5))
    o = opts()
    start = _start(snap, o)
    if start is None:
        pytest.skip("no feasible start")
    n = rng.randint(1, 50)
    result = local_search(snap, n, o, start=start)
    assert result.found and result.method == "local"
    _check_feasible(snap, result.best.stages, o)
    start_cost = exhaustive_search(snap, n, o, compositions=[tuple(s for s, _ in start)]).ranked
    priced = {c.stages: c.cost.T_total.mid for c in start_cost}
    if start in priced:
        assert result.best.cost.T_total.mid <= priced[start] + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_local_search_ends_at_a_local_optimum_and_bounds_the_oracle(seed):
    rng = random.Random(seed)
    snap = random_snapshot(rng, n_workers=4)
    o = opts()
    start = _start(snap, o)
    if start is None:
        pytest.skip("no feasible start")
    n = rng.randint(1, 30)
    result = local_search(snap, n, o, start=start)
    best = result.best.cost.T_total.mid
    hood = _Neighbourhood(snap, o, keep_resident=False)
    from foreman.planning.cost_model import pipeline_cost

    for _, stages in hood.moves(result.best.stages):
        if hood.feasible(stages):
            assert pipeline_cost(snap, stages, n, o).T_total.mid >= best - 1e-9
    exact, approx, gap = local_gap(snap, n, o, start)
    assert approx >= exact - 1e-9 and gap >= -1e-9


def test_local_search_gap_is_small_on_average():
    gaps = []
    for seed in SEEDS:
        rng = random.Random(seed)
        snap = random_snapshot(rng, n_workers=4)
        o = opts()
        start = _start(snap, o)
        if start is None:
            continue
        _, _, gap = local_gap(snap, rng.randint(1, 30), o, start)
        gaps.append(gap)
    assert gaps
    # A regression guard, not a guarantee: local optima exist.
    assert sum(gaps) / len(gaps) < 0.02


def test_local_search_shifts_a_cut_when_one_stage_is_slow():
    """Atom 2 is expensive on the 'b' worker only; moving it into 'c' requires a cut shift."""
    from tests.planning.fixtures import shard
    from foreman.planning.types import ShardCatalog

    shards = dict(three_stage_catalog().shards)
    shards["b1"] = shard("b1", 1, 1, size=50 * MB)
    shards["c2"] = shard("c2", 2, 3, size=60 * MB)
    catalog = ShardCatalog("m", 4, shards, ("a", "b", "c"), None)
    slow = profile(compute={"a": 0.1, "b": 4.0, "b1": 0.2, "c": 0.1, "c2": 0.3})
    workers = [worker("p1", resident="a"), worker("p2", resident="b"), worker("p3", resident="c")]
    snap = snapshot(workers, catalog, {"phone": slow})
    o = opts()
    result = local_search(snap, 200, o, start=(("a", "p1"), ("b", "p2"), ("c", "p3")))
    assert tuple(s for s, _ in result.best.stages) == ("a", "b1", "c2")
    assert [w for _, w in result.best.stages] == ["p1", "p2", "p3"]


def test_local_search_repairs_a_dead_worker_with_a_standby():
    workers = [worker("p1", resident="a"), worker("p2", resident="b", alive=False), worker("p3", resident="c"),
               worker("spare", disk={"b"})]
    snap = snapshot(workers)
    o = opts()
    result = local_search(snap, 10, o, start=(("a", "p1"), ("b", "p2"), ("c", "p3")))
    assert result.found
    assert "p2" not in [w for _, w in result.best.stages]
    _check_feasible(snap, result.best.stages, o)


def test_local_search_without_any_standby_is_not_found():
    workers = [worker("p1", resident="a"), worker("p2", resident="b", alive=False), worker("p3", resident="c")]
    result = local_search(snapshot(workers), 10, opts(), start=(("a", "p1"), ("b", "p2"), ("c", "p3")))
    assert not result.found


def test_local_policy_falls_back_to_full_search_and_runs_it_every_k_calls():
    rng = random.Random(3)
    snap = random_snapshot(rng, n_workers=4)
    o = opts()
    policy = SAwarePolicy(o, search="local", full_every=3)
    first = policy.plan(snap, None, 10, Objective())
    assert first.search.method == "exhaustive"
    plan = build_plan(snap, first.stages, epoch=1, reason="t", opts=o)
    methods = [policy.plan(snap, plan, 10, Objective()).search.method for _ in range(6)]
    # Calls 3 and 6 also run the full search; on a tie the local result is kept.
    assert methods[0] == methods[1] == "local"
    best = exhaustive_search(snap, 10, o).best.cost.T_total.mid
    assert policy.plan(snap, plan, 10, Objective()).cost.T_total.mid >= best - 1e-9


# ---------------------------------------------------------------------------
# Regret-counter controller
# ---------------------------------------------------------------------------

CFG = ControllerConfig(persist_k=1, cooldown_s=0, margin_abs_s=0.5, margin_frac=0.0)


def _setup():
    """s2 is slow for b (2.0 s/request); the cold spare does b in 0.2 s."""
    p = profile(compute={"a": 0.1, "b": 2.0, "c": 0.1})
    fast = profile("fast", compute={"a": 0.1, "b": 0.2, "c": 0.1})
    workers = [worker("s1", resident="a"), worker("s2", resident="b"), worker("s3", resident="c"),
               worker("spare", device_class="fast")]
    snap = snapshot(workers, profiles={"phone": p, "fast": fast})
    o = opts()
    plan = build_plan(snap, (("a", "s1"), ("b", "s2"), ("c", "s3")), epoch=1, reason="init", opts=o)
    return snap, plan, o


def _run(policy, snap, plan, ns, triggers=("bw_drift",)):
    state = ControllerState()
    out = []
    for step, n in enumerate(ns):
        decision, state = decide(snap, plan, n, policy, Objective(), state, CFG, triggers, 100.0 + step)
        out.append((decision, state))
        if decision.switches:
            break
    return out


def test_regret_waits_for_served_requests_then_switches():
    snap, plan, o = _setup()
    policy = SAwarePolicy(o, recut=False, switch_rule="regret")
    steps = _run(policy, snap, plan, [500 - i for i in range(0, 60, 1)])
    first, _ = steps[0]
    assert first.action == "stay" and first.reason == "accumulating regret"
    assert first.break_even["regret"] == 0.0  # nothing served yet
    last, state = steps[-1]
    assert last.action == "switch" and last.reason == "regret >= C_switch + margin"
    assert last.break_even["regret"] >= last.break_even["switch_cost"] + last.break_even["margin"]
    assert state.regret_s == 0.0
    # Gain ~1.8 s/request and a cold switch of ~12 s: about 7 served requests.
    assert 5 <= len(steps) - 1 <= 10


def test_break_even_switches_at_once_where_regret_waits():
    snap, plan, o = _setup()
    be, _ = _run(SAwarePolicy(o, recut=False), snap, plan, [500])[0]
    rg, _ = _run(SAwarePolicy(o, recut=False, switch_rule="regret"), snap, plan, [500])[0]
    assert be.action == "switch"
    assert rg.action == "stay"


def test_regret_keeps_accumulating_without_a_persistent_trigger():
    snap, plan, o = _setup()
    policy = SAwarePolicy(o, recut=False, switch_rule="regret")
    state = ControllerState()
    _, state = decide(snap, plan, 100, policy, Objective(), state, CFG, ("bw_drift",), 100.0)
    _, state = decide(snap, plan, 95, policy, Objective(), state, CFG, ("bw_drift",), 101.0)
    assert state.regret_s > 0
    decision, state2 = decide(snap, plan, 94, policy, Objective(), state, CFG, (), 102.0)
    assert decision.planner_called and state2.regret_s > state.regret_s


def test_regret_resets_when_no_alternative_is_cheaper():
    snap, plan, o = _setup()
    policy = SAwarePolicy(o, recut=False, switch_rule="regret")
    state = ControllerState(regret_s=5.0, last_n=100)
    fast = replace(snap, workers={**snap.workers, "spare": replace(snap.workers["spare"], alive=False)})
    decision, state = decide(fast, plan, 99, policy, Objective(), state, CFG, ("bw_drift",), 100.0)
    assert decision.action == "stay"
    assert state.regret_s == 0.0


def test_forced_replans_bypass_the_regret_rule():
    snap, plan, o = _setup()
    dead = replace(snap, workers={**snap.workers, "s2": replace(snap.workers["s2"], alive=False)})
    decision, _ = decide(dead, plan, 2, SAwarePolicy(o, recut=False, switch_rule="regret"), Objective(),
                         ControllerState(), CFG, ("worker_lost",), 100.0)
    assert decision.action == "forced_switch"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,search,rule", [
    ("S-aware-local", "local", "break_even"),
    ("S-aware-regret", "full", "regret"),
    ("S-aware-local-regret", "local", "regret"),
])
def test_new_policies_are_registered(name, search, rule):
    assert name in POLICIES
    RunConfig(policy=name)
    policy = make_policy(name, opts())
    assert policy.name == name and policy.search == search and policy.switch_rule == rule


def test_baselines_keep_their_switch_rules():
    assert make_policy("B2", opts()).switch_rule == "always"
    assert make_policy("B3", opts()).switch_rule == "static"
    assert make_policy("S-aware", opts()).switch_rule == "break_even"
