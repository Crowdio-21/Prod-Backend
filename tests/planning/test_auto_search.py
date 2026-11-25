"""Planning must stay fast as the fleet grows: the search is chosen by instance size (BERT-base, 24 shards)."""

import time

from foreman.planning.config import RunConfig
from foreman.planning.cost_model import memory_feasible
from foreman.planning.planner import (
    AUTO_EXHAUSTIVE_LIMIT,
    Objective,
    exhaustive_search,
    exhaustive_work,
    pareto_dp_search,
    search,
)
from tests.planning.fixtures import opts
from tests.planning.test_ram_cuts import CATALOG, GiB, phones, snap_of


def test_auto_is_the_default_and_unknown_modes_are_rejected():
    assert RunConfig().planner_mode == "auto"
    for mode in ("auto", "exhaustive", "pareto_dp", "scalar_dp"):
        RunConfig.from_mapping({"planner_mode": mode})
    try:
        RunConfig.from_mapping({"planner_mode": "fast"})
    except ValueError:
        return
    raise AssertionError("an unknown planner mode must be rejected")


def test_the_work_estimate_grows_steeply_with_the_fleet():
    sizes = [exhaustive_work(snap_of(phones(*([2 * GiB] * n)))) for n in (3, 5, 7, 10)]
    assert sizes == sorted(sizes) and sizes[1] <= AUTO_EXHAUSTIVE_LIMIT < sizes[2]
    assert sizes[3] > 100 * sizes[1]


def test_a_small_fleet_still_gets_the_exact_answer():
    snap = snap_of(phones(*([2 * GiB] * 4)))
    auto = search(snap, 100, opts(), Objective(), mode="auto")
    exact = exhaustive_search(snap, 100, opts(), Objective())
    assert auto.method == "exhaustive"
    assert auto.best.cost.T_total.mid == exact.best.cost.T_total.mid


def test_a_medium_fleet_uses_the_exact_dp_and_it_matches_exhaustive():
    seven = snap_of(phones(*([2 * GiB] * 7)))
    assert exhaustive_work(seven) > AUTO_EXHAUSTIVE_LIMIT
    assert search(seven, 100, opts(), Objective(), mode="auto").method == "pareto_dp"
    six = snap_of(phones(*([2 * GiB] * 6)))  # small enough to check against the exhaustive answer
    dp = pareto_dp_search(six, 100, opts(), Objective())
    exact = exhaustive_search(six, 100, opts(), Objective())
    assert abs(dp.best.cost.T_total.mid - exact.best.cost.T_total.mid) < 1e-9


def test_a_large_fleet_is_planned_quickly_and_the_plan_is_valid_and_close_to_optimal():
    snap = snap_of(phones(*([2 * GiB] * 12)))
    started = time.perf_counter()
    auto = search(snap, 100, opts(), Objective(), mode="auto")
    elapsed = time.perf_counter() - started
    assert elapsed < 8.0, f"planning 12 phones took {elapsed:.1f} s"
    assert auto.found
    workers = [w for _, w in auto.best.stages]
    assert len(set(workers)) == len(workers)  # one shard per phone
    assert all(memory_feasible(snap, w, s, opts()) for s, w in auto.best.stages)
    # Against the exact answer at 8 phones, the same code path (scalar DP + local search) is within a few percent.
    eight = snap_of(phones(*([2 * GiB] * 8)))
    approx = search(eight, 100, opts(), Objective(), mode="auto")
    exact = pareto_dp_search(eight, 100, opts(), Objective())
    assert approx.best.cost.T_total.mid <= exact.best.cost.T_total.mid * 1.05


def test_the_planner_respects_ram_at_scale():
    rams = [3 * GiB, 150 * 1024 * 1024, 150 * 1024 * 1024] + [300 * 1024 * 1024] * 7
    snap = snap_of(phones(*rams))
    result = search(snap, 100, opts(), Objective(), mode="auto")
    assert result.found
    assert all(memory_feasible(snap, w, s, opts()) for s, w in result.best.stages)
    assert max(CATALOG.shards[s].bytes for s, w in result.best.stages if w == "p1") <= 75 * 1024 * 1024
