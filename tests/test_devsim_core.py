import asyncio
import random

import pytest

from evaluation.devsim.bandwidth import (
    MIN_FACTOR, Compose, Markov, Outage, RandomOutages, RandomWalk, Stable, Step, build_trace)
from evaluation.devsim.clock import VirtualClock
from evaluation.devsim.fleet import MBIT, make_fleet
from evaluation.devsim.link import LinkEngine
from evaluation.devsim.scenarios import SCENARIOS, get_scenario, prepare


def trace(model, seed=1, horizon=200.0):
    return build_trace(model, random.Random(seed), horizon)


# ------------------------------------------------------------------ bandwidth models
def test_models_are_seeded_and_deterministic():
    for model in (RandomWalk(), Markov(), RandomOutages()):
        assert trace(model, 5).values == trace(model, 5).values
        assert trace(model, 5).values != trace(model, 6).values


def test_stable_step_and_outage():
    assert set(trace(Stable()).values) == {1.0}
    step = trace(Step(at_s=30, factor=0.1, until_s=60))
    assert step.factor(29) == 1.0 and step.factor(30) == 0.1 and step.factor(59) == 0.1 and step.factor(60) == 1.0
    outage = trace(Outage(((10.0, 15.0),)))
    assert outage.factor(9) == 1.0 and outage.factor(12) == MIN_FACTOR and outage.factor(15) == 1.0
    assert step.factor(10_000) == 1.0  # holds the last value after the end


def test_random_walk_stays_in_bounds_and_moves():
    values = trace(RandomWalk(sigma=0.3, lo=0.2, hi=1.4), horizon=500).values
    assert min(values) >= 0.2 and max(values) <= 1.4 and max(values) - min(values) > 0.3


def test_markov_visits_both_states_roughly_in_proportion():
    values = trace(Markov(good=1.0, bad=0.2, p_bad=0.1, p_good=0.1), horizon=4000).values
    bad = sum(1 for v in values if v < 0.5) / len(values)
    assert 0.35 < bad < 0.65  # symmetric chain: about half the time congested


def test_compose_multiplies():
    both = trace(Compose((Step(at_s=0, factor=0.5), Step(at_s=0, factor=0.5))))
    assert both.factor(5) == pytest.approx(0.25)
    assert min(trace(Compose((Stable(), Outage(((0.0, 5.0),))))).values) == MIN_FACTOR


def test_trace_mean():
    t = trace(Step(at_s=50, factor=0.0, until_s=100), horizon=200)
    assert t.mean(0, 50) == 1.0 and t.mean(50, 100) == pytest.approx(MIN_FACTOR) and 0.4 < t.mean(0, 100) < 0.6


# ------------------------------------------------------------------ fleet
def test_a_fleet_of_15_has_the_expected_mix():
    fleet = make_fleet(15, seed=3)
    assert len(fleet) == 15 and len({d.device_id for d in fleet}) == 15
    assert [sum(1 for d in fleet if d.tier == t) for t in ("flagship", "mid", "low")] == [3, 6, 6]
    assert all(d.kind == "phone" for d in fleet)
    assert make_fleet(15, seed=3) == fleet and make_fleet(15, seed=4) != fleet
    flagship = [d.down_mbps for d in fleet if d.tier == "flagship"]
    low = [d.down_mbps for d in fleet if d.tier == "low"]
    assert min(flagship) > max(low)  # tiers are clearly different


def test_fleet_with_pcs_and_validation():
    fleet = make_fleet(15, seed=1, pc=2)
    assert sum(1 for d in fleet if d.kind == "pc") == 2 and len(fleet) == 15
    assert fleet[0].down_bps == fleet[0].down_mbps * MBIT
    for bad in ({"n": 0}, {"n": 3, "pc": 4}):
        with pytest.raises(ValueError):
            make_fleet(**bad)


# ------------------------------------------------------------------ scenarios
def test_every_scenario_builds_for_15_devices():
    fleet = make_fleet(15, seed=2)
    for name in SCENARIOS:
        prepared = prepare(get_scenario(name), fleet, seed=1, horizon_s=300)
        assert set(prepared.traces) == {d.device_id for d in fleet}
        assert all(v > 0 for t in prepared.traces.values() for v in t.values)
    with pytest.raises(ValueError, match="Available"):
        get_scenario("nope")


def test_scenarios_are_repeatable_and_seed_dependent():
    fleet = make_fleet(15, seed=2)
    a, b, c = (prepare(get_scenario("congestion"), fleet, seed=s, horizon_s=200) for s in (1, 1, 2))
    assert {k: v.values for k, v in a.traces.items()} == {k: v.values for k, v in b.traces.items()}
    assert {k: v.values for k, v in a.traces.items()} != {k: v.values for k, v in c.traces.items()}


def test_drop_hits_only_the_fastest_third():
    fleet = make_fleet(15, seed=2)
    prepared = prepare(get_scenario("drop"), fleet, seed=1, horizon_s=200)
    hit = {d for d, t in prepared.traces.items() if t.factor(50) < 0.5}
    assert hit == {d.device_id for d in sorted(fleet, key=lambda d: -d.down_mbps)[:5]}
    assert all(t.factor(10) == 1.0 and t.factor(120) == 1.0 for t in prepared.traces.values())


def test_churn_events_and_shared_medium():
    fleet = make_fleet(15, seed=2)
    churn = prepare(get_scenario("churn"), fleet, seed=1)
    assert [e.action for e in churn.events].count("leave") == 3 and [e.action for e in churn.events].count("join") == 2
    assert churn.events == sorted(churn.events, key=lambda e: e.at_s)
    assert prepare(get_scenario("shared-wifi"), fleet).shared_capacity_mbps == 100.0
    assert prepare(get_scenario("drift"), fleet).shared_capacity_mbps is None


# ------------------------------------------------------------------ link engine
SPEED = 20.0  # 20 virtual seconds per real second keeps these tests quick


def engine(fleet_size=3, model=None, shared=None, speed=SPEED):
    fleet = make_fleet(fleet_size, seed=1, spread=0.0)
    # flat nominal links make the arithmetic exact: every device 1 MB/s down, 0.5 MB/s up
    flat = [d.__class__(**{**d.__dict__, "down_mbps": 8.0, "up_mbps": 4.0}) for d in fleet]
    clock = VirtualClock(speed)
    traces = {d.device_id: build_trace(model or Stable(), random.Random(0), 600) for d in flat}
    return clock, flat, LinkEngine(clock, {d.device_id: d for d in flat}, traces, shared_capacity_mbps=shared)


def run(coro):
    return asyncio.run(coro)


def test_a_transfer_takes_bytes_over_rate():
    async def go():
        clock, fleet, eng = engine()
        took = await eng.transfer(fleet[0].device_id, "down", 4_000_000)  # 1 MB/s -> 4 s
        up = await eng.transfer(fleet[0].device_id, "up", 2_000_000)  # 0.5 MB/s -> 4 s
        await eng.close()
        return took, up

    took, up = run(go())
    # real-time sleeps: allow for a busy machine
    assert took == pytest.approx(4.0, rel=0.35) and up == pytest.approx(4.0, rel=0.35)


def test_tiny_messages_cost_nothing_and_bad_direction_is_rejected():
    async def go():
        clock, fleet, eng = engine()
        assert await eng.transfer(fleet[0].device_id, "up", 800) == 0.0
        with pytest.raises(ValueError):
            await eng.transfer(fleet[0].device_id, "sideways", 10_000)
        await eng.close()

    run(go())


def test_a_bandwidth_drop_mid_transfer_slows_it_down():
    async def go():
        # 1 MB/s until t=2 s, then 10% of that: 4 MB = 2 MB in the first 2 s, 2 MB at 0.1 MB/s = 20 s more
        clock, fleet, eng = engine(model=Step(at_s=2.0, factor=0.1))
        took = await eng.transfer(fleet[0].device_id, "down", 4_000_000)
        await eng.close()
        return took

    assert run(go()) == pytest.approx(22.0, rel=0.12)


def test_an_outage_stalls_a_transfer_until_the_link_recovers():
    async def go():
        clock, fleet, eng = engine(model=Outage(((1.0, 6.0),)))
        took = await eng.transfer(fleet[0].device_id, "down", 2_000_000)  # 1 s of data, stall 1..6, 1 s more
        await eng.close()
        return took

    assert run(go()) == pytest.approx(7.0, rel=0.12)


def test_a_devices_own_flows_share_its_link():
    async def go():
        clock, fleet, eng = engine()
        d = fleet[0].device_id
        a, b = await asyncio.gather(eng.transfer(d, "down", 2_000_000), eng.transfer(d, "down", 2_000_000))
        await eng.close()
        return a, b

    a, b = run(go())
    assert a == pytest.approx(4.0, rel=0.15) and b == pytest.approx(4.0, rel=0.15)  # 2 MB each at 0.5 MB/s


def test_different_devices_do_not_slow_each_other_without_a_shared_medium():
    async def go():
        clock, fleet, eng = engine()
        times = await asyncio.gather(*(eng.transfer(d.device_id, "down", 2_000_000) for d in fleet))
        await eng.close()
        return times

    assert all(t == pytest.approx(2.0, rel=0.2) for t in run(go()))


def test_a_shared_medium_is_split_fairly():
    async def go():
        # three devices could each do 8 Mbit/s but share 12 Mbit/s: 4 Mbit/s = 0.5 MB/s each
        clock, fleet, eng = engine(shared=12.0)
        times = await asyncio.gather(*(eng.transfer(d.device_id, "down", 2_000_000) for d in fleet))
        await eng.close()
        return times

    assert all(t == pytest.approx(4.0, rel=0.15) for t in run(go()))


def test_abort_fails_a_devices_transfers_only():
    async def go():
        clock, fleet, eng = engine()
        slow = asyncio.create_task(eng.transfer(fleet[0].device_id, "down", 40_000_000))
        other = asyncio.create_task(eng.transfer(fleet[1].device_id, "down", 1_000_000))
        await asyncio.sleep(0.05)
        assert eng.abort(fleet[0].device_id) == 1
        with pytest.raises(ConnectionError):
            await slow
        took = await other
        await eng.close()
        return took

    assert run(go()) == pytest.approx(1.0, rel=0.3)


def test_virtual_clock_speed():
    async def go():
        clock = VirtualClock(speed=50)
        await clock.sleep(5.0)  # 0.1 s real
        return clock.now()

    assert 4.5 < run(go()) < 7.5
    with pytest.raises(ValueError):
        VirtualClock(0)
