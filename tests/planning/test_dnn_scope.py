"""DNN work runs on mobile workers; PC workers only when validating partitions (scope "pc")."""

import asyncio

import pytest

from foreman.planning.config import RunConfig
from foreman.planning.contract import WorkerState
from foreman.planning.planner import Objective, exhaustive_search
from foreman.planning.policies import B0Policy, SAwarePolicy
from foreman.planning.scope import infer_kind, kind_allowed
from foreman.planning.telemetry import TelemetryStore
from tests.planning.fake_workers import FakeClock, FakeNetwork, FakeWorker, make_coordinator, results, submit
from tests.planning.fixtures import opts, three_stage_catalog


def _store():
    store = TelemetryStore(clock=lambda: 0.0)
    for worker_id, kind in (("m1", "mobile"), ("m2", "mobile"), ("m3", "mobile"), ("pc1", "pc"), ("pc2", "pc"), ("pc3", "pc")):
        store.ingest_state(WorkerState(worker_id=worker_id, device_class="phone" if kind == "mobile" else "pc"))
        store.set_kind(worker_id, kind)
    return store


@pytest.mark.parametrize("scope,allowed", [("mobile", {"m1", "m2", "m3"}), ("pc", {"pc1", "pc2", "pc3"}),
                                           ("any", {"m1", "m2", "m3", "pc1", "pc2", "pc3"})])
def test_planners_only_use_workers_in_scope(scope, allowed):
    snap = _store().snapshot(three_stage_catalog(), {}, dnn_scope=scope)
    assert set(snap.alive_workers()) == allowed
    o = opts(min_samples=0)
    for decision in (B0Policy(o).plan(snap, None, 10, Objective()), SAwarePolicy(o).plan(snap, None, 10, Objective())):
        assert {w for _, w in decision.stages} <= allowed
    for candidate in exhaustive_search(snap, 10, o).ranked:
        assert {w for _, w in candidate.stages} <= allowed


def test_kind_rules():
    assert infer_kind("pc") == "pc" and infer_kind("android") == "mobile" and infer_kind(None) == "mobile"
    assert kind_allowed("mobile", "mobile") and not kind_allowed("pc", "mobile")
    assert not kind_allowed("unknown", "any")
    with pytest.raises(ValueError):
        RunConfig(dnn_workers="desktop")


@pytest.mark.parametrize("scope", ["mobile", "pc"])
def test_coordinator_places_shards_only_on_workers_in_scope(scope):
    async def scenario():
        net, clock = FakeNetwork(), FakeClock()
        make_coordinator(net, clock, dnn_workers=scope)
        for prefix, kind in (("m", "mobile"), ("pc", "pc")):
            for index, shard in enumerate(("a", "b", "c"), 1):
                FakeWorker(f"{prefix}{index}", net, device_class="phone", resident=shard)
                net.coordinator.worker_connected(f"{prefix}{index}", "phone", kind)
        for _ in range(2):
            clock.now += 1
            for worker in net.workers.values():
                worker.report()
            await net.settle()
        client = await submit(net, 4)
        for _ in range(20):
            await net.settle()
            if results(client):
                break
            clock.now += 1
            await net.coordinator.tick()
        assert len(results(client)) == 4
        used = {w for _, w in net.coordinator.plan.assignment()}
        expected_prefix = "m" if scope == "mobile" else "pc"
        assert all(w.startswith(expected_prefix) and not (scope == "mobile" and w.startswith("pc")) for w in used)
        assert all(net.workers[w].computed == 0 for w in net.workers if w not in used)

    asyncio.run(scenario())
