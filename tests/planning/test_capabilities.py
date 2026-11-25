"""The Foreman only sends a worker what the worker declared it can handle."""

import asyncio

from foreman.core.connection_manager import WorkerInfo
from tests.planning.fake_workers import MB, FakeClock, FakeNetwork, FakeWorker, make_coordinator, make_manifest, submit
from tests.planning.test_integration_fake_workers import _check_results, _drive, _heartbeats, _no_violations

BASELINE = ("probe", "msg_ack", "flags", "load", "run", "route")
PLANNING_ONLY = ("adaptive_planning", "problem_reports")


def run(coro):
    return asyncio.run(coro)


def _world(flags):
    net, clock = FakeNetwork(), FakeClock()
    make_coordinator(net, clock, manifest=make_manifest(b_bytes=50 * MB), flags=dict(flags, adaptive_planning=True))
    for worker_id, shard in (("w1", "a"), ("w2", "b"), ("w3", "c")):
        FakeWorker(worker_id, net, resident=shard)
        net.connect(worker_id)
    return net, clock


# ------------------------------------------------------------ worker_ready parsing
def test_worker_ready_declares_protocol_and_capabilities():
    declared = WorkerInfo.from_worker_ready_message("p", None, {
        "platform": "android", "runtime": "jvm", "worker_type": "android_kotlin", "proto_version": 2,
        "supported_flags": list(PLANNING_ONLY), "supported_messages": list(BASELINE)})
    assert declared.proto_version == 2
    assert declared.supported_flags == PLANNING_ONLY
    assert declared.supported_messages == BASELINE
    legacy = WorkerInfo.from_worker_ready_message("pc", None, {"platform": "linux", "runtime": "cpython"})
    assert legacy.proto_version == 1 and legacy.supported_flags is None and legacy.supported_messages is None
    odd = WorkerInfo.from_worker_ready_message("x", None, {"proto_version": "two", "supported_flags": "all"})
    assert odd.proto_version == 1 and odd.supported_flags is None


# ------------------------------------------------------------ legacy workers
def test_a_legacy_worker_is_never_sent_a_v2_message_or_planned_onto():
    async def scenario():
        net, clock = _world({"problem_reports": True})
        legacy = FakeWorker("legacy", net)
        legacy.report = lambda **kwargs: None  # a legacy worker never sends a v2 report
        net.connect("legacy", "pc", "pc", proto_version=1, flags=None, messages=None)
        assert "legacy" not in net.coordinator.connected
        await _heartbeats(net, clock)
        client = await submit(net, 4)
        await _drive(net, clock, client, 4)
        _check_results(client, 4)
        assert legacy.received == []  # no probe, no load, no run
        assert all(w != "legacy" for _, w in net.coordinator.plan.assignment())
        _no_violations(net)
    run(scenario())


def test_a_legacy_worker_that_later_sends_a_v2_report_joins_the_plan():
    async def scenario():
        net, clock = _world({})
        late = FakeWorker("late", net)
        net.connect("late", "phone", proto_version=1, flags=None, messages=None)
        assert "late" not in net.coordinator.connected
        late.report()
        await net.settle()
        assert "late" in net.coordinator.connected
    run(scenario())


def test_a_worker_without_the_engine_messages_is_not_planned_onto():
    async def scenario():
        net, clock = _world({})
        light = FakeWorker("light", net)
        net.connect("light", "phone", messages=("probe", "msg_ack", "flags"))  # no load / run
        await _heartbeats(net, clock)
        client = await submit(net, 3)
        await _drive(net, clock, client, 3)
        assert light.loads == [] and all(w != "light" for _, w in net.coordinator.plan.assignment())
    run(scenario())


# ------------------------------------------------------------ per-flag gating
def test_prefetch_goes_only_to_workers_that_declare_it():
    async def scenario():
        net, clock = _world({"prefetch": True})
        capable = FakeWorker("capable", net)
        net.connect("capable")
        incapable = FakeWorker("incapable", net)
        net.connect("incapable", flags=PLANNING_ONLY, messages=BASELINE)
        await _heartbeats(net, clock)
        client = await submit(net, 2)
        await _drive(net, clock, client, 2)
        for _ in range(3):
            await net.coordinator.tick()
            await net.settle()
        assert incapable.prefetches == [] and "prefetch" not in incapable.received
        assert capable.prefetches, "the capable standby was prefetched"
    run(scenario())


def _lose_w2_with_donor(net, clock, donor_flags, donor_messages):
    async def go():
        spare = FakeWorker("spare", net)
        net.connect("spare")
        donor = FakeWorker("donor", net, disk={"b"})
        net.connect("donor", "pc", "pc", flags=donor_flags, messages=donor_messages)  # a PC: may donate, never runs shards
        donor.peer_links = {"spare": (True, 2.0, 100 * MB)}
        net.coordinator.capability.record("donor", "spare", True, 0.0)
        await _heartbeats(net, clock)
        first = await submit(net, 2)
        await _drive(net, clock, first, 2)
        net.workers["w2"].disconnect()
        await net.coordinator.worker_disconnected("w2")
        client = await submit(net, 3, start=10)
        await _drive(net, clock, client, 3)
        _check_results(client, 3, start=10)
        return spare, donor
    return go()


def test_a_donor_that_cannot_serve_is_not_used_and_the_load_falls_back_to_the_foreman():
    async def scenario():
        net, clock = _world({"peer_fetch": True})
        await _lose_w2_with_donor(net, clock, PLANNING_ONLY, BASELINE)
        for worker in net.workers.values():
            assert all(m.source is None for m in worker.load_msgs), worker.worker_id
            assert "serve_allowed" not in worker.received
    run(scenario())


def test_a_capable_donor_is_named_with_its_address():
    async def scenario():
        net, clock = _world({"peer_fetch": True})
        spare, donor = await _lose_w2_with_donor(
            net, clock, ("adaptive_planning", "peer_fetch"),
            BASELINE + ("peer_fetch", "serve_allowed"))
        sourced = [m for w in net.workers.values() for m in w.load_msgs if m.source]
        assert sourced, "the donor was cheaper than the Foreman, so it was named as the source"
        assert all(m.source == "donor" and m.source_address == "donor:7000" for m in sourced)
        assert "serve_allowed" in donor.received
        _no_violations(net)
    run(scenario())


def test_direct_routes_need_both_ends_to_declare_direct_path():
    async def scenario():
        net, clock = _world({"direct_path": True})
        net.connect("w2", flags=PLANNING_ONLY, messages=BASELINE)  # w2 does not declare direct_path
        for a, b in (("w1", "w2"), ("w2", "w3")):
            net.workers[a].peer_links[b] = (True, 2.0, 100 * MB)
            net.coordinator.capability.record(a, b, True, 0.0)
        await _heartbeats(net, clock, w1=0.2 * MB, w2=0.2 * MB, w3=0.2 * MB)
        client = await submit(net, 3)
        await _drive(net, clock, client, 3)
        _check_results(client, 3)
        for worker_id in ("w1", "w2"):
            assert all(r["allowed_routes"] == ["via_foreman"] for r in net.workers[worker_id].routes.values())
    run(scenario())


# ------------------------------------------------------------ flags are announced to workers
def test_every_v2_worker_is_told_the_active_flags_once():
    async def scenario():
        net, clock = _world({"problem_reports": True})
        await _heartbeats(net, clock, rounds=3)
        await net.coordinator.tick()
        await net.settle()
        for worker_id in ("w1", "w2", "w3"):
            assert net.workers[worker_id].received.count("flags") == 1, worker_id
        sent = [e for e in net.coordinator.log.events if e["event"] == "flags_sent"]
        assert len(sent) == 3 and all(e["flags"]["problem_reports"] for e in sent)
        # a worker that reconnects is told again
        await net.coordinator.worker_disconnected("w1")
        net.connect("w1")
        await _heartbeats(net, clock, rounds=2)
        await net.coordinator.tick()
        await net.settle()
        assert net.workers["w1"].received.count("flags") == 2
    run(scenario())
