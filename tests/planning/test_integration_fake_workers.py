"""Foreman integration with fake workers (in-memory network, v2 protocol)."""

import asyncio

import pytest

from tests.planning.fake_workers import (
    MB,
    FakeClock,
    FakeNetwork,
    FakeWorker,
    make_coordinator,
    results,
    submit,
    make_manifest,
)

N_ATOMS = 4


def run(coro):
    return asyncio.run(coro)


async def _heartbeats(net, clock, rounds=2, **per_worker_bw):
    for _ in range(rounds):
        clock.now += 1.0
        for worker in net.workers.values():
            if worker.connected:
                worker.report(bw=per_worker_bw.get(worker.worker_id))
        await net.settle()


async def _drive(net, clock, client, expected, max_ticks=50):
    for _ in range(max_ticks):
        await net.settle()
        if len(results(client)) >= expected:
            return
        clock.now += 1.0
        await net.coordinator.tick()
    await net.settle()


def _check_results(client, n, start=0):
    got = sorted(results(client), key=lambda r: int(r["request_id"].rsplit("-", 1)[1]))
    assert len(got) == n, f"expected {n} results, got {len(got)}"
    assert len({r["request_id"] for r in got}) == n, "each request completes exactly once"
    for index, result in enumerate(got):
        assert result["tensor"]["value"] == start + index + N_ATOMS


def _network(spare_disk=(), b_bytes=50 * MB, **coordinator_kwargs):
    net, clock = FakeNetwork(), FakeClock()
    make_coordinator(net, clock, manifest=make_manifest(b_bytes=b_bytes), **coordinator_kwargs)
    FakeWorker("w1", net, resident="a")
    FakeWorker("w2", net, resident="b")
    FakeWorker("w3", net, resident="c")
    FakeWorker("spare", net, disk=set(spare_disk))
    for worker_id in net.workers:
        net.coordinator.worker_connected(worker_id, "phone")
    return net, clock


def _no_violations(net):
    for worker in net.workers.values():
        assert worker.violations == [], (worker.worker_id, worker.violations)


# ------------------------------------------------------------------ baseline
def test_initial_plan_uses_resident_workers_and_completes():
    async def scenario():
        net, clock = _network()
        await _heartbeats(net, clock)
        client = await submit(net, 6)
        await _drive(net, clock, client, 6)
        _check_results(client, 6)
        plan = net.coordinator.plan
        assert plan.assignment() == (("a", "w1"), ("b", "w2"), ("c", "w3"))
        assert all(not w.loads for w in net.workers.values())  # nothing had to move
        _no_violations(net)
    run(scenario())


# --------------------------------------------- bandwidth drop: on disk vs absent
@pytest.mark.parametrize("warm", [True, False])
def test_bandwidth_drop_switches_only_when_replacement_is_cheap(warm):
    async def scenario():
        # b is 500 MB: a cold fetch (~50 s) is not worth it for 6 requests; a warm load (~1.5 s) is.
        net, clock = _network(spare_disk={"b"} if warm else set(), b_bytes=500 * MB)
        await _heartbeats(net, clock)
        first = await submit(net, 2)
        await _drive(net, clock, first, 2)
        epoch_before = net.coordinator.epoch
        await _heartbeats(net, clock, rounds=30, w2=0.1 * MB)  # persistent drop; the EWMA variance must settle first
        client = await submit(net, 6, start=100)
        await _drive(net, clock, client, 6)
        _check_results(client, 6, start=100)
        plan = net.coordinator.plan
        if warm:
            assert ("b", "spare") in plan.assignment()
            assert net.coordinator.epoch == epoch_before + 1
            assert net.workers["spare"].loads == [("b", net.coordinator.epoch)]
        else:
            assert ("b", "w2") in plan.assignment()
            assert net.coordinator.epoch == epoch_before
            assert not net.workers["spare"].loads
        _no_violations(net)
        # Replay: the live coordinator's logged planner inputs reproduce its decisions.
        from evaluation.reconfig.replay import replay

        outcome = replay(net.coordinator.log.events)
        assert outcome["checked"] >= 2 and outcome["identical"], outcome["mismatches"][:2]
    run(scenario())


# ------------------------------------------------ worker leaves: warm vs cold
def test_worker_loss_resumes_in_flight_and_prefers_warm_replacement():
    async def scenario():
        net, clock = _network()
        FakeWorker("cold", net)
        FakeWorker("warm", net, disk={"b"})
        for worker_id in ("cold", "warm"):
            net.coordinator.worker_connected(worker_id, "phone")
        net.workers["spare"].disconnect()
        net.coordinator.connected.discard("spare")
        net.coordinator.telemetry.mark_departed("spare")
        await _heartbeats(net, clock)
        net.workers["w2"].stall = True  # requests pile up at stage b
        client = await submit(net, 4)
        await net.settle()
        assert net.workers["w2"].stalled
        net.workers["w2"].disconnect()
        await net.coordinator.worker_disconnected("w2")
        await _drive(net, clock, client, 4)
        _check_results(client, 4)
        assert ("b", "warm") in net.coordinator.plan.assignment()
        resumed = [e for e in net.coordinator.log.events if e.get("kind") == "resume"]
        assert resumed, "in-flight requests resumed from the retained boundary activation"
        _no_violations(net)
    run(scenario())


@pytest.mark.parametrize("warm", [True, False])
def test_worker_loss_cold_replacement_pays_fetch(warm):
    async def scenario():
        net, clock = _network(spare_disk={"b"} if warm else set())
        await _heartbeats(net, clock)
        first = await submit(net, 2)
        await _drive(net, clock, first, 2)
        net.workers["w2"].disconnect()
        await net.coordinator.worker_disconnected("w2")
        client = await submit(net, 3, start=10)
        await _drive(net, clock, client, 3)
        _check_results(client, 3, start=10)
        switch = [e for e in net.coordinator.log.events if e["event"] == "switch_start"][-1]
        assert switch["mode"] == "resume"
        load = [e for e in net.coordinator.log.events if e.get("kind") == "load_result"][-1]
        assert (load["timings"]["fetch_s"] > 0) == (not warm)
        _no_violations(net)
    run(scenario())


# ------------------------------------------------------ load failure -> replan
def test_load_failure_excludes_worker_and_replans():
    async def scenario():
        net, clock = _network()
        FakeWorker("other", net, disk={"b"})
        net.coordinator.worker_connected("other", "phone")
        net.workers["spare"].fail_load = True
        net.workers["spare"].disk.add("b")
        await _heartbeats(net, clock)
        net.workers["w2"].disconnect()
        await net.coordinator.worker_disconnected("w2")
        client = await submit(net, 3)
        await _drive(net, clock, client, 3)
        _check_results(client, 3)
        assert ("spare", "b") in net.coordinator.excluded or ("b", "other") in net.coordinator.plan.assignment()
        _no_violations(net)
    run(scenario())


# ------------------------------------------- direct path blocked mid-run
def _direct_network():
    net, clock = _network(flags={"adaptive_planning": True, "direct_path": True, "problem_reports": True})
    # Relay through the Foreman is slow; peers are fast and reachable.
    for a, b in (("w1", "w2"), ("w2", "w3")):
        net.workers[a].peer_links[b] = (True, 2.0, 100 * MB)
        net.coordinator.capability.record(a, b, True, 0.0)
    return net, clock


@pytest.mark.parametrize("failure", ["refused", "silent"])
def test_peer_path_blocked_falls_back_to_relay_cleanly(failure):
    async def scenario():
        net, clock = _direct_network()
        await _heartbeats(net, clock, w1=0.2 * MB, w2=0.2 * MB, w3=0.2 * MB, spare=0.2 * MB)
        first = await submit(net, 3)
        await _drive(net, clock, first, 3)
        _check_results(first, 3)
        assert net.workers["w1"].routes[0]["allowed_routes"][0] == "direct"
        # The peer path breaks mid-run: either refused (worker falls back at once) or a
        # silent loss (ack timeout -> NEXT_HOP_DOWN -> Foreman reroutes -> worker resends).
        (net.blocked if failure == "refused" else net.blackhole).add(("w1", "w2"))
        client = await submit(net, 4, start=50)
        await asyncio.sleep(0.1)  # ack timeouts fire
        await _drive(net, clock, client, 4)
        _check_results(client, 4, start=50)
        if failure == "silent":
            assert "NEXT_HOP_DOWN" in net.workers["w1"].problems
            assert net.workers["w1"].routes[0]["allowed_routes"] == ["via_foreman"]
            flips = [e for e in net.coordinator.log.events if e["event"] == "route_flip"]
            assert flips and flips[-1]["mode"] == "via_foreman"
        assert not [e for e in net.coordinator.log.events if e["event"] == "failure"]
        _no_violations(net)
    run(scenario())


def test_duplicate_after_direct_to_relay_fallback_is_deduplicated():
    async def scenario():
        net, clock = _direct_network()
        await _heartbeats(net, clock, w1=0.2 * MB, w2=0.2 * MB, w3=0.2 * MB, spare=0.2 * MB)
        warm = await submit(net, 1)
        await _drive(net, clock, warm, 1)
        net.drop_acks.add(("w2", "w1"))  # activation arrives, the ack is lost
        client = await submit(net, 3, start=20)
        await asyncio.sleep(0.1)
        await _drive(net, clock, client, 3)
        _check_results(client, 3, start=20)
        assert "NEXT_HOP_DOWN" in net.workers["w1"].problems
        dropped = net.workers["w2"].duplicates + net.coordinator.counters.get("late_activation_dropped", 0)             + net.coordinator.counters.get("duplicates_dropped", 0)
        assert dropped >= 1, "the resent activation was de-duplicated (at w2 or at the Foreman)"
        _no_violations(net)
    run(scenario())


# ------------------------------------------------------------ prefetch
def test_prefetch_warms_backups_and_is_logged():
    async def scenario():
        net, clock = _network(flags={"adaptive_planning": True, "prefetch": True})
        await _heartbeats(net, clock)
        client = await submit(net, 2)
        await _drive(net, clock, client, 2)
        await net.coordinator.tick()
        await net.settle()
        assert net.workers["spare"].prefetches
        assert net.workers["spare"].resident is None  # prefetch never loads
        issued = [e for e in net.coordinator.log.events if e["event"] == "prefetch" and e["status"] == "issued"]
        assert issued
    run(scenario())


# ------------------------------------------------------- flags off: no coordinator
def test_all_flags_off_builds_no_coordinator(monkeypatch):
    monkeypatch.delenv("CROWDIO_FLAGS", raising=False)
    from foreman.core.plan_runtime import build_plan_coordinator

    assert build_plan_coordinator(connection_manager=None) is None
