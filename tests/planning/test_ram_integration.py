"""The coordinator cuts the real BERT-base model to the RAM phones report, and re-plans when a phone runs low."""

import asyncio
from pathlib import Path

from foreman.planning.registry import load_manifest
from tests.planning.fake_workers import MB, FakeClock, FakeNetwork, FakeWorker, make_coordinator, submit
from tests.planning.test_integration_fake_workers import _drive, _heartbeats, _no_violations

MiB = 1 << 20
GiB = 1 << 30
MANIFEST = load_manifest(Path(__file__).parent / "fixtures" / "bert_base_sst2")[0]
N_ATOMS = len(MANIFEST.atoms)  # every correct cut adds up to this many atoms of work


def run(coro):
    return asyncio.run(coro)


def _world(rams, **config):
    net, clock = FakeNetwork(), FakeClock()
    make_coordinator(net, clock, policy="S-aware", manifest=MANIFEST, **config)
    for index, ram in enumerate(rams):
        FakeWorker(f"p{index}", net, ram_free=ram)
        net.connect(f"p{index}")
    return net, clock


def _results(client):
    out = []
    for message in client:
        if message["type"] == "inference_result":
            out.extend(message["data"]["results"])
    return sorted(out, key=lambda r: int(r["request_id"].rsplit("-", 1)[1]))


def _check(client, n, start=0):
    got = _results(client)
    assert len(got) == n and len({r["request_id"] for r in got}) == n
    for index, result in enumerate(got):
        assert result["tensor"]["value"] == start + index + N_ATOMS  # whatever the cut, all 8 atoms ran once


def test_a_job_runs_with_every_stage_fitting_the_ram_of_the_phone_that_holds_it():
    async def scenario():
        rams = [400 * MiB, 300 * MiB, 300 * MiB, 300 * MiB, 300 * MiB, 300 * MiB]
        net, clock = _world(rams)
        await _heartbeats(net, clock)
        client = await submit(net, 6)
        await _drive(net, clock, client, 6)
        _check(client, 6)
        catalog = net.coordinator.registry.catalog
        for shard_id, worker_id in net.coordinator.plan.assignment():
            available = net.workers[worker_id].ram_free * (1 - 0.15)  # the default headroom
            assert catalog.shards[shard_id].bytes <= available, (shard_id, worker_id)
        _no_violations(net)  # no phone was ever asked to load more than it had free
    run(scenario())


def test_small_phones_get_a_finer_cut_than_big_ones():
    async def scenario():
        async def stages(ram):
            net, clock = _world([ram] * 6)
            await _heartbeats(net, clock)
            client = await submit(net, 4)
            await _drive(net, clock, client, 4)
            _check(client, 4)
            return net.coordinator.plan.assignment()
        small, big = await stages(150 * MiB), await stages(2 * GiB)
        assert len(small) > len(big) or max(map(_atoms, small)) <= max(map(_atoms, big))
        assert max(_atoms(s) for s in small) <= 2  # 150 MiB x 0.85 leaves room for at most a 108 MiB shard
    run(scenario())


def _atoms(stage):
    shard = MANIFEST.shard(stage[0])
    return shard.end_atom - shard.start_atom + 1


def test_a_phone_that_runs_low_on_ram_mid_run_makes_the_coordinator_replan():
    async def scenario():
        net, clock = _world([2 * GiB] * 7)
        await _heartbeats(net, clock)
        first = await submit(net, 4)
        await _drive(net, clock, first, 4)
        _check(first, 4)
        before = net.coordinator.plan.assignment()
        epoch = net.coordinator.epoch
        victim = before[1][1]
        net.workers[victim].ram_free = 40 * MiB  # the OS has taken the memory back
        await _heartbeats(net, clock, rounds=3)
        second = await submit(net, 4, start=100)
        await _drive(net, clock, second, 4)
        _check(second, 4, start=100)
        assert net.coordinator.epoch > epoch, "the coordinator re-planned"
        triggers = [t for e in net.coordinator.log.events if e["event"] == "controller_decision"
                    for t in e.get("triggers", [])]
        assert "ram_pressure" in triggers
        after = net.coordinator.plan.assignment()
        assert before[1] not in after or after != before
        assert net.workers[victim].ram_free == 40 * MiB
        # the shard the victim held is no longer the plan's choice for that phone
        assert (before[1][0], victim) not in after
        _no_violations(net)
    run(scenario())


def _loads_asked(net):
    return sum(len(w.load_msgs) for w in net.workers.values())


def test_a_phone_that_refuses_for_memory_is_not_asked_again_for_shards_that_big():
    async def scenario():
        # The phones report 400 MiB free but secretly keep 350 MiB back, so they can really hold almost nothing.
        net, clock = _world([400 * MiB] * 6)
        for worker in net.workers.values():
            worker.hidden_reserve = 350 * MiB
        await _heartbeats(net, clock)
        client = await submit(net, 4)
        await _drive(net, clock, client, 4, max_ticks=40)
        refusals = sum(w.memory_refusals for w in net.workers.values())
        assert refusals >= 1
        # Bounded: the phone says what it had, so each phone is refused about once, not once per shard size.
        assert all(w.memory_refusals <= 2 for w in net.workers.values()), [w.memory_refusals for w in net.workers.values()]
        assert _loads_asked(net) <= 6 * 2
        assert not _results(client)  # nothing could be placed, and the Foreman stopped trying
    run(scenario())


def test_memory_exclusions_are_lifted_when_a_phone_reports_much_more_free_ram():
    async def scenario():
        net, clock = _world([400 * MiB] * 6)
        for worker in net.workers.values():
            worker.hidden_reserve = 350 * MiB
        await _heartbeats(net, clock)
        client = await submit(net, 4)
        await _drive(net, clock, client, 4, max_ticks=40)
        assert net.coordinator.mem_excluded, "the refusals were remembered"
        for worker in net.workers.values():  # the OS frees memory: now they really can hold shards
            worker.ram_free = 2 * 1024 * MiB
        await _heartbeats(net, clock, rounds=3)
        assert not net.coordinator.mem_excluded, "the exclusions were released"
        await _drive(net, clock, client, 4, max_ticks=60)
        _check(client, 4)
    run(scenario())
