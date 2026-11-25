"""Disk space: a shard that is not on a worker's disk must fit under its cap, after LRU eviction."""

import asyncio

from tests.planning.fake_workers import MB, FakeClock, FakeNetwork, FakeWorker, make_coordinator, make_manifest, submit
from tests.planning.test_integration_fake_workers import _check_results, _drive, _heartbeats, _no_violations


def run(coro):
    return asyncio.run(coro)


def _world(spare_cap, spare_disk=()):
    """w1/w2/w3 hold a/b/c (a=50 MB, b=60 MB, c=5 MB); 'spare' has a disk cap and some files."""
    net, clock = FakeNetwork(), FakeClock()
    make_coordinator(net, clock, manifest=make_manifest(b_bytes=60 * MB))
    FakeWorker("w1", net, resident="a")
    FakeWorker("w2", net, resident="b")
    FakeWorker("w3", net, resident="c")
    FakeWorker("spare", net, disk=set(spare_disk), disk_cap=spare_cap)
    for worker_id in net.workers:
        net.connect(worker_id, "phone")
    return net, clock


async def _lose_w2(net, clock):
    await _heartbeats(net, clock)
    first = await submit(net, 2)
    await _drive(net, clock, first, 2)
    net.workers["w2"].disconnect()
    await net.coordinator.worker_disconnected("w2")
    return first


def test_a_worker_whose_disk_cannot_hold_the_shard_is_never_chosen():
    async def scenario():
        net, clock = _world(spare_cap=40 * MB)  # b is 60 MB: cannot fit even on an empty disk
        await _lose_w2(net, clock)
        client = await submit(net, 3, start=100)
        await _drive(net, clock, client, 3, max_ticks=10)
        spare = net.workers["spare"]
        assert all(shard != "b" for shard, _ in spare.loads)  # b (60 MB) cannot fit under a 40 MB cap
        assert spare.violations == []  # and nothing that cannot fit was ever sent to it
        _no_violations(net)
    run(scenario())


def test_the_foreman_tells_the_worker_which_files_to_delete_to_make_room():
    async def scenario():
        # cap 100 MB, holds a (50) and c (5): 55 + 60 > 100, so one file must go; both are evictable.
        net, clock = _world(spare_cap=100 * MB, spare_disk={"a", "c"})
        await _lose_w2(net, clock)
        client = await submit(net, 3, start=100)
        await _drive(net, clock, client, 3)
        spare = net.workers["spare"]
        assert ("b", net.coordinator.epoch) in spare.loads
        assert spare.evicted, "an eviction was required to fit b"
        assert spare._disk_used() <= 100 * MB  # fits after the deletions
        assert spare.violations == []
        _check_results(client, 3, start=100)
        _no_violations(net)
    run(scenario())


def test_nothing_is_evicted_when_the_file_already_fits():
    async def scenario():
        net, clock = _world(spare_cap=500 * MB, spare_disk={"a"})
        await _lose_w2(net, clock)
        client = await submit(net, 3, start=100)
        await _drive(net, clock, client, 3)
        spare = net.workers["spare"]
        assert ("b", net.coordinator.epoch) in spare.loads and spare.evicted == []
        assert spare.evict_requests == [[]]
        _no_violations(net)
    run(scenario())
