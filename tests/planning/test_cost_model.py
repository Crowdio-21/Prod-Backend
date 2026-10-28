"""Hand-computed cost-model cases for all three residency states."""

import pytest

from foreman.planning.cost_model import (
    Residency,
    c_fetch,
    c_switch,
    memory_feasible,
    pipeline_cost,
    t_comm,
)
from tests.planning.fixtures import MB, link, opts, profile, snapshot, worker

# Profile: load 1.0 s, init 0.5 s, write 0.2 s, unload 0.3 s. Foreman link 10 MB/s, rtt 0.1 s.
# Shard "b" is 100 MB.


def test_resident_costs_nothing():
    snap = snapshot([worker("w", resident="b")])
    cost = c_switch(snap, "w", "b", opts())
    assert cost.residency is Residency.RESIDENT
    assert cost.total.mid == 0.0


def test_on_disk_with_other_shard_resident():
    snap = snapshot([worker("w", resident="a", disk={"b"})])
    cost = c_switch(snap, "w", "b", opts())
    assert cost.residency is Residency.ON_DISK
    assert cost.total.mid == pytest.approx(0.3 + 1.0 + 0.5)  # unload + load + init


def test_on_disk_nothing_resident():
    snap = snapshot([worker("w", disk={"b"})])
    assert c_switch(snap, "w", "b", opts()).total.mid == pytest.approx(1.0 + 0.5)


def test_absent_fetches_from_foreman():
    snap = snapshot([worker("w")])
    cost = c_switch(snap, "w", "b", opts())
    assert cost.residency is Residency.ABSENT
    fetch = 100 * MB / (10 * MB) + 0.1
    assert cost.total.mid == pytest.approx(fetch + 0.2 + 1.0 + 0.5)
    assert cost.source is None


def _peer_snapshot(**donor_kwargs):
    donor = worker("d", disk={"b"}, **donor_kwargs)
    return snapshot(
        [worker("w"), donor],
        links={("d", "w"): link(50 * MB, 0.01)},
        capability={("d", "w"): True},
    )


def test_absent_with_peer_fetch_uses_faster_donor():
    snap = _peer_snapshot()
    cost = c_switch(snap, "w", "b", opts(peer_fetch=True))
    assert cost.source == "d"
    assert cost.total.mid == pytest.approx(100 / 50 + 0.01 + 0.2 + 1.0 + 0.5)


def test_absent_without_peer_fetch_ignores_donor():
    snap = _peer_snapshot()
    cost = c_switch(snap, "w", "b", opts(peer_fetch=False))
    assert cost.source is None
    assert cost.total.mid == pytest.approx(10.1 + 0.2 + 1.5)


@pytest.mark.parametrize("donor_kwargs,critical", [
    ({"battery": 10.0}, frozenset()),  # battery headroom
    ({"thermal": 5}, frozenset()),  # heat headroom
    ({}, frozenset({"d"})),  # on the critical path
])
def test_ineligible_donors_are_skipped(donor_kwargs, critical):
    snap = _peer_snapshot(**donor_kwargs)
    fetch = c_fetch(snap, "w", snap.catalog.shards["b"], opts(peer_fetch=True, critical_workers=critical))
    assert fetch.source is None


def test_donor_upload_cap():
    snap = _peer_snapshot()
    snap = snap.__class__(**{**snap.__dict__, "uploads_in_flight": {"d": 1}})
    assert c_fetch(snap, "w", snap.catalog.shards["b"], opts(peer_fetch=True)).source is None


def test_relay_link_uses_min_bandwidth_and_summed_rtt():
    snap = snapshot([worker("i", bw=4 * MB, rtt=0.05), worker("j", bw=8 * MB, rtt=0.02)], compression=0.5)
    cost = t_comm(snap, "i", "j", 8 * MB, opts())
    assert cost.route == "via_foreman"
    assert cost.time.mid == pytest.approx(8 * 0.5 / 4 + 0.07)


def test_direct_link_used_only_when_allowed_and_cheaper():
    snap = snapshot(
        [worker("i", bw=4 * MB, rtt=0.05), worker("j", bw=8 * MB, rtt=0.02)],
        links={("i", "j"): link(40 * MB, 0.005)}, capability={("i", "j"): True},
    )
    assert t_comm(snap, "i", "j", 8 * MB, opts(direct_path=False)).route == "via_foreman"
    direct = t_comm(snap, "i", "j", 8 * MB, opts(direct_path=True))
    assert direct.route == "direct"
    assert direct.time.mid == pytest.approx(8 / 40 + 0.005)


def test_pipeline_formula_by_hand():
    # Two stages a (w1, resident) -> c (w2, absent, 10 MB). Activation 1000 bytes/token, 1 token.
    p = profile(compute={"a": 2.0, "c": 1.0})
    snap = snapshot([worker("w1", resident="a"), worker("w2")], profiles={"phone": p})
    o = opts(include_io_legs=True)
    stages = (("a", "w1"), ("c", "w2"))
    cost = pipeline_cost(snap, stages, n=5, opts=o)
    comm = 1000 / (10 * MB) + 0.1  # Foreman leg
    relay = 1000 / (10 * MB) + 0.2  # w1 -> w2 via Foreman (rtt summed)
    ready2 = 10 * MB / (10 * MB) + 0.1 + 0.2 + 1.0 + 0.5
    finish1 = max(0.0, 0.0 + comm) + 2.0
    finish2 = max(ready2, finish1 + relay) + 1.0
    period = max(2.0, 1.0, comm, relay)
    out = 1000 / (10 * MB) + 0.1
    assert cost.finish_K.mid == pytest.approx(finish2)
    assert cost.T_steady.mid == pytest.approx(period)
    assert cost.T_total.mid == pytest.approx(finish2 + out + 4 * period)
    assert cost.C_switch_total.mid == pytest.approx(ready2)


def test_pipeline_formula_without_io_legs_matches_brief():
    p = profile(compute={"a": 2.0, "c": 1.0})
    snap = snapshot([worker("w1", resident="a"), worker("w2", resident="c")], profiles={"phone": p})
    cost = pipeline_cost(snap, (("a", "w1"), ("c", "w2")), n=3, opts=opts(include_io_legs=False))
    relay = 1000 / (10 * MB) + 0.2
    assert cost.T_total.mid == pytest.approx((2.0 + relay + 1.0) + 2 * 2.0)


def test_uncertainty_bounds_widen_with_spread_and_age():
    from foreman.planning.types import Est

    snap = snapshot([worker("w").__class__(**{**worker("w").__dict__, "down": Est(10 * MB, 2 * MB, 5, 10.0)})])
    tight = c_switch(snap, "w", "b", opts(age_growth=0.0))
    aged = c_switch(snap, "w", "b", opts(age_growth=0.01))
    assert tight.total.lo < tight.total.mid < tight.total.hi
    assert aged.total.hi - aged.total.lo > tight.total.hi - tight.total.lo


def test_memory_feasibility_counts_resident_release():
    # peak = shard bytes (mult 1.0); b needs 100 MB.
    snap = snapshot([worker("w", resident="a", ram=60 * MB)])
    assert memory_feasible(snap, "w", "b", opts())  # 60 MB free + 100 MB released by a
    assert not memory_feasible(snap, "w", "b", opts(), keep_resident=True)
    snap = snapshot([worker("w", ram=60 * MB)])
    assert not memory_feasible(snap, "w", "b", opts())
