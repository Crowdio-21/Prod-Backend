"""Adaptive model sizing on the real BERT-base export: the variant follows the free RAM the phones report."""

from pathlib import Path

import pytest

from foreman.planning.adaptive import (
    AdaptivePlanner,
    derive_precision_variant,
    measure_capacity,
    precision_ladder,
    quick_fit,
    snapshot_for,
    variant_from_manifest,
)
from foreman.planning.cost_model import memory_feasible
from foreman.planning.planner import Objective
from tests.planning.fixtures import opts, profile, snapshot, worker

MiB = 1 << 20
GiB = 1 << 30
FIXTURE = Path(__file__).parent / "fixtures" / "bert_base_sst2"
BASE = variant_from_manifest(FIXTURE, quality=0.930)
LADDER = precision_ladder(BASE, {"fp16": 0.929, "int8": 0.921})
FP32, FP16, INT8 = (v.name for v in LADDER)
PROFILES = {"phone": profile(peak_mult=2.0, flops_per_s=5e9)}


def snap_of(*rams, resident=None):
    resident = resident or {}
    return snapshot([worker(f"p{i}", ram=r, resident=resident.get(f"p{i}")) for i, r in enumerate(rams)],
                    BASE.catalog, PROFILES)


def planner(**kw):
    return AdaptivePlanner(LADDER, opts(), **kw)


def test_derived_variants_scale_the_weights_and_keep_the_cut_grid():
    assert BASE.precision == "fp32" and not BASE.derived
    fp16, int8 = LADDER[1], LADDER[2]
    assert fp16.derived and int8.derived
    assert fp16.catalog.n_atoms == int8.catalog.n_atoms == BASE.catalog.n_atoms
    assert len(int8.catalog.compositions()) == len(BASE.catalog.compositions())
    assert int8.full_bytes == pytest.approx(BASE.full_bytes * 0.25 * 1.02, rel=0.01)
    assert all(sid.endswith("@int8") for sid in int8.catalog.shards)


def test_plenty_of_ram_runs_the_full_precision_model():
    decision = planner().plan(snap_of(3 * GiB, 3 * GiB, 3 * GiB), 100)
    assert decision.feasible and decision.variant.name == FP32
    assert [a.outcome for a in decision.attempts] == ["chosen"]


@pytest.mark.parametrize("ram_mib, expected", [(400, "fp16"), (200, "int8")])
def test_less_ram_steps_down_the_ladder(ram_mib, expected):
    snap = snap_of(*[ram_mib * MiB] * 3)
    decision = planner().plan(snap, 100)
    assert decision.feasible and decision.variant.precision == expected
    vsnap = snapshot_for(snap, decision.variant, LADDER)
    assert all(memory_feasible(vsnap, w, s, opts()) for s, w in decision.stages)
    assert decision.attempts[0].outcome in ("infeasible", "too_big")


def test_more_phones_buy_back_the_larger_model():
    # 3 x 300 MiB only holds int8; 5 x 300 MiB holds the fp32 model cut five ways.
    assert planner().plan(snap_of(*[300 * MiB] * 3), 100).variant.name == INT8
    wide = planner().plan(snap_of(*[300 * MiB] * 5), 100)
    assert wide.variant.name == FP32 and len(wide.stages) == 5


def test_nothing_fits_reports_why():
    decision = planner().plan(snap_of(60 * MiB, 60 * MiB, 60 * MiB), 100)
    assert not decision.feasible and decision.variant is None
    assert {a.outcome for a in decision.attempts} <= {"too_big", "infeasible"}
    assert decision.summary()["capacity"]["total"] == pytest.approx(180 * MiB)


def test_quick_fit_rejects_a_model_bigger_than_all_phones_together():
    snap = snap_of(100 * MiB, 100 * MiB)
    ok, why = quick_fit(LADDER[0], snap, measure_capacity(snap, opts()), opts())
    assert not ok and "in total" in why


def test_unknown_ram_does_not_block_the_largest_model():
    decision = planner().plan(snap_of(None, None, None), 100)
    assert decision.variant.name == FP32


def test_a_resident_shard_of_another_variant_counts_as_freeable():
    int8_full = LADDER[2].catalog.full_shard_id
    # 330 MiB free plus a resident 109 MiB int8 model: fp16 (needs ~427 MiB to load in one piece) fits after unloading.
    snap = snap_of(330 * MiB, resident={"p0": int8_full})
    fp16_snap = snapshot_for(snap, LADDER[1], LADDER)
    assert fp16_snap.workers["p0"].resident_shard is None
    assert fp16_snap.workers["p0"].ram_free == 330 * MiB + LADDER[2].catalog.shards[int8_full].bytes
    assert planner().plan(snap, 100).variant.name == FP16


def test_upgrade_needs_extra_headroom_but_downgrade_is_immediate():
    p = planner(upgrade_headroom=0.20)
    assert p.plan(snap_of(*[200 * MiB] * 3), 100).variant.name == INT8
    # RAM recovers to just enough for fp16: without the headroom it would switch up, with it the phone stays on int8.
    just_enough = snap_of(*[390 * MiB] * 3)
    assert planner(upgrade_headroom=0.0).plan(just_enough, 100).variant.name == FP16
    held = p.plan(just_enough, 100)
    assert held.variant.name == INT8
    assert any(a.outcome == "upgrade_blocked" for a in held.attempts)
    # Plenty of room: the upgrade goes through.
    assert p.plan(snap_of(*[3 * GiB] * 3), 100).variant.name == FP32
    # Memory drops again: straight down, no headroom asked for.
    assert p.plan(snap_of(*[200 * MiB] * 3), 100).variant.name == INT8


def test_deadline_prefers_a_smaller_faster_variant():
    fast_int8 = derive_precision_variant(BASE, "int8", 0.921, compute_factor=0.25)
    p = AdaptivePlanner([BASE, fast_int8], opts())
    snap = snap_of(*[3 * GiB] * 3)
    free = p.plan(snap, 100, commit=False)
    assert free.variant.name == FP32
    t32 = free.decision.cost.T_total.mid
    deadline = p.plan(snap, 100, Objective("min_total", deadline_s=t32 * 0.8), commit=False)
    assert deadline.variant.name == fast_int8.name
    assert deadline.attempts[0].outcome == "misses_deadline"


def test_ladder_rejects_clashing_shard_ids():
    with pytest.raises(ValueError):
        AdaptivePlanner([BASE, BASE], opts())
