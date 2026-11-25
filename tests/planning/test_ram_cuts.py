"""BERT-base (a real SDK export, 8 atoms, 24 shards): the cut follows the free RAM the phones report."""

from pathlib import Path

import pytest

from foreman.planning.config import ControllerConfig
from foreman.planning.controller import ControllerState, decide, detect_triggers, Baseline
from foreman.planning.cost_model import memory_feasible, memory_need
from foreman.planning.planner import Objective
from foreman.planning.policies import SAwarePolicy, build_plan
from foreman.planning.registry import catalog_from_manifest, load_manifest
from tests.planning.fixtures import opts, profile, snapshot, worker

MiB = 1 << 20
GiB = 1 << 30
FIXTURE = Path(__file__).parent / "fixtures" / "bert_base_sst2"
CATALOG = catalog_from_manifest(load_manifest(FIXTURE)[0])
EMBED = "bert-base-sst2.embed"
HEAD = "bert-base-sst2.head"
FULL = "bert-base-sst2.full"
PROFILES = {"phone": profile(peak_mult=2.0, flops_per_s=5e9)}  # loading peaks at about twice the file, as measured


def phones(*rams, resident=None):
    return [worker(f"p{i}", ram=ram, resident=(resident or {}).get(f"p{i}")) for i, ram in enumerate(rams)]


def snap_of(workers):
    return snapshot(workers, CATALOG, PROFILES)


def plan_for(rams, **options):
    snap = snap_of(phones(*rams))
    return SAwarePolicy(opts(**options)).plan(snap, None, 100, Objective()), snap


def atoms_in(shard_id):
    info = CATALOG.shards[shard_id]
    return info.end_atom - info.start_atom + 1


def test_the_manifest_is_the_real_bert_base_export():
    assert CATALOG.n_atoms == 8 and len(CATALOG.shards) == 24
    assert CATALOG.shards[FULL].bytes == pytest.approx(417.8 * MiB, rel=0.01)
    assert CATALOG.shards[EMBED].pinned and CATALOG.shards[HEAD].pinned


def test_phones_with_plenty_of_ram_run_the_model_in_a_few_large_stages():
    decision, snap = plan_for([3 * GiB] * 3)
    assert decision.feasible
    shards = [s for s, _ in decision.stages]
    assert len(shards) <= 3 and max(atoms_in(s) for s in shards) >= 3
    assert all(memory_feasible(snap, w, s, opts()) for s, w in decision.stages)


def test_tight_ram_forces_a_finer_cut():
    # 300 MiB free: a shard may be at most 150 MiB, so two layers-groups (108 MiB) fit and three (162 MiB) do not.
    decision, snap = plan_for([300 * MiB] * 5)
    assert decision.feasible
    shards = [s for s, _ in decision.stages]
    encoder = [s for s in shards if CATALOG.shards[s].role == "encoder"]
    assert max(atoms_in(s) for s in encoder) <= 2
    assert len(shards) == 5 and shards[0] == EMBED and shards[-1] == HEAD
    assert all(memory_feasible(snap, w, s, opts()) for s, w in decision.stages)
    assert FULL not in shards


def test_plenty_of_ram_does_not_stop_the_model_being_split_for_throughput():
    decision, snap = plan_for([3 * GiB] * 5)
    assert decision.feasible and len(decision.stages) == 5  # embed, three encoder pairs, head
    assert FULL not in [s for s, _ in decision.stages]


def test_one_big_phone_among_tiny_ones_runs_the_whole_model():
    # The other phones can hold only the 2 MiB head, so the one phone that can must take everything else.
    decision, snap = plan_for([3 * GiB, 100 * MiB, 100 * MiB])
    assert decision.feasible
    assert [(s, w) for s, w in decision.stages] == [(FULL, "p0")]
    assert memory_feasible(snap, "p0", FULL, opts())
    assert not memory_feasible(snap, "p1", FULL, opts())


def test_the_embedding_needs_a_phone_with_enough_ram():
    snap = snap_of(phones(150 * MiB, 250 * MiB))
    need = memory_need(snap, "p0", EMBED, opts())
    assert need == pytest.approx(2 * CATALOG.shards[EMBED].bytes)  # about 182 MiB
    assert not memory_feasible(snap, "p0", EMBED, opts())
    assert memory_feasible(snap, "p1", EMBED, opts())


def test_when_nothing_fits_the_planner_says_so_instead_of_returning_a_bad_plan():
    decision, _ = plan_for([100 * MiB] * 6)  # the embedding alone needs about 182 MiB
    assert not decision.feasible and decision.stages is None


def test_unknown_ram_is_not_a_reason_to_refuse():
    snap = snap_of(phones(None, None, None))
    assert memory_feasible(snap, "p0", FULL, opts())


def test_headroom_makes_the_ram_estimate_conservative():
    enc = "bert-base-sst2.enc.a0-a1"  # 108 MiB file, needs about 217 MiB while loading
    snap = snap_of(phones(230 * MiB))
    assert memory_feasible(snap, "p0", enc, opts())
    assert memory_feasible(snap, "p0", enc, opts(ram_headroom_frac=0.05))
    assert not memory_feasible(snap, "p0", enc, opts(ram_headroom_frac=0.15))


def test_a_phone_that_runs_low_on_ram_can_no_longer_keep_its_shard():
    enc = "bert-base-sst2.enc.a0-a1"
    critical = opts(ram_critical_bytes=128 * MiB)
    snap = snap_of(phones(40 * MiB, 500 * MiB, None, resident={"p0": enc, "p1": enc, "p2": enc}))
    assert not memory_feasible(snap, "p0", enc, critical)  # 40 MiB left: under pressure
    assert memory_feasible(snap, "p1", enc, critical)
    assert memory_feasible(snap, "p2", enc, critical)  # unknown RAM is not pressure
    assert memory_feasible(snap, "p0", enc, opts())  # the floor is off by default


def test_low_ram_on_a_plan_worker_is_a_trigger_only_when_the_floor_is_set():
    snap = snap_of(phones(3 * GiB, 3 * GiB, 3 * GiB, 3 * GiB))
    plan = build_plan(snap, ((EMBED, "p0"), ("bert-base-sst2.enc.a0-a5", "p1"), (HEAD, "p2")), epoch=1,
                      reason="init", opts=opts())
    low = snap_of(phones(3 * GiB, 60 * MiB, 3 * GiB, 3 * GiB))
    cfg = ControllerConfig(persist_k=1)
    base = Baseline.capture(snap)
    assert "ram_pressure" not in detect_triggers(low, plan, base, cfg)
    assert "ram_pressure" in detect_triggers(low, plan, base, cfg, ram_critical_bytes=128 * MiB)
    assert "ram_pressure" not in detect_triggers(snap, plan, base, cfg, ram_critical_bytes=128 * MiB)


def _pressure_setup(rams):
    snap = snap_of(phones(*rams))
    stages = ((EMBED, "p0"), ("bert-base-sst2.enc.a0-a2", "p1"), ("bert-base-sst2.enc.a3-a5", "p2"), (HEAD, "p3"))
    snap0 = snap_of(phones(3 * GiB, 3 * GiB, 3 * GiB, 3 * GiB, 3 * GiB))
    plan = build_plan(snap0, stages, epoch=1, reason="init", opts=opts())
    resident = {"p0": EMBED, "p1": "bert-base-sst2.enc.a0-a2", "p2": "bert-base-sst2.enc.a3-a5", "p3": HEAD}
    snap = snap_of(phones(*rams, resident=resident))
    return snap, plan


def test_persistent_memory_pressure_moves_or_cuts_the_shard_and_a_blip_does_not():
    o = opts(ram_critical_bytes=128 * MiB)
    # p1 is nearly out of RAM; p4 is an idle phone with plenty.
    snap, plan = _pressure_setup([3 * GiB, 60 * MiB, 3 * GiB, 3 * GiB, 3 * GiB])
    cfg = ControllerConfig(persist_k=2, cooldown_s=0, margin_abs_s=0.5, margin_frac=0.0)
    policy = SAwarePolicy(o)
    first, state = decide(snap, plan, 200, policy, Objective(), ControllerState(), cfg, ("ram_pressure",), 100.0)
    assert first.action == "stay"  # one low reading is not enough
    second, _ = decide(snap, plan, 200, policy, Objective(), state, cfg, ("ram_pressure",), 101.0)
    assert second.action == "forced_switch" and second.reason == "ram_pressure"
    assert ("bert-base-sst2.enc.a0-a2", "p1") not in second.target
    target = dict((w, s) for s, w in second.target)
    assert all(memory_feasible(snap, w, s, o) for w, s in target.items() if snap.workers[w].resident_shard != s)


def test_when_every_phone_is_short_of_ram_the_current_plan_keeps_serving():
    o = opts(ram_critical_bytes=128 * MiB)
    snap, plan = _pressure_setup([60 * MiB] * 5)
    cfg = ControllerConfig(persist_k=1, cooldown_s=0)
    decision, _ = decide(snap, plan, 200, SAwarePolicy(o), Objective(), ControllerState(), cfg, ("ram_pressure",), 100.0)
    assert decision.action == "stay" and decision.reason == "no plan fits the free memory"
