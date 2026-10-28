"""Contract, flags, config, telemetry, registry, routes, problems and prefetch."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from foreman.planning.config import PrefetchConfig, RouteConfig, RunConfig
from foreman.planning.contract import (
    Envelope,
    Manifest,
    ProblemMsg,
    ProblemType,
    WorkerState,
    is_v2_envelope,
    make_envelope,
)
from foreman.planning.eventlog import EventLog, stable_hash
from foreman.planning.flags import FeatureFlags
from foreman.planning.prefetch import plan_prefetch
from foreman.planning.problems import ProblemAggregator
from foreman.planning.registry import DiskEntry, manifest_from_legacy, plan_evictions
from foreman.planning.routes import CapabilityTracker, RouteSelector
from foreman.planning.telemetry import TelemetryStore, legacy_to_worker_state
from foreman.planning.policies import build_plan
from tests.planning.fixtures import MB, link, opts, snapshot, three_stage_catalog, worker

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- flags/config
def test_flags_default_off_and_parse():
    assert not FeatureFlags().any_on
    flags = FeatureFlags.from_env({"CROWDIO_FLAGS": "adaptive_planning,prefetch"})
    assert flags.adaptive_planning and flags.prefetch and not flags.direct_path
    assert FeatureFlags.from_env({"CROWDIO_FLAGS": '{"peer_fetch": true}'}).peer_fetch
    with pytest.raises(ValueError):
        FeatureFlags.from_mapping({"nope": True})


def test_run_config_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "policy: S-aware\nseed: 7\nflags: {adaptive_planning: true}\ncontroller: {persist_k: 3}\n",
        encoding="utf-8",
    )
    config = RunConfig.load(path)
    assert config.policy == "S-aware" and config.seed == 7
    assert config.flags.adaptive_planning and config.controller.persist_k == 3
    with pytest.raises(ValueError):
        RunConfig(policy="nope")


# --------------------------------------------------------------- contract
def test_envelope_round_trip_and_validation():
    raw = make_envelope("problem", ProblemMsg(type=ProblemType.OOM, stage=1, epoch=2), msg_id="m1", epoch=2)
    assert is_v2_envelope(raw)
    body = Envelope.model_validate(raw).body()
    assert body.type is ProblemType.OOM
    with pytest.raises(ValueError):
        Envelope.model_validate({**raw, "type": "bogus"})
    # v1 envelopes are not v2
    assert not is_v2_envelope({"protocol": "crowdio.reconfiguration", "version": 1})


def test_worker_state_ignores_unknown_fields():
    state = WorkerState.model_validate({"worker_id": "w", "future_field": 1, "cpu": 0.5})
    assert state.cpu == 0.5


def test_legacy_manifest_adapter_matches_export():
    manifest = manifest_from_legacy(ROOT / "tools" / "onnx_partitions_sentiment", hash_files=False)
    assert [s.shard_id for s in manifest.shards] == ["cell_a", "cell_b", "cell_c"]
    assert [a.name for a in manifest.atoms] == ["embed", "L0", "L1", "L2", "L3", "L4", "L5", "head"]
    assert manifest.shard("cell_c").pinned
    with pytest.raises(ValueError):
        Manifest.model_validate({**manifest.model_dump(), "default_composition": ["cell_a", "cell_c"]})


# --------------------------------------------------------------- telemetry
def test_ewma_age_and_samples():
    clock = [0.0]
    store = TelemetryStore(clock=lambda: clock[0])
    store.observe_transfer("w", "down", 10 * MB, 1.0)
    clock[0] = 5.0
    store.observe_transfer("w", "down", 20 * MB, 1.0)
    clock[0] = 8.0
    snap = store.snapshot(three_stage_catalog(), {})
    est = snap.workers["w"].down
    assert est.n == 2 and est.age_s == pytest.approx(3.0)
    assert 10 * MB < est.mean < 20 * MB and est.std > 0
    assert snap.workers["w"].up.n == 0  # prior, stale


def test_probe_only_when_old_and_rate_limited():
    clock = [0.0]
    store = TelemetryStore(clock=lambda: clock[0])
    store.ingest_state(WorkerState(worker_id="w", session_state="SERVING"))
    store.observe_transfer("w", "down", MB, 0.1)
    store.observe_transfer("w", "up", MB, 0.1)
    assert not store.needs_probe("w")
    clock[0] = 100.0
    assert store.needs_probe("w")
    store.mark_probed("w")
    clock[0] = 110.0
    assert not store.needs_probe("w")


def test_legacy_heartbeats_both_shapes():
    nested = legacy_to_worker_state("pc", {"performance_metrics": {"cpu_usage_percent": 50, "ram_available_mb": 100}})
    flat = legacy_to_worker_state("pc", {"cpu_usage_percent": 50, "ram_available_mb": 100, "network_speed_mbps": 900})
    assert nested.cpu == flat.cpu == 0.5 and nested.ram_free == flat.ram_free
    assert flat.observed_down_bps is None  # NIC utilisation is not bandwidth
    android = legacy_to_worker_state("a", {
        "net": {"bw_down": 80e6, "bw_up": 16e6},
        "res": {"free_ram_mb": 900, "thermal": 1, "battery_pct": 70, "cpu_load": 0.3},
        "residency": {"state": "SERVING", "shard": "task-123"},
        "on_disk": [{"id": "cell_b", "bytes": 5}],
    })
    assert android.observed_down_bps == 10e6 and android.device_class == "android"
    assert android.resident_shard is None  # task id is not trusted as a shard id
    assert android.disk_shards[0].id == "cell_b"


def test_peer_links_from_heartbeat():
    store = TelemetryStore(clock=lambda: 10.0)
    store.ingest_state(WorkerState(worker_id="a", peer_links=[
        {"peer": "b", "reachable": True, "rtt_ms": 4, "bw_est": 30 * MB, "age_s": 2}]))
    snap = store.snapshot(three_stage_catalog(), {})
    assert snap.capability[("a", "b")]
    assert snap.links[("a", "b")].bw.mean == 30 * MB and snap.links[("a", "b")].bw.age_s == pytest.approx(2)


# --------------------------------------------------------------- registry
def test_lru_never_evicts_resident():
    entries = [DiskEntry("old", 50, 1.0), DiskEntry("resident", 50, 0.0), DiskEntry("newer", 50, 5.0)]
    assert plan_evictions(entries, 150, 200, 100, protected={"resident"}) == ["old"]
    assert plan_evictions(entries, 150, 150, 100, protected={"resident"}) == ["old", "newer"]
    assert plan_evictions(entries, 150, 200, 40) == []
    assert plan_evictions(entries, 150, 160, 200, protected={"resident"}) is None
    assert plan_evictions(entries, 150, None, 10**12) == []


# --------------------------------------------------------------- routes
def _route_snap(direct_bw):
    return snapshot(
        [worker("i", bw=2 * MB, rtt=0.05), worker("j", bw=2 * MB, rtt=0.05)],
        links={("i", "j"): link(direct_bw, 0.005)}, capability={("i", "j"): True},
    )


def test_route_hysteresis_and_dwell():
    selector = RouteSelector(RouteConfig(margin_frac=0.2, min_dwell_s=10))
    o = opts(direct_path=True)
    choice = selector.choose(_route_snap(20 * MB), "i", "j", MB, o, now=0)
    assert choice.mode.value == "direct" and choice.flipped
    # Slightly worse direct link: within margin, hold.
    assert not selector.choose(_route_snap(2.1 * MB), "i", "j", MB, o, now=20).flipped
    # Much worse but dwell not elapsed since the flip at t=0? elapsed -> flips back.
    back = selector.choose(_route_snap(0.2 * MB), "i", "j", MB, o, now=5)
    assert not back.flipped  # dwell blocks
    back = selector.choose(_route_snap(0.2 * MB), "i", "j", MB, o, now=11)
    assert back.flipped and back.mode.value == "via_foreman"


def test_unreachable_peer_falls_back_immediately():
    selector = RouteSelector(RouteConfig(min_dwell_s=1000))
    o = opts(direct_path=True)
    selector.choose(_route_snap(20 * MB), "i", "j", MB, o, now=0)
    blocked = snapshot([worker("i"), worker("j")], links={("i", "j"): link(20 * MB, 0.005, reachable=False)},
                       capability={("i", "j"): True})
    choice = selector.choose(blocked, "i", "j", MB, o, now=1)
    assert choice.mode.value == "via_foreman" and choice.flipped


def test_capability_retest_rules():
    tracker = CapabilityTracker(RouteConfig(capability_retest_s=100))
    assert tracker.needs_test("a", "b", 0)
    tracker.record("a", "b", True, 0)
    assert not tracker.needs_test("a", "b", 50) and tracker.needs_test("a", "b", 101)
    tracker.network_changed("a")
    assert tracker.needs_test("a", "b", 51)


# --------------------------------------------------------------- problems
def test_problem_dedupe_rate_limit_and_urgency():
    agg = ProblemAggregator(per_worker_rate=1.0, per_worker_burst=3, dedupe_window_s=10)
    slow = ProblemMsg(type=ProblemType.SLOW_STAGE, stage=1, epoch=1)
    assert agg.accept("w", slow, 0).status == "counted"
    assert agg.accept("w", slow, 0.1).status == "duplicate"
    assert agg.accept("w", ProblemMsg(type=ProblemType.OOM, stage=1, epoch=1), 0.2).status == "urgent"
    assert agg.accept("w", ProblemMsg(type=ProblemType.LINK_DEGRADED, stage=2, epoch=1), 0.3).status == "rate_limited"
    assert agg.recent_count(1.0) == 1


def test_problem_storm_is_capped():
    agg = ProblemAggregator(global_cap_per_s=5, per_worker_burst=100, per_worker_rate=100)
    statuses = [agg.accept(f"w{i}", ProblemMsg(type=ProblemType.SLOW_STAGE, stage=i, epoch=1), 0.0).status
                for i in range(20)]
    assert statuses.count("storm_dropped") == 15


# --------------------------------------------------------------- prefetch
def test_prefetch_respects_battery_heat_link_and_disk():
    catalog = three_stage_catalog()
    workers = [
        worker("w0", resident="a"), worker("w1", resident="b"), worker("w2", resident="c"),
        worker("ok"), worker("flat", battery=5.0), worker("hot", thermal=4), worker("busy"),
    ]
    snap = snapshot(workers, catalog)
    plan = build_plan(snap, (("a", "w0"), ("b", "w1"), ("c", "w2")), epoch=1, reason="t", opts=opts(), n_backups=10)
    orders = plan_prefetch(snap, plan, [], PrefetchConfig(), link_utilisation={"busy": 0.9})
    targets = {o.worker_id for o in orders}
    assert "ok" in targets
    assert not targets & {"flat", "hot", "busy", "w0", "w1", "w2"}


def test_prefetch_evicts_lru_but_not_resident():
    full_disk = replace(worker("x", resident="a", disk={"a", "c"}), disk_used=110 * MB, disk_cap=205 * MB)
    snap = snapshot([full_disk])
    orders = plan_prefetch(snap, None, [(("b", "x"),)], PrefetchConfig(), link_utilisation={},
                           last_used={("x", "c"): 1.0, ("x", "a"): 0.0})
    assert orders[0].evict == ("c",)


# --------------------------------------------------------------- event log
def test_event_log_jsonl(tmp_path):
    path = tmp_path / "log.jsonl"
    log = EventLog(path, run_id="r", seed=3, policy="S-aware", flags={"prefetch": True}, clock=lambda: 1.5)
    log.epoch = 4
    log.emit("planner_call", inputs_hash=stable_hash({"a": 1}), chosen=[["a", "w"]])
    log.close()
    record = json.loads(path.read_text().strip())
    assert record["epoch"] == 4 and record["seed"] == 3 and record["event"] == "planner_call"
    with pytest.raises(ValueError):
        EventLog().emit("nope")
