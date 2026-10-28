"""Simulator, replay, metrics and scenario hooks."""

from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.reconfig.metrics import compute_metrics
from evaluation.reconfig.replay import replay
from evaluation.reconfig.scenario import LiveScenarioRunner, ScenarioEvent, load_scenario
from evaluation.reconfig.sim import Simulator
from evaluation.reconfig.spec import load_spec
from foreman.planning.config import RunConfig
from foreman.planning.eventlog import read_events
from foreman.planning.flags import FeatureFlags
from foreman.planning.serialize import snapshot_from_dict, snapshot_to_dict
from tests.planning.fixtures import random_snapshot

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "evaluation" / "reconfig" / "configs"


def _config(**kw):
    base = RunConfig.load(CONFIGS / "run_example.yaml")
    return replace(base, n_requests=120, **kw)


def _sim(config, oracle=False, log_path=None):
    spec = load_spec(CONFIGS / "spec_distilbert_5phones.yaml")
    return Simulator(spec, config, load_scenario(config.scenario), oracle=oracle, log_path=log_path).run()


def test_snapshot_round_trip():
    import random

    snap = random_snapshot(random.Random(1))
    assert snapshot_to_dict(snapshot_from_dict(snapshot_to_dict(snap))) == snapshot_to_dict(snap)


def test_simulation_completes_and_replays_identically(tmp_path):
    config = _config(policy="S-aware", scenario=[{"at": 20, "action": "network", "worker": "p2", "bw": 1.5e5}])
    log = tmp_path / "run.jsonl"
    result = _sim(config, log_path=str(log))
    assert result.completed == 120
    events = read_events(log)
    outcome = replay(events)
    assert outcome["checked"] > 0 and outcome["identical"], outcome["mismatches"][:3]


def test_oracle_is_a_lower_bound_on_this_scenario():
    config = _config(policy="S-aware")
    oracle = _sim(config, oracle=True)
    for policy in ("B0", "B2", "S-aware"):
        flags = FeatureFlags() if policy == "B0" else config.flags
        result = _sim(replace(config, policy=policy, flags=flags))
        assert result.total_time >= oracle.total_time - 1e-6, policy


def test_metrics_report_required_fields():
    config = _config(policy="B2")
    result = _sim(config)
    oracle = _sim(config, oracle=True)
    m = compute_metrics(result.log.events, oracle_total=oracle.total_time)
    for key in ("total_completion_s", "reconfiguration_s", "inference_s", "switches", "stall_s",
                "wasted_switches", "regret_s", "route_shares", "route_flips", "bytes_moved",
                "planner_time_s", "deadline_hit_rate", "control_bytes_per_request"):
        assert key in m
    assert m["regret_s"] >= -1e-6
    assert m["requests_completed"] == 120


def test_no_fault_run_never_switches():
    config = replace(_config(policy="S-aware", flags=FeatureFlags.all_on()), scenario=[])
    result = _sim(config)
    assert result.switches == 0 and result.stall_time == 0


def test_live_scenario_runner_calls_external_command():
    calls = []
    runner = LiveScenarioRunner(
        [ScenarioEvent(at=0, action="network", worker="p2", bw=1000, command="shape {worker} {bw}"),
         ScenarioEvent(at=50, action="drop_worker", worker="p3", command="kill {worker}")],
        runner=lambda cmd: calls.append(cmd) or 0,
    )
    runner.poll(now=runner.started + 1)
    assert calls == ["shape p2 1000"]
    runner.poll(now=runner.started + 60)
    assert calls[-1] == "kill p3"
    with pytest.raises(ValueError):
        ScenarioEvent.from_mapping({"at": 1, "action": "explode"})
