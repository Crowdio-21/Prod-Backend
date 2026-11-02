"""Every feature-flag combination: parsing, the master switch, and a full fake-worker job."""

import asyncio
import itertools

import pytest

from foreman.core import plan_runtime
from foreman.planning.flags import FeatureFlags
from tests.planning.test_integration_fake_workers import (
    _check_results,
    _drive,
    _heartbeats,
    _network,
    _no_violations,
    submit,
)

NAMES = ("adaptive_planning", "direct_path", "peer_fetch", "prefetch", "problem_reports")
COMBINATIONS = [dict(zip(NAMES, bits)) for bits in itertools.product((False, True), repeat=len(NAMES))]
WITH_PLANNING = [c for c in COMBINATIONS if c["adaptive_planning"]]


def test_there_are_thirty_two_combinations_and_sixteen_with_planning():
    assert len(COMBINATIONS) == 32 and len(WITH_PLANNING) == 16


def test_all_flags_default_off():
    assert not FeatureFlags().any_on and FeatureFlags().to_dict() == dict.fromkeys(NAMES, False)


def test_a_comma_list_and_json_set_the_same_flags(monkeypatch):
    monkeypatch.setenv("CROWDIO_FLAGS", "adaptive_planning, prefetch")
    from_list = FeatureFlags.from_env()
    monkeypatch.setenv("CROWDIO_FLAGS", '{"adaptive_planning": true, "prefetch": true}')
    assert from_list == FeatureFlags.from_env()
    assert from_list.to_dict() == {**dict.fromkeys(NAMES, False), "adaptive_planning": True, "prefetch": True}


def test_an_unknown_flag_is_an_error(monkeypatch):
    monkeypatch.setenv("CROWDIO_FLAGS", "adaptive_planing")
    with pytest.raises(ValueError):
        FeatureFlags.from_env()
    with pytest.raises(ValueError):
        FeatureFlags.from_mapping({"nope": True})


def test_the_environment_replaces_the_run_config_flags(monkeypatch, tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("flags:\n  adaptive_planning: true\n  direct_path: true\n")
    monkeypatch.setenv("CROWDIO_RUN_CONFIG", str(config))
    monkeypatch.delenv("CROWDIO_FLAGS", raising=False)
    assert plan_runtime.load_run_config().flags.direct_path
    monkeypatch.setenv("CROWDIO_FLAGS", "adaptive_planning")
    flags = plan_runtime.load_run_config().flags
    assert flags.adaptive_planning and not flags.direct_path  # replaced, not merged


@pytest.mark.parametrize("flags", [c for c in COMBINATIONS if not c["adaptive_planning"]],
                         ids=lambda c: ",".join(k for k, v in c.items() if v) or "none")
def test_without_adaptive_planning_there_is_no_coordinator(flags, monkeypatch):
    monkeypatch.delenv("CROWDIO_RUN_CONFIG", raising=False)
    monkeypatch.setenv("CROWDIO_FLAGS", ",".join(k for k, v in flags.items() if v) or "")
    assert plan_runtime.build_plan_coordinator(object()) is None


@pytest.mark.parametrize("flags", WITH_PLANNING, ids=lambda c: ",".join(k for k, v in c.items() if v))
def test_a_job_completes_with_every_flag_combination(flags):
    async def scenario():
        net, clock = _network(flags=flags)
        await _heartbeats(net, clock)
        client = await submit(net, 6)
        await _drive(net, clock, client, 6)
        _check_results(client, 6)
        assert net.coordinator.config.flags.to_dict() == flags
        _no_violations(net)
    asyncio.run(scenario())
