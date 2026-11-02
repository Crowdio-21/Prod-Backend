"""The simulator against the real Foreman: small worlds, then a full 15-device world."""

import asyncio
import json
from pathlib import Path

import pytest

from evaluation.devsim.device import find_hint, find_model
from evaluation.devsim.runner import SimConfig, format_summary, run_once
from evaluation.devsim.workloads import make_model_size, payload, sim_task


def run(coro):
    return asyncio.run(coro)


def cfg(tmp_path, **kw):
    base = dict(scenario="stable", devices=4, scheduler="fifo", workload="uniform", tasks=8, seed=1, speed=4.0,
                out_dir=str(tmp_path / "out"), work_scale=0.1, timeout_s=120)
    base.update(kw)
    return SimConfig(**base)


# ------------------------------------------------------------------ helpers that need no Foreman
def test_hints_and_model_names_are_found_where_the_real_flow_puts_them():
    hint = {"sim": {"work_s": 2.0, "out_bytes": 10}}
    assert find_hint([hint]) == hint["sim"]  # a map task: args are [payload]
    assert find_hint({"original_args": hint, "upstream_results": {}}) == hint["sim"]  # a pipeline stage
    assert find_hint([{"no": "hint"}]) == {} and find_hint("text") == {}
    assert find_model([payload(100, 1, 1, "cell_a.onnx")]) == "cell_a.onnx"
    assert find_model({"original_args": {"model_path": "C:\\models\\cell_b.onnx"}}) == "cell_b.onnx"
    assert find_model([payload(100, 1, 1)]) is None


def test_the_shipped_task_function_honours_its_hint():
    result = sim_task({"sim": {"work_s": 0.0, "out_bytes": 5}})
    assert result == {"sim": True, "pad": "xxxxx"}
    assert sim_task({"original_args": {"sim": {"out_bytes": 2}}})["pad"] == "xx"


def test_model_sizes_are_scaled():
    full, tiny = make_model_size(1.0), make_model_size(0.01)
    assert full("cell_a.onnx") >= 100 * tiny("cell_a.onnx") - 100 and tiny("cell_a.onnx") > 0
    assert full("not-a-model.onnx") is None


def test_invalid_configurations_are_rejected(tmp_path):
    for bad in ({"scenario": "nope"}, {"workload": "nope"}, {"mode": "real", "speed": 2.0}, {"speed": 0}):
        with pytest.raises(ValueError):
            cfg(tmp_path, **bad).validate()


# ------------------------------------------------------------------ small worlds
def test_a_small_fleet_completes_a_job_through_the_real_foreman(tmp_path):
    summary = run(run_once(cfg(tmp_path)))
    assert summary["ok"] and summary["tasks_run"] == 8 and summary["tasks_expected"] == 8
    assert sum(d["tasks"] for d in summary["devices"]) == 8
    out = Path(tmp_path / "out")
    assert {p.name for p in out.iterdir()} >= {"summary.json", "devices.csv", "bandwidth_trace.csv", "events.jsonl", "run_logs"}
    assert json.loads((out / "summary.json").read_text())["ok"]
    assert "FAILED" not in format_summary(summary) and "4 devices" in format_summary(summary)


def test_the_run_is_recorded_in_the_foremans_own_run_log(tmp_path):
    run(run_once(cfg(tmp_path)))
    records = [json.loads(line) for line in (tmp_path / "out" / "run_logs" / "runs.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["tasks"]["completed"] == 8 and records[0]["scheduler"] == "FIFOScheduler"
    assert {w["kind"] for w in records[0]["workers"]} == {"mobile"}  # the simulated phones are classified as phones


def test_simulated_devices_register_like_the_android_app(tmp_path):
    from foreman.core.connection_manager import WorkerInfo
    from evaluation.devsim.device import VirtualDevice
    from evaluation.devsim.fleet import make_fleet
    from evaluation.devsim.clock import VirtualClock
    from evaluation.devsim.link import LinkEngine

    phone, pc = make_fleet(2, pc=1)[1], make_fleet(2, pc=1)[0]
    dev = lambda spec: VirtualDevice(spec, "ws://x", VirtualClock(), LinkEngine(VirtualClock(), {}, {}))  # noqa: E731
    kinds = {s.kind: WorkerInfo.from_worker_ready_message(s.device_id, None, dev(s)._registration()).worker_type.value
             for s in (phone, pc)}
    assert kinds == {"phone": "android_kotlin", "pc": "pc_python"}


def test_pcs_are_not_given_dnn_work_under_the_default_mobile_rule(tmp_path):
    summary = run(run_once(cfg(tmp_path, devices=4, pc=2, workload="dnn3", tasks=4, work_scale=0.05, speed=4.0)))
    assert summary["ok"]
    by_kind = {k: sum(d["tasks"] for d in summary["devices"] if d["kind"] == k) for k in ("pc", "phone")}
    assert by_kind["pc"] == 0 and by_kind["phone"] == 12  # 4 inputs x 3 stages, all on phones


def test_a_pipeline_downloads_each_model_once_per_device(tmp_path):
    summary = run(run_once(cfg(tmp_path, devices=3, workload="dnn3", tasks=6, work_scale=0.05)))
    assert summary["ok"] and summary["tasks_run"] == 18
    assert all(d["model_downloads"] <= 3 for d in summary["devices"])  # three model files at most
    assert sum(d["model_downloads"] for d in summary["devices"]) >= 3


def test_devices_that_leave_and_return_do_not_lose_the_job(tmp_path):
    from evaluation.devsim import scenarios as sc

    original = sc.SCENARIOS["churn"]
    sc.SCENARIOS["churn"] = sc.Scenario(
        "churn", "test", original.models,
        lambda fleet, rng: [sc.ScenarioEvent(1.0, "leave", fleet[0].device_id),
                            sc.ScenarioEvent(5.0, "join", fleet[0].device_id)])
    try:
        summary = run(run_once(cfg(tmp_path, scenario="churn", tasks=16, work_scale=0.3)))
    finally:
        sc.SCENARIOS["churn"] = original
    assert summary["ok"] and summary["tasks_run"] >= 16
    left = summary["devices"][0]
    assert left["connects"] == 2  # it really dropped and reconnected


# ------------------------------------------------------------------ bandwidth changes are visible to the scheduler
def test_the_bandwidth_aware_scheduler_sees_a_drop_and_moves_work_away(tmp_path):
    """When the two fastest links collapse from the start, the data volume shifts to the other phones."""
    from evaluation.devsim import scenarios as sc

    original = sc.SCENARIOS["drop"]
    sc.SCENARIOS["drop"] = sc.Scenario("drop", "test", lambda fleet, rng: {
        d.device_id: sc.Step(at_s=0.0, factor=0.02) for d in sorted(fleet, key=lambda d: -d.down_mbps)[:2]})
    try:
        summary = run(run_once(cfg(tmp_path, scenario="drop", scheduler="bandwidth_aware", devices=6, tasks=48,
                                   workload="mixed", work_scale=0.4, speed=2.0, heartbeat_s=1.0)))
    finally:
        sc.SCENARIOS["drop"] = original
    assert summary["ok"]
    ranked = sorted(summary["devices"], key=lambda d: -d["down_mbps"])
    collapsed, healthy = ranked[:2], ranked[2:]
    # a collapsed link may still take small tasks, but it should be sent far fewer bytes
    assert sum(d["mb_down"] for d in collapsed) < sum(d["mb_down"] for d in healthy) / 4


# ------------------------------------------------------------------ the full 15-device world
def test_fifteen_devices_with_changing_bandwidth(tmp_path):
    summary = run(run_once(cfg(tmp_path, scenario="congestion", devices=15, scheduler="bandwidth_aware",
                               workload="mixed", tasks=45, work_scale=0.4, speed=3.0)))
    assert summary["ok"] and summary["tasks_run"] == 45
    assert len(summary["devices"]) == 15 and summary["devices_used"] >= 8
    assert summary["makespan_virtual_s"] > 0 and summary["link_estimate_error_pct"] is not None
    rows = (tmp_path / "out" / "bandwidth_trace.csv").read_text().splitlines()
    assert rows[0] == "device,t_s,down_mbps,up_mbps,factor" and len(rows) > 15 * 5
    factors = {row.split(",")[0]: [] for row in rows[1:]}
    for row in rows[1:]:
        factors[row.split(",")[0]].append(float(row.split(",")[4]))
    assert any(max(v) - min(v) > 0.3 for v in factors.values())  # bandwidth really changed over the run
