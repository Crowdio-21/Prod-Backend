"""The network model and the scheduler simulator behind the regime sweep."""

import math
from dataclasses import replace

import numpy as np
import pytest

from evaluation.netsim.model import (
    MBIT,
    DeviceTrace,
    NetParams,
    ReplayWorld,
    World,
    correlation_rssi_throughput,
    generate_world,
    link_rate_mbps,
)
from evaluation.netsim.sim import (
    Reporting,
    Workload,
    calibrate_curve,
    make_tasks,
    run_scheduler,
    run_seed,
    true_task_seconds,
)
from evaluation.netsim.sweep import bootstrap_median_ci, build_conditions

SMALL = NetParams(devices=5, horizon_s=600.0)


def trace(down, up=None, dt=1.0, rtt=0.0):
    down = np.asarray(down, dtype=float) * MBIT
    up = down if up is None else np.asarray(up, dtype=float) * MBIT
    return DeviceTrace("d", dt, np.full(len(down), -60.0), down, up, rtt, 1.0)


# ---------------------------------------------------------------- transfers over a varying rate
def test_constant_rate_transfer():
    d = trace([8.0] * 100)  # 8 Mbit/s = 1 MB/s
    assert d.transfer_end("down", 0.0, 5e6) == pytest.approx(5.0)
    assert d.transfer_end("down", 2.5, 5e6) == pytest.approx(7.5)


def test_transfer_integrates_across_a_rate_change():
    d = trace([8.0] * 10 + [0.8] * 100)  # 1 MB/s for 10 s, then 0.1 MB/s
    assert d.transfer_end("down", 8.0, 4e6) == pytest.approx(10.0 + 2e6 / 1e5)


def test_transfer_stalls_through_an_outage_and_resumes():
    d = trace([8.0] * 5 + [0.0] * 10 + [8.0] * 100)  # outage floors at 0.01 Mbit/s
    assert d.transfer_end("down", 0.0, 8e6) == pytest.approx(5 + 10 + 3, abs=0.1)


def test_transfer_past_the_horizon_holds_the_last_rate():
    d = trace([8.0] * 10)
    assert d.transfer_end("down", 0.0, 12e6) == pytest.approx(12.0)
    assert d.transfer_end("down", 50.0, 1e6) == pytest.approx(51.0)


# ---------------------------------------------------------------- the physical model
def test_world_is_reproducible_and_seed_dependent():
    a, b, c = generate_world(SMALL, 3), generate_world(SMALL, 3), generate_world(SMALL, 4)
    assert all(np.array_equal(x.down, y.down) and np.array_equal(x.rssi, y.rssi) for x, y in zip(a.devices, b.devices))
    assert not np.array_equal(a.devices[0].rssi, c.devices[0].rssi)


def test_rssi_falls_with_distance_and_link_rate_rises_with_snr():
    near = generate_world(replace(SMALL, area_m=5.0, speed_mps=0.0, ring=True), 1)
    far = generate_world(replace(SMALL, area_m=60.0, speed_mps=0.0, ring=True), 1)
    assert np.mean([d.rssi.mean() for d in near.devices]) > np.mean([d.rssi.mean() for d in far.devices]) + 15
    rates = link_rate_mbps(SMALL, np.array([0.0, 10.0, 20.0, 30.0, 40.0]))
    assert np.all(np.diff(rates) > 0) and rates[-1] <= SMALL.max_link_mbps


def test_throughput_is_the_minimum_of_link_rate_and_backhaul_cap():
    capped = generate_world(replace(SMALL, backhaul_mbps=2.0, spread=1.0, backhaul_var=0.0), 2)
    assert all(d.down.max() / MBIT <= 2.0 + 1e-9 for d in capped.devices)
    open_ = generate_world(replace(SMALL, backhaul_mbps=math.inf), 2)
    assert np.allclose(open_.devices[0].down / MBIT, np.minimum(open_.devices[0].link_mbps, open_.devices[0].link_mbps))


def test_rssi_is_informative_when_radio_limited_and_blind_when_cap_limited():
    radio = generate_world(replace(SMALL, devices=8, backhaul_mbps=math.inf, area_m=80.0), 5)
    capped = generate_world(replace(SMALL, devices=8, backhaul_mbps=3.0, spread=1.0, backhaul_var=0.0, area_m=15.0), 5)
    assert correlation_rssi_throughput(radio) > 0.6
    assert abs(correlation_rssi_throughput(capped)) < 0.4  # cap flat: RSSI says little


def test_contention_lowers_throughput_without_touching_rssi():
    base = generate_world(replace(SMALL, backhaul_mbps=math.inf), 7)
    busy = generate_world(replace(SMALL, backhaul_mbps=math.inf, contention=0.6), 7)
    assert all(np.array_equal(a.rssi, b.rssi) for a, b in zip(base.devices, busy.devices))
    assert np.mean([d.down.mean() for d in busy.devices]) < 0.7 * np.mean([d.down.mean() for d in base.devices])


def test_events_hit_throughput_but_only_outages_show_in_rssi():
    clean = generate_world(SMALL, 9)
    throttled = generate_world(replace(SMALL, drop_frac=1.0), 9)
    bursts = generate_world(replace(SMALL, burst_per_min=6.0, backhaul_mbps=math.inf), 9)
    assert all(np.array_equal(a.rssi, b.rssi) for a, b in zip(clean.devices, throttled.devices))
    assert any((d.down / MBIT < 1.3).sum() > 20 for d in throttled.devices)
    clean_open = generate_world(replace(SMALL, backhaul_mbps=math.inf), 9)
    assert all(np.array_equal(a.rssi, b.rssi) for a, b in zip(clean_open.devices, bursts.devices))
    assert np.mean([d.down.mean() for d in bursts.devices]) < np.mean([d.down.mean() for d in clean_open.devices])
    outages = generate_world(replace(SMALL, outage_per_min=6.0), 9)
    assert any((d.rssi == -100.0).sum() >= 2 for d in outages.devices)  # outage steps sit at the RSSI floor
    assert all((d.rssi == -100.0).sum() == 0 or (d.down / MBIT <= 0.0101).any() for d in outages.devices)


def test_mobility_speed_changes_how_fast_rssi_decorrelates():
    def lag1(speed):
        w = generate_world(replace(SMALL, devices=6, speed_mps=speed, horizon_s=900.0, fast_sigma_db=0.0), 4)
        return np.mean([np.corrcoef(d.rssi[:-1], d.rssi[1:])[0, 1] for d in w.devices])
    assert lag1(0.0) > lag1(20.0)


def test_replayed_trace_round_trips(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("device,t,rssi_dbm,down_mbps,up_mbps\na,0,-50,10,5\na,10,-70,2,1\nb,0,-60,4,\n", encoding="utf-8")
    world = ReplayWorld.load(path, dt=1.0, horizon_s=20.0)
    a = world.device("a")
    assert a.rate("down", 5.0) == pytest.approx(10 * MBIT) and a.rate("down", 15.0) == pytest.approx(2 * MBIT)
    assert a.rssi_at(15.0) == -70 and world.device("b").rate("up", 1.0) == pytest.approx(2 * MBIT)


# ---------------------------------------------------------------- simulator
def test_task_sizes_follow_the_requested_transfer_ratio():
    world = generate_world(SMALL, 1)
    tasks = make_tasks(world, Workload(tc_ratio=1.0, size_sigma=0.0), 1)
    transfer_s = tasks[0].bytes_in / world.reference_rate()
    assert transfer_s == pytest.approx(tasks[0].work_s * 1.0, rel=1e-6)
    fixed = make_tasks(world, Workload(tc_ratio=0.5, payload_bytes=2e6, size_sigma=0.0), 1)
    assert fixed[0].bytes_in == pytest.approx(2e6) and fixed[0].bytes_in / world.reference_rate() == pytest.approx(
        0.5 * fixed[0].work_s, rel=1e-6)


def test_task_time_adds_up():
    d = trace([8.0] * 200, up=[4.0] * 200, rtt=0.2)
    world = World(SMALL, [d])
    task = make_tasks(world, Workload(), 0)[0].__class__("t", 1e6, 5e5, 2.0)
    finish, measured = true_task_seconds(world, "d", task, 0.0)
    assert finish == pytest.approx(0.1 + 1.0 + 2.0 + 1.0 + 0.1)  # rtt/2 + 1 MB down + compute + 0.5 MB up + rtt/2
    assert measured["down_bps"] == pytest.approx(1e6) and measured["up_bps"] == pytest.approx(5e5)
    small = task.__class__("s", 1e5, 1e5, 0.0)  # under 256 KB each way: RTT-bound, nothing to measure
    assert true_task_seconds(world, "d", small, 0.0)[1] == {}


def test_every_scheduler_completes_and_paired_runs_are_reproducible():
    names = ("fifo", "round_robin", "bandwidth_aware", "rssi_aware", "oracle")
    a = run_seed(SMALL, Workload(tasks_per_device=3.0), Reporting(), 1, names, None)
    b = run_seed(SMALL, Workload(tasks_per_device=3.0), Reporting(), 1, names, None)
    assert [r.makespan_s for r in a if r.scheduler != "fifo"] == [r.makespan_s for r in b if r.scheduler != "fifo"]
    assert all(r.makespan_s > 0 and math.isfinite(r.makespan_s) for r in a)


def test_oracle_has_no_estimate_error_and_unmeasured_links_are_flagged():
    world = generate_world(SMALL, 2)
    tasks = make_tasks(world, Workload(tasks_per_device=3.0), 2)
    oracle = run_scheduler("oracle", world, tasks, seed=2)
    assert oracle.est_error == pytest.approx(0.0, abs=1e-6) and oracle.imputed_share == 0.0
    assert math.isnan(run_scheduler("fifo", world, tasks, seed=2).est_error)
    # Reports only after transfers: the very first assignments are made on imputed links.
    cold = run_scheduler("bandwidth_aware", world, tasks, Reporting(bw_report="transfers"), seed=2)
    warm = run_scheduler("bandwidth_aware", world, tasks, Reporting(bw_report="heartbeat:1"), seed=2)
    assert cold.imputed_share > 0.0 and warm.imputed_share == 0.0


def test_fresher_reports_give_better_estimates_over_a_long_run():
    errors = {"heartbeat:1": [], "heartbeat:60": []}
    for seed in range(4):
        world = generate_world(replace(SMALL, devices=6, horizon_s=1500.0), seed)
        tasks = make_tasks(world, Workload(tc_ratio=1.0, tasks_per_device=16.0), seed)
        for mode in errors:
            errors[mode].append(run_scheduler("bandwidth_aware", world, tasks, Reporting(bw_report=mode), seed=seed).est_error)
    assert np.mean(errors["heartbeat:1"]) < np.mean(errors["heartbeat:60"])


def test_calibration_returns_a_curve_fitted_on_held_out_worlds():
    curve = calibrate_curve(SMALL, seeds=[100, 101])
    assert curve.samples > 100 and curve.db_per_decade > 0


# ---------------------------------------------------------------- statistics and conditions
def test_bootstrap_ci_brackets_the_median_and_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    small = bootstrap_median_ci(rng.normal(0.05, 0.05, 10))
    large = bootstrap_median_ci(rng.normal(0.05, 0.05, 200))
    assert small[1] <= small[0] <= small[2] and (large[2] - large[1]) < (small[2] - small[1])
    assert bootstrap_median_ci([0.0] * 20) == (0.0, 0.0, 0.0)
    assert math.isnan(bootstrap_median_ci([])[0])


def test_conditions_are_unique_and_cover_every_axis():
    conds = build_conditions()
    assert len({c.cond_id for c in conds}) == len(conds)
    axes = {axis for c in conds for axis, _ in c.tags}
    assert {"transfer/compute ratio", "payload size (ratio fixed at 0.5)", "heterogeneity", "mobility",
            "bandwidth report freshness", "fleet size (window 24)", "fleet size (window = fleet)", "cold start (reports only after transfers)", "contention from other devices", "backhaul cap (median)",
            "network events", "tasks per device (queue depth)", "regime map"} <= axes
    assert next(c for c in conds if ("fleet size (window = fleet)", "100") in c.tags).workload.sched_window == 100
    assert sum(1 for c in conds if any(a == "regime map" for a, _ in c.tags)) >= 15
