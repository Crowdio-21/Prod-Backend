import asyncio
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from foreman.core import payload_store
from foreman.core.scheduling import BandwidthAwareScheduler, FIFOScheduler, LinkEstimator, create_scheduler
from foreman.core.scheduling.bandwidth_scheduler import fair_level
from foreman.core.scheduling.scheduler_interface import Task
from foreman.core.worker_metrics import extract_link_report
from pc_worker.core.link_probe import LinkProbe, foreman_http_base

MB = 1_000_000.0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def task(task_id, size=100_000, model=None, code=""):
    args = json.dumps({"original_args": {"model_path": model} if model else {}, "data": "x" * size})
    return Task(id=task_id, job_id="j", args=args, stage_func_code=code or None)


def scheduler(rates, rtt=0.01, **options):
    """rates: worker -> (down MB/s, up MB/s). Link estimates are primed as if the workers had reported them."""
    clock = options.pop("clock", Clock())
    sched = BandwidthAwareScheduler(clock=clock, model_size=options.pop("model_size", lambda name: 0), **options)
    for worker, (down, up) in rates.items():
        sched.observe_link_report(worker, {"down_bps": down * MB, "up_bps": up * MB, "rtt_s": rtt})
    return sched


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ estimator
def test_estimator_averages_time_not_rate():
    """One lucky fast sample must not make a slow link look fast (harmonic-style EWMA)."""
    est = LinkEstimator(clock=lambda: 0.0)
    for _ in range(5):
        est.observe_rate("w", "down", 1 * MB)
    est.observe_rate("w", "down", 100 * MB)
    arithmetic = 0.3 * 100 * MB + 0.7 * 1 * MB
    assert est.estimate("w").down_bps < 2 * MB < arithmetic


def test_estimator_ignores_tiny_and_implausible_samples():
    est = LinkEstimator(clock=lambda: 0.0)
    assert not est.observe_transfer("w", "down", 500, 0.01)  # RTT-dominated, says nothing about bandwidth
    assert not est.observe_rate("w", "down", 1.0) and not est.observe_rate("w", "sideways", 1 * MB)
    assert not est.observe_rtt("w", 0.0) and est.estimate("w").imputed
    assert est.observe_transfer("w", "down", 1_000_000, 0.5)
    assert est.estimate("w").down_bps == pytest.approx(2 * MB)


def test_unknown_workers_look_average_and_stale_ones_drift_back():
    clock = Clock()
    est = LinkEstimator(clock=clock, max_age_s=100, prior_bps=5 * MB)
    assert est.estimate("new").down_bps == pytest.approx(5 * MB) and est.estimate("new").imputed  # prior
    est.observe_rate("fast", "down", 40 * MB)
    est.observe_rate("slow", "down", 2 * MB)
    new = est.estimate("new")
    assert new.imputed and 2 * MB < new.down_bps < 40 * MB  # fleet median of seconds/byte
    fresh = est.estimate("slow").down_bps
    clock.now += 150  # halfway through the drift window
    assert fresh < est.estimate("slow").down_bps < est.estimate("new").down_bps
    clock.now += 500
    assert est.estimate("slow").down_bps == pytest.approx(est.estimate("new").down_bps)


def test_a_direction_nobody_measured_copies_the_other():
    est = LinkEstimator(clock=lambda: 0.0)
    est.observe_report("w", {"down_bps": 8 * MB})
    e = est.estimate("w")
    assert e.up_bps == pytest.approx(8 * MB) and e.n_up == 0 and not e.imputed


def test_compute_estimate_fallback_order():
    est = LinkEstimator(clock=lambda: 0.0)
    assert est.compute_seconds("a", "k", default=1.0) == 1.0
    est.observe_compute("a", "k", 2.0)
    est.observe_compute("b", "other", 4.0)
    assert est.compute_seconds("a", "k", 1.0) == 2.0  # this worker, this key
    assert est.compute_seconds("c", "k", 1.0) == 2.0  # fleet, same key
    assert est.compute_seconds("b", "never", 1.0) == 4.0  # the worker's own average
    assert est.compute_seconds("c", "never", 1.0) == pytest.approx(3.0)  # fleet average


# ------------------------------------------------------------------ choosing workers
def test_large_payload_goes_to_the_fast_link():
    sched = scheduler({"slow": (1, 1), "fast": (20, 20)})
    chosen = run(sched.select_worker(task("t", size=5_000_000), {"slow", "fast"}, {}))
    assert chosen == "fast"


def test_compute_can_outweigh_bandwidth_for_small_payloads():
    sched = scheduler({"fast_link": (50, 50), "fast_cpu": (5, 5)})
    sched.estimator.observe_compute("fast_link", "default", 5.0)
    sched.estimator.observe_compute("fast_cpu", "default", 0.5)
    assert run(sched.select_worker(task("t", size=1000), {"fast_link", "fast_cpu"}, {})) == "fast_cpu"


def test_fewer_tasks_than_workers_use_only_the_best_links():
    sched = scheduler({"a": (1, 1), "b": (30, 30), "c": (10, 10), "d": (0.5, 0.5)})
    pairs = run(sched.batch_select_workers([task("t1", 2_000_000), task("t2", 2_000_000)], {"a", "b", "c", "d"}, {}))
    assert {w for _, w in pairs} == {"b", "c"}


def test_biggest_payload_gets_the_fastest_link():
    """Sufferage: the task that loses most from a slow link chooses first."""
    sched = scheduler({"fast": (40, 40), "slow": (2, 2)})
    pairs = dict((t.id, w) for t, w in run(sched.batch_select_workers(
        [task("small", 10_000), task("big", 4_000_000)], {"fast", "slow"}, {})))
    assert pairs == {"big": "fast", "small": "slow"}


def test_batch_assigns_at_most_one_task_per_worker_and_keeps_order_stable():
    sched = scheduler({"a": (10, 10), "b": (10, 10)})
    tasks = [task(f"t{i}", 50_000) for i in range(10)]
    pairs = run(sched.batch_select_workers(tasks, {"a", "b"}, {}))
    assert len(pairs) == 2 and len({w for _, w in pairs}) == 2
    assert [t.id for t, _ in pairs] == ["t0", "t1"]  # equal costs: first come, first served


def test_without_any_measurements_it_behaves_deterministically():
    sched = BandwidthAwareScheduler(clock=Clock(), model_size=lambda n: 0)
    first = run(sched.batch_select_workers([task("t1"), task("t2")], {"b", "a"}, {}))
    again = BandwidthAwareScheduler(clock=Clock(), model_size=lambda n: 0)
    assert [(t.id, w) for t, w in first] == [(t.id, w) for t, w in run(again.batch_select_workers([task("t1"), task("t2")], {"b", "a"}, {}))]
    assert run(sched.batch_select_workers([], {"a"}, {})) == [] and run(sched.batch_select_workers([task("x")], set(), {})) == []


def test_ties_prefer_the_more_reliable_worker():
    sched = scheduler({"flaky": (10, 10), "solid": (10, 10)})
    workers = {"flaky": SimpleNamespace(success_rate=0.5), "solid": SimpleNamespace(success_rate=1.0)}
    assert run(sched.select_worker(task("t"), {"flaky", "solid"}, workers)) == "solid"


# ------------------------------------------------------------------ contention
def test_fair_level_is_max_min_fair():
    assert fair_level([10, 10], 10) == pytest.approx(5)
    assert fair_level([2, 10], 10) == pytest.approx(8)
    assert fair_level([3, 3], 10) == math.inf
    assert fair_level([], 10) == math.inf


def test_shared_medium_stretches_concurrent_transfers():
    rates = {"a": (10, 10), "b": (10, 10)}
    tasks = [task("t1", 3_000_000), task("t2", 3_000_000)]
    free = scheduler(rates, default_compute_s=0.01)
    shared = scheduler(rates, default_compute_s=0.01, shared_capacity_bps=10 * MB)  # both share one 10 MB/s medium
    run(free.batch_select_workers(tasks, {"a", "b"}, {}))
    run(shared.batch_select_workers(tasks, {"a", "b"}, {}))
    free_s = max(d["estimated_s"] for d in free.decisions)
    shared_s = max(d["estimated_s"] for d in shared.decisions)
    assert shared_s == pytest.approx(1.22, abs=0.01) and free_s == pytest.approx(0.62, abs=0.01)  # 5 MB/s per flow, not 10


# ------------------------------------------------------------------ models and payloads
def test_model_download_is_paid_once_per_worker():
    sizes = {"cell_a.onnx": 180_000_000}
    sched = scheduler({"has": (10, 10), "lacks": (10, 10)}, model_size=sizes.get)
    # Both links are equal; the worker that already holds the model is cheaper.
    run(sched.select_worker(task("warm-up", model="cell_a.onnx"), {"has"}, {}))
    chosen = run(sched.select_worker(task("next", model="cell_a.onnx"), {"has", "lacks"}, {}))
    assert chosen == "has"
    profile = sched.profile(task("p", model="cell_a.onnx"))
    est = sched.estimator.estimates(["has", "lacks"], 0)
    assert sched.cost(profile, "lacks", est["lacks"]) - sched.cost(profile, "has", est["has"]) == pytest.approx(18.0, rel=0.01)


def test_payload_refs_are_sized_by_their_uncompressed_length(tmp_path, monkeypatch):
    monkeypatch.setattr(payload_store, "_ROOT_DIR", str(tmp_path))
    text = json.dumps({"original_args": {"model_path": "cell_b.onnx"}, "upstream_results": {"a": "y" * 600_000}})
    ref = payload_store.store_text_if_large(text, "args", "task1", max_inline_bytes=1000)
    assert payload_store.is_payload_ref(ref)
    sched = scheduler({"w": (10, 10)}, model_size=lambda name: 0)
    profile = sched.profile(Task(id="t", job_id="j", args=ref))
    assert profile.args_bytes == len(text.encode()) and profile.model == "cell_b.onnx"
    assert profile.args_bytes > len(ref)  # not the tiny reference string


# ------------------------------------------------------------------ learning
def test_completions_teach_compute_time_and_result_size():
    clock = Clock()
    sched = scheduler({"w": (10, 10)}, clock=clock)
    t = task("t", 100_000)
    run(sched.select_worker(t, {"w"}, {}))
    clock.now += 3.0
    sched.observe_completion("t", "w", result_bytes=40_000)
    key = sched.profile(t).key
    # 3 s turnaround minus ~0.01 s rtt and ~0.014 s of transfer is compute
    assert sched.estimator.compute_seconds("w", key, 99) == pytest.approx(3.0, abs=0.1)
    assert sched.profile(task("t2", 100_000)).out_bytes == 40_000
    sched.observe_completion("t", "w", 1)  # unknown or repeated completions are ignored
    sched.observe_completion("ghost", "w", 1)


def test_select_task_prefers_the_task_this_worker_is_best_for():
    sched = scheduler({"big_pipe": (50, 50), "small_pipe": (1, 1)})
    pending = [task("heavy", 8_000_000), task("light", 20_000)]
    assert run(sched.select_task(pending, "small_pipe")).id == "light"  # least regret on a slow link
    assert run(sched.select_task(pending, "big_pipe")).id == "heavy"
    assert run(sched.select_task([], "big_pipe")) is None


def test_monitoring_snapshot_and_noop_hooks():
    sched = scheduler({"w": (8, 4)})
    run(sched.select_worker(task("t"), {"w"}, {}))
    snap = sched.link_snapshot()
    assert snap["workers"]["w"]["down_mbps"] == pytest.approx(64.0, rel=0.01)
    assert snap["workers"]["w"]["up_mbps"] == pytest.approx(32.0, rel=0.01)
    assert snap["recent_decisions"][0]["worker_id"] == "w"
    fifo = FIFOScheduler()
    assert fifo.link_snapshot() is None
    fifo.observe_link_report("w", {"down_bps": 1.0})  # no-ops, must not raise
    fifo.observe_completion("t", "w", 5)


# ------------------------------------------------------------------ feeding it
def test_link_reports_from_each_kind_of_worker():
    assert extract_link_report({"link": {"down_bps": 5e6, "up_bps": 2e6, "rtt_s": 0.02}}) == \
        {"down_bps": 5e6, "up_bps": 2e6, "rtt_s": 0.02}
    # Android reports bits per second
    assert extract_link_report({"net": {"bw_down": 80e6, "bw_up": 16e6, "rssi": -50}}) == {"down_bps": 10e6, "up_bps": 2e6}
    # PC "network_speed_mbps" is NIC utilisation, not bandwidth: never used
    assert extract_link_report({"network_speed_mbps": 900.0, "performance_metrics": {"network_speed_mbps": 900}}) == {}
    assert extract_link_report({"net": {"bw_down": None}}) == {} and extract_link_report("junk") == {}


def test_link_probe_measures_rate_and_rtt_and_is_rate_limited():
    clock = Clock()

    def get(url, timeout):
        size = int(url.rsplit("/", 1)[1])
        clock.now += 0.05 + size / (4 * MB)  # 50 ms RTT, 4 MB/s down
        return size

    def post(url, body, timeout):
        clock.now += 0.05 + len(body) / (1 * MB)  # 1 MB/s up

    probe = LinkProbe("http://foreman:8000/", get=get, post=post, clock=clock, min_interval_s=60)
    assert probe.due() and probe.last() is None
    report = probe.measure()
    assert report["rtt_s"] == pytest.approx(0.05 + 1 / (4 * MB), rel=0.05)
    assert report["down_bps"] == pytest.approx(4 * MB, rel=0.02) and report["up_bps"] == pytest.approx(1 * MB, rel=0.02)
    assert not probe.due()
    clock.now += 61
    assert probe.due() and probe.last()["age_s"] == pytest.approx(61, abs=1)


def test_link_probe_never_reports_an_absurd_rate_when_the_transfer_is_as_fast_as_the_round_trip():
    clock = Clock()
    calls = []

    def get(url, timeout):
        size = int(url.rsplit("/", 1)[1])
        calls.append(size)
        clock.now += 0.100 if len(calls) == 1 else 0.004  # first call pays one-off costs, later ones are fast
        return size

    def post(url, body, timeout):
        clock.now += 0.004

    report = LinkProbe("http://x", get=get, post=post, clock=clock).measure()
    assert report["rtt_s"] == pytest.approx(0.004)  # the fastest of three, not the cold first request
    # a 256 KB download in 4 ms (equal to the RTT) is bounded by 20% of the time, not by a 0.1 ms floor
    assert report["down_bps"] <= 262_144 / (0.2 * 0.004) * 1.01
    assert report["down_bps"] < 1e9


def test_link_probe_keeps_the_last_value_when_the_foreman_is_unreachable():
    clock = Clock()
    ok = LinkProbe("http://x", get=lambda u, t: int(u.rsplit("/", 1)[1]), post=lambda *a: None, clock=clock)
    good = ok.measure()

    def down(*args):
        raise OSError("unreachable")

    ok._get = down
    assert ok.measure() == good


def test_foreman_http_address_comes_from_the_websocket_url(monkeypatch):
    monkeypatch.delenv("CROWDIO_FOREMAN_HTTP", raising=False)
    assert foreman_http_base("ws://192.168.1.7:9000") == "http://192.168.1.7:8000"
    monkeypatch.setenv("CROWDIO_FOREMAN_HTTP", "http://proxy:8123/")
    assert foreman_http_base("ws://192.168.1.7:9000") == "http://proxy:8123"


# ------------------------------------------------------------------ wiring
def test_factory_and_environment(monkeypatch):
    monkeypatch.delenv("CROWDIO_SHARED_CAPACITY_MBPS", raising=False)
    assert isinstance(create_scheduler("bandwidth_aware"), BandwidthAwareScheduler)
    assert create_scheduler("bandwidth_aware").shared_capacity_bps is None
    monkeypatch.setenv("CROWDIO_SHARED_CAPACITY_MBPS", "80")
    assert create_scheduler("bandwidth_aware").shared_capacity_bps == pytest.approx(10 * MB)


def test_websocket_manager_picks_the_scheduler_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from foreman.core.ws_manager import WebSocketManager

    monkeypatch.setenv("CROWDIO_SCHEDULER", "bandwidth_aware")
    assert isinstance(WebSocketManager().scheduler, BandwidthAwareScheduler)
    monkeypatch.delenv("CROWDIO_SCHEDULER")
    assert isinstance(WebSocketManager().scheduler, FIFOScheduler)
    assert isinstance(WebSocketManager(scheduler_type="priority").scheduler.__class__(), type(WebSocketManager(scheduler_type="priority").scheduler))


def test_links_endpoint(monkeypatch):
    import foreman.main as main

    async def get(path):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as client:
            return (await client.get(path)).json()

    monkeypatch.setattr(main, "ws_manager", SimpleNamespace(task_dispatcher=SimpleNamespace(scheduler=FIFOScheduler())))
    assert run(get("/api/links")) == {"scheduler": "FIFOScheduler", "bandwidth_aware": False}
    sched = scheduler({"w": (8, 4)})
    monkeypatch.setattr(main, "ws_manager", SimpleNamespace(task_dispatcher=SimpleNamespace(scheduler=sched)))
    body = run(get("/api/links"))
    assert body["bandwidth_aware"] and body["workers"]["w"]["down_mbps"] == pytest.approx(64.0, rel=0.01)


# ------------------------------------------------------------------ waiting for a better worker
def _busy_fast_idle_slow(clock, **options):
    """'fast' is running a task that should end in ~0.6 s; only 'slow' is idle."""
    sched = scheduler({"fast": (40, 40), "slow": (1, 1)}, clock=clock, default_compute_s=0.5, **options)
    run(sched.select_worker(task("running", 100_000), {"fast"}, {}))
    return sched


def test_a_big_task_waits_for_the_busy_fast_worker_instead_of_a_slow_idle_one():
    clock = Clock()
    sched = _busy_fast_idle_slow(clock)
    big = task("big", 6_000_000)  # ~12 s on the slow link, ~0.4 s on the fast one
    assert run(sched.batch_select_workers([big], {"slow"}, {})) == []  # deferred, not sent to 'slow'
    clock.now += 0.7
    sched.observe_completion("running", "fast", 10_000)  # fast is free now
    assert [(t.id, w) for t, w in run(sched.batch_select_workers([big], {"fast", "slow"}, {}))] == [("big", "fast")]


def test_it_never_waits_when_nothing_is_running():
    sched = scheduler({"slow": (1, 1)}, default_compute_s=0.5)
    pairs = run(sched.batch_select_workers([task("big", 6_000_000)], {"slow"}, {}))
    assert [(t.id, w) for t, w in pairs] == [("big", "slow")]  # no completion would ever wake it up


def test_waiting_is_bounded_by_aging():
    clock = Clock()
    sched = _busy_fast_idle_slow(clock, max_wait_s=10.0)
    big = task("big", 6_000_000)
    assert run(sched.batch_select_workers([big], {"slow"}, {})) == []
    clock.now += 11  # 'fast' still has not reported back
    assert [(t.id, w) for t, w in run(sched.batch_select_workers([big], {"slow"}, {}))] == [("big", "slow")]


def test_it_does_not_wait_for_a_worker_that_is_offline_or_too_far_away():
    clock = Clock()
    sched = _busy_fast_idle_slow(clock)
    big = task("big", 6_000_000)
    offline = {"fast": SimpleNamespace(status="offline", success_rate=1.0)}
    assert [w for _, w in run(sched.batch_select_workers([big], {"slow"}, offline))] == ["slow"]
    far = _busy_fast_idle_slow(Clock(), max_defer_s=0.1)  # the fast worker is busy for ~0.6 s > 0.1 s
    assert [w for _, w in far_run(far, big)] == ["slow"]


def far_run(sched, big):
    return run(sched.batch_select_workers([big], {"slow"}, {}))


# ------------------------------------------------------------------ against the existing schedulers
def test_simulation_bandwidth_aware_beats_the_existing_schedulers_on_uneven_links():
    from evaluation.scheduler_sim import SCHEDULERS, SimTask, SimWorker, scenarios, simulate

    totals = {name: {"hetero": 0.0, "same": 0.0} for name in SCHEDULERS}
    seeds = range(5)
    for seed in seeds:
        built = scenarios(seed)
        for name, (workers, tasks) in built.items():
            kind = "hetero" if name.startswith("heterogeneous links (0.5") else "same" if name.startswith("identical") else None
            if kind is None:
                continue
            for sched, make in SCHEDULERS.items():
                totals[sched][kind] += simulate(make, workers, tasks)["makespan_s"]
    for other in ("fifo", "round_robin", "least_loaded"):
        assert totals["bandwidth_aware"]["hetero"] < 0.7 * totals[other]["hetero"], other  # at least 30% faster on average
        assert totals["bandwidth_aware"]["same"] <= 1.02 * totals[other]["same"], other  # and no worse when links are equal
