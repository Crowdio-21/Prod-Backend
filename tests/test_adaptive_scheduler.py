import asyncio
import json
from types import SimpleNamespace

from foreman.core.scheduling import AdaptiveScheduler, BandwidthAwareScheduler, create_scheduler
from foreman.core.scheduling.scheduler_interface import Task

MB = 1_000_000.0


def task(task_id, size=100_000):
    return Task(id=task_id, job_id="j", args=json.dumps({"original_args": {}, "data": "x" * size}))


def worker(completed=0, failed=0, battery=100.0, charging=True):
    return SimpleNamespace(status="online", tasks_completed=completed, tasks_failed=failed,
                           battery_level=battery, is_charging=charging, success_rate=1.0)


def primed(cls, rates, **options):
    sched = cls(clock=lambda: 1000.0, model_size=lambda name: 0, **options)
    for name, (down, up) in rates.items():
        sched.observe_link_report(name, {"down_bps": down * MB, "up_bps": up * MB, "rtt_s": 0.01})
    return sched


def pick(sched, workers, n=1):
    tasks = [task(f"t{i}") for i in range(n)]
    return asyncio.run(sched.batch_select_workers(tasks, set(workers), workers))


def test_factory_builds_adaptive_and_passes_options():
    assert isinstance(create_scheduler("adaptive"), AdaptiveScheduler)
    assert create_scheduler("adaptive", shared_capacity_bps=1e6).shared_capacity_bps == 1e6


def test_healthy_fleet_schedules_exactly_like_bandwidth_aware():
    rates = {"a": (10, 5), "b": (2, 1), "c": (6, 3)}
    workers = {w: worker(completed=20) for w in rates}
    for n in (1, 3):
        base = pick(primed(BandwidthAwareScheduler, rates), workers, n)
        adaptive = pick(primed(AdaptiveScheduler, rates), workers, n)
        assert [(t.id, w) for t, w in adaptive] == [(t.id, w) for t, w in base]


def test_unreliable_phone_loses_a_close_call():
    rates = {"fast_flaky": (10, 5), "steady": (8, 4)}
    workers = {"fast_flaky": worker(completed=2, failed=8), "steady": worker(completed=10)}
    assert pick(primed(BandwidthAwareScheduler, rates), workers)[0][1] == "fast_flaky"
    assert pick(primed(AdaptiveScheduler, rates), workers)[0][1] == "steady"


def test_reliability_never_exiles_a_phone_with_a_much_better_link():
    rates = {"fast_flaky": (100, 50), "slow": (0.5, 0.25)}
    workers = {"fast_flaky": worker(completed=0, failed=50), "slow": worker(completed=50)}
    assert pick(primed(AdaptiveScheduler, rates, default_compute_s=0.01), workers)[0][1] == "fast_flaky"


def test_new_phone_is_not_penalised_so_it_gets_tried():
    assert AdaptiveScheduler().health(worker())["penalty"] == 1.0


def test_each_failure_raises_the_penalty():
    sched = AdaptiveScheduler()
    penalties = [sched.health(worker(completed=10, failed=f))["penalty"] for f in (0, 1, 3, 10)]
    assert penalties == sorted(penalties) and penalties[0] == 1.0 and penalties[-1] > penalties[1] > 1.0


def test_low_battery_only_counts_when_discharging():
    sched = AdaptiveScheduler()
    assert sched.health(worker(battery=5, charging=True))["battery_factor"] == 1.0
    assert sched.health(worker(battery=15, charging=False))["battery_factor"] == sched.low_battery_penalty
    assert sched.health(worker(battery=5, charging=False))["battery_factor"] == sched.critical_battery_penalty
    assert sched.health(worker(battery=60, charging=False))["battery_factor"] == 1.0


def test_pc_without_a_battery_is_not_penalised():
    assert AdaptiveScheduler().health(worker(battery=None, charging=False))["battery_factor"] == 1.0


def test_low_battery_phone_yields_to_a_slightly_slower_one():
    rates = {"drained": (10, 5), "ok": (8, 4)}
    workers = {"drained": worker(completed=10, battery=8, charging=False), "ok": worker(completed=10)}
    assert pick(primed(AdaptiveScheduler, rates), workers)[0][1] == "ok"


def test_decisions_and_snapshot_explain_the_penalty():
    rates = {"a": (10, 5), "b": (8, 4)}
    workers = {"a": worker(completed=1, failed=5), "b": worker(completed=10)}
    sched = primed(AdaptiveScheduler, rates)
    pick(sched, workers)
    assert sched.decisions[-1]["health_penalty"] == 1.0  # the healthy phone was chosen
    health = sched.link_snapshot()["health"]
    assert health["a"]["penalty"] > health["b"]["penalty"] == 1.0
    assert set(health["a"]) >= {"reliability", "battery_factor", "penalty"}
