"""Run the real scheduler classes on a :class:`~evaluation.netsim.model.World`.

What is simulated and what is real:

* Real: the scheduler classes (FIFO, round-robin, least-loaded, bandwidth-aware, RSSI-aware,
  adaptive). They are driven through ``batch_select_workers`` / ``observe_*`` exactly as the Foreman
  does, with the clock injected.
* Simulated: the devices. A task is ``rtt/2 + download + compute + upload + rtt/2``, where each
  transfer is integrated over the device's *time-varying* achievable rate. Workers take one task at
  a time. There is no dispatch bottleneck (the real Foreman's ~6-7 tasks/s WebSocket path is
  bypassed on purpose), so differences come from placement alone.
* What the scheduler *knows* is deliberately limited, as on real phones:
  - ``bw_report`` = ``"heartbeat:<s>"``: every ``s`` seconds each device reports its current
    achievable rate (8% measurement noise); ``"transfers"``: a device reports only the rate
    measured on a large transfer (>= 256 KB) when its task finishes, as the Android app does.
  - RSSI is reported every ``radio_period_s`` (whole dB, noise ``rssi_noise_db``).
  - The ``oracle`` is told the true current rate of every device at decision time (an upper bound).
* The scheduler is told each task's true result size (the real system estimates it), the same for
  every bandwidth-aware scheduler.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from foreman.core.scheduling import (
    AdaptiveScheduler,
    BandwidthAwareScheduler,
    FIFOScheduler,
    LeastLoadedScheduler,
    LinkEstimator,
    RoundRobinScheduler,
    RssiAwareScheduler,
    RssiCurve,
    fit_rssi_curve,
)
from foreman.core.scheduling.bandwidth_scheduler import TaskProfile
from foreman.core.scheduling.link_estimator import LinkEstimate
from foreman.core.scheduling.scheduler_interface import Task, TaskScheduler, Worker

from .model import MBIT, NetParams, World, generate_world

MEASURE_MIN_BYTES = 256_000
REPORT_NOISE = 0.08
FRAMING = 512

SCHEDULER_NAMES = ("fifo", "round_robin", "least_loaded", "bandwidth_aware", "adaptive",
                   "rssi_aware", "rssi_aware_cal", "oracle")


@dataclass(frozen=True)
class Workload:
    tasks_per_device: float = 8.0
    tc_ratio: float = 0.5  # transfer time / compute time of a median task on a median device
    compute_s: float = 1.0  # compute time of a median task on a reference device
    size_sigma: float = 0.5  # lognormal spread of payload sizes
    out_ratio: float = 0.2  # result bytes / payload bytes
    payload_bytes: Optional[float] = None  # fixed median payload (overrides tc_ratio)
    sched_window: int = 24  # tasks the bandwidth-based schedulers map per call (real default 64); cost is quadratic in it


@dataclass(frozen=True)
class Reporting:
    bw_report: str = "heartbeat:10"  # "heartbeat:<s>" or "transfers"
    radio_period_s: float = 5.0


@dataclass(frozen=True)
class SimTask:
    task_id: str
    bytes_in: float
    bytes_out: float
    work_s: float


@dataclass
class RunResult:
    scheduler: str
    makespan_s: float
    est_error: float  # mean |ln(estimated downlink / true downlink)| at assignment; nan if no estimate
    mean_task_s: float
    imputed_share: float  # share of assignments made on an unmeasured (imputed) link; nan if n/a


# ------------------------------------------------------------------ tasks
def make_tasks(world: World, workload: Workload, seed: int) -> List[SimTask]:
    """Task sizes are set from the *realised* median downlink so the transfer/compute ratio is what is asked for.

    ``tc_ratio`` alone: payload = ratio x compute_s x median rate. With ``payload_bytes`` too: the
    payload is fixed and the compute time is derived, so the ratio stays put while the payload moves
    from RTT-dominated (20 KB) to bandwidth-dominated (tens of MB).
    """
    rng = np.random.default_rng([seed, 15485863])
    n = max(1, int(round(workload.tasks_per_device * len(world.devices))))
    ref = world.reference_rate()
    if workload.payload_bytes is not None:
        median_payload, compute = workload.payload_bytes, workload.payload_bytes / (workload.tc_ratio * ref)
    else:
        median_payload, compute = workload.tc_ratio * workload.compute_s * ref, workload.compute_s
    sizes = median_payload * np.exp(rng.normal(0.0, workload.size_sigma, n))
    return [SimTask(f"t{i:04d}", float(max(1000.0, s)), float(max(100.0, s * workload.out_ratio)), compute)
            for i, s in enumerate(sizes)]


def true_task_seconds(world: World, device_id: str, task: SimTask, t0: float) -> Tuple[float, Dict[str, float]]:
    """(finish time, measured rates) of ``task`` started on ``device_id`` at ``t0``."""
    d = world.by_id[device_id]
    t = t0 + d.rtt_s / 2.0
    t_down = d.transfer_end("down", t, task.bytes_in)
    measured: Dict[str, float] = {}
    if task.bytes_in >= MEASURE_MIN_BYTES and t_down > t:
        measured["down_bps"] = task.bytes_in / (t_down - t)
    t = t_down + task.work_s * d.cpu_factor
    t_up = d.transfer_end("up", t, task.bytes_out)
    if task.bytes_out >= MEASURE_MIN_BYTES and t_up > t:
        measured["up_bps"] = task.bytes_out / (t_up - t)
    return t_up + d.rtt_s / 2.0, measured


# ------------------------------------------------------------------ schedulers
class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class TruthEstimator(LinkEstimator):
    """Oracle: the true current link of every device, whatever was reported."""

    def __init__(self, world: World, clock: _Clock) -> None:
        super().__init__(clock=clock)
        self.world, self._clock = world, clock

    def _truth(self, worker_id: str, now: float) -> LinkEstimate:
        d = self.world.by_id[worker_id]
        return LinkEstimate(d.rate("down", now), d.rate("up", now), d.rtt_s, 99, 99, 0.0, False)

    def estimates(self, worker_ids, now=None):
        now = self._clock.now if now is None else now
        return {w: self._truth(w, now) for w in worker_ids}

    def estimate(self, worker_id, now=None):
        return self.estimates([worker_id], now)[worker_id]


def make_scheduler(name: str, world: World, clock: _Clock, curve: Optional[RssiCurve] = None,
                   window: int = 24) -> TaskScheduler:
    # The mapping step is quadratic in the window, so it is fixed (24 = twice the base fleet) unless a condition says otherwise.
    common = dict(clock=clock, default_compute_s=1.0, max_wait_s=30.0, max_window=window, model_size=lambda _n: 0)
    if name == "fifo":
        return FIFOScheduler()
    if name == "round_robin":
        return RoundRobinScheduler()
    if name == "least_loaded":
        return LeastLoadedScheduler()
    if name == "bandwidth_aware":
        return BandwidthAwareScheduler(**common)
    if name == "adaptive":
        return AdaptiveScheduler(**common)
    if name == "rssi_aware":
        return RssiAwareScheduler(**common)
    if name == "rssi_aware_cal":
        return RssiAwareScheduler(curve=curve, **common)
    if name == "oracle":
        return BandwidthAwareScheduler(estimator=TruthEstimator(world, clock), **common)
    raise ValueError(f"unknown scheduler {name!r}")


def calibrate_curve(params: NetParams, seeds: Sequence[int], stride: int = 5) -> RssiCurve:
    """Fit the RSSI-throughput curve on held-out worlds from ``params`` (as one would from a real run's
    heartbeat log), then keep it fixed for every test condition."""
    pairs: List[Tuple[float, float]] = []
    for seed in seeds:
        world = generate_world(params, seed)
        for d in world.devices:
            pairs += list(zip(d.rssi[::stride].round(), d.down[::stride] / MBIT))
    return fit_rssi_curve(pairs, min_samples=20)


# ------------------------------------------------------------------ one run
def _estimated_down(sched: TaskScheduler, worker_id: str, now: float) -> Optional[Tuple[float, bool]]:
    estimator = getattr(sched, "estimator", None)
    if estimator is None:
        return None
    est = estimator.estimate(worker_id, now)
    if isinstance(sched, RssiAwareScheduler):
        est = sched.adjust(worker_id, est)[0]
    return est.down_bps, est.imputed


async def _run(name: str, world: World, tasks: List[SimTask], rep: Reporting, curve: Optional[RssiCurve],
               seed: int, window: int = 24) -> RunResult:
    clock = _Clock()
    sched = make_scheduler(name, world, clock, curve, window)
    by_id = {t.task_id: t for t in tasks}
    if isinstance(sched, BandwidthAwareScheduler):
        sched.profile = lambda task: TaskProfile(  # type: ignore[method-assign]
            task.id, int(by_id[task.id].bytes_in) + FRAMING, int(by_id[task.id].bytes_in),
            float(by_id[task.id].bytes_out), None, 0, "work")
    workers = {d.device_id: Worker(id=d.device_id, status="online") for d in world.devices}
    queue = [Task(id=t.task_id, job_id="sim", args="") for t in tasks]
    idle = set(workers)
    rng = np.random.default_rng([seed, 2654435761])
    events: List[tuple] = []  # (time, order, kind, payload)
    counter = 0

    def push(t, kind, payload):
        nonlocal counter
        counter += 1
        heapq.heappush(events, (t, counter, kind, payload))

    hb_s = None if rep.bw_report == "transfers" else float(rep.bw_report.split(":")[1])
    for d in world.devices:  # phones announce themselves on connect (first heartbeat), then at their own phase
        push(0.0, "radio", d.device_id)
        if hb_s is not None:
            push(0.0, "bw", d.device_id)
    errs: List[float] = []
    imputed: List[bool] = []
    durations: List[float] = []
    remaining = len(tasks)

    def process(t: float, kind: str, payload) -> None:
        nonlocal remaining
        if kind == "finish":
            task_id, wid, measured = payload
            if rep.bw_report == "transfers" and measured:
                sched.observe_link_report(wid, {**measured, "rtt_s": world.by_id[wid].rtt_s})
            sched.observe_completion(task_id, wid, int(by_id[task_id].bytes_out))
            idle.add(wid)
            remaining -= 1
        elif kind == "bw":
            d = world.by_id[payload]
            noise = lambda: math.exp(rng.normal(0.0, REPORT_NOISE))  # noqa: E731
            sched.observe_link_report(payload, {"down_bps": d.rate("down", t) * noise(), "up_bps": d.rate("up", t) * noise(),
                                                "rtt_s": d.rtt_s * noise()})
            push(t + hb_s, "bw", payload)
        elif kind == "radio":
            d = world.by_id[payload]
            rssi = round(d.rssi_at(t) + rng.normal(0.0, world.params.rssi_noise_db))
            sched.observe_radio(payload, {"rssi_dbm": float(max(-100, min(-20, rssi)))})
            push(t + rep.radio_period_s, "radio", payload)

    while remaining:
        # Everything that has happened by now (heartbeats, finished tasks) is known before work is handed out.
        while events and events[0][0] <= clock.now:
            t, _, kind, payload = heapq.heappop(events)
            process(t, kind, payload)
        if not remaining:
            break
        if idle and queue:
            now = clock.now
            for task, wid in await sched.batch_select_workers(queue, set(idle), workers):
                queue = [q for q in queue if q.id != task.id]
                idle.discard(wid)
                est = _estimated_down(sched, wid, now)
                if est is not None:
                    errs.append(abs(math.log(max(est[0], 1.0) / max(world.by_id[wid].rate("down", now), 1.0))))
                    imputed.append(est[1])
                finish, measured = true_task_seconds(world, wid, by_id[task.id], now)
                durations.append(finish - now)
                push(finish, "finish", (task.id, wid, measured))
        if not events:
            raise RuntimeError("deadlock: tasks left but no events")
        t, _, kind, payload = heapq.heappop(events)
        clock.now = max(clock.now, t)
        process(t, kind, payload)
    return RunResult(
        name, clock.now, float(np.mean(errs)) if errs else float("nan"), float(np.mean(durations)),
        float(np.mean(imputed)) if imputed else float("nan"))


def run_scheduler(name: str, world: World, tasks: List[SimTask], rep: Reporting = Reporting(),
                  curve: Optional[RssiCurve] = None, seed: int = 0, window: int = 24) -> RunResult:
    logging.disable(logging.CRITICAL)  # the schedulers log every decision at INFO
    return asyncio.run(_run(name, world, tasks, rep, curve, seed, window))


def run_seed(params: NetParams, workload: Workload, rep: Reporting, seed: int, schedulers: Sequence[str],
             curve: Optional[RssiCurve]) -> List[RunResult]:
    """Every scheduler on the identical world, tasks and report noise (a paired run)."""
    world = generate_world(params, seed)
    tasks = make_tasks(world, workload, seed)
    return [run_scheduler(s, world, tasks, rep, curve, seed, workload.sched_window) for s in schedulers]
