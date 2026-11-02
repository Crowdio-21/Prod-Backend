"""Compare schedulers on a simulated job over workers with different links.

Runs the real scheduler classes against a simple event simulation: workers take one
task at a time, a task takes ``rtt + bytes_in / down + compute + bytes_out / up`` on the
worker it is given to, and a scheduler is asked for assignments whenever workers are idle.

The bandwidth-aware scheduler is told each worker's true link rates up front (as if the
workers had already reported them) and learns compute times from completions. The
simulation uses the same cost structure as the scheduler's model, so it shows what the
policy does *when its estimates are right*; it does not validate the estimates themselves.

  python -m evaluation.scheduler_sim
"""

from __future__ import annotations

import asyncio
import heapq
import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from foreman.core.scheduling import (
    BandwidthAwareScheduler,
    FIFOScheduler,
    LeastLoadedScheduler,
    RoundRobinScheduler,
)
from foreman.core.scheduling.scheduler_interface import Task, TaskScheduler, Worker

MB = 1_000_000.0


@dataclass(frozen=True)
class SimWorker:
    worker_id: str
    down_mbps: float  # megabytes per second, Foreman -> worker
    up_mbps: float
    rtt_s: float = 0.02
    sec_per_unit: float = 0.5  # compute seconds per unit of task work


@dataclass(frozen=True)
class SimTask:
    task_id: str
    megabytes_in: float
    megabytes_out: float
    work_units: float = 1.0


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def true_seconds(task: SimTask, worker: SimWorker) -> float:
    return (worker.rtt_s + task.megabytes_in / worker.down_mbps
            + task.work_units * worker.sec_per_unit + task.megabytes_out / worker.up_mbps)


def _scheduler_task(task: SimTask) -> Task:
    # The payload text has the real size; the scheduler only ever sees the text.
    return Task(id=task.task_id, job_id="sim", args=json.dumps({"data": "x" * int(task.megabytes_in * MB)}))


async def _simulate(make: Callable[[_Clock], TaskScheduler], workers: List[SimWorker],
                    tasks: List[SimTask]) -> Dict[str, object]:
    clock = _Clock()
    scheduler = make(clock)
    if isinstance(scheduler, BandwidthAwareScheduler):
        for w in workers:
            scheduler.observe_link_report(w.worker_id, {"down_bps": w.down_mbps * MB, "up_bps": w.up_mbps * MB,
                                                         "rtt_s": w.rtt_s})
    by_id = {w.worker_id: w for w in workers}
    all_workers = {w.worker_id: Worker(id=w.worker_id, status="online") for w in workers}
    pending = {t.task_id: t for t in tasks}
    queue = [_scheduler_task(t) for t in tasks]
    idle = set(by_id)
    running: list = []
    done: Dict[str, int] = {w: 0 for w in by_id}
    while queue or running:
        if idle and queue:
            for sched_task, worker_id in await scheduler.batch_select_workers(queue, set(idle), all_workers):
                sim = pending[sched_task.id]
                queue = [q for q in queue if q.id != sched_task.id]
                idle.discard(worker_id)
                heapq.heappush(running, (clock.now + true_seconds(sim, by_id[worker_id]), sim.task_id, worker_id))
        finish, task_id, worker_id = heapq.heappop(running)
        clock.now = finish
        scheduler.observe_completion(task_id, worker_id, int(pending[task_id].megabytes_out * MB))
        done[worker_id] += 1
        idle.add(worker_id)
    return {"makespan_s": round(clock.now, 2), "tasks_per_worker": done}


def simulate(make: Callable[[_Clock], TaskScheduler], workers: List[SimWorker], tasks: List[SimTask]):
    return asyncio.run(_simulate(make, workers, tasks))


SCHEDULERS: Dict[str, Callable[[_Clock], TaskScheduler]] = {
    "fifo": lambda clock: FIFOScheduler(),
    "round_robin": lambda clock: RoundRobinScheduler(),
    "least_loaded": lambda clock: LeastLoadedScheduler(),
    "bandwidth_aware": lambda clock: BandwidthAwareScheduler(clock=clock, default_compute_s=0.5, max_wait_s=15.0),
}

LINKS = (0.5, 1, 2, 5, 10, 20)  # MB/s


def scenarios(seed: int) -> Dict[str, tuple]:
    """Scenario builders; the seed changes task sizes and which worker gets which link."""
    rng = random.Random(seed)
    ids = [f"w{i}" for i in range(6)]
    rng.shuffle(ids)  # the baselines pick by id/set order, so the layout must vary
    mixed = [SimTask(f"t{i}", rng.choice([0.1, 0.3, 1.0, 3.0, 6.0]), 0.1) for i in range(24)]
    hetero = [SimWorker(w, d, d / 2) for w, d in zip(ids, LINKS)]
    return {
        "heterogeneous links (0.5 to 20 MB/s), 24 mixed-size tasks": (hetero, mixed),
        "heterogeneous links, only 3 tasks for 6 workers": (hetero, [SimTask(f"t{i}", 4.0, 0.1) for i in range(3)]),
        "identical links (5 MB/s), 24 mixed-size tasks": ([SimWorker(w, 5.0, 5.0) for w in ids], mixed),
    }


def compare(seeds: range = range(30)) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean and worst makespan of each scheduler per scenario over ``seeds`` random layouts."""
    samples: Dict[str, Dict[str, List[float]]] = {}
    for seed in seeds:
        for name, (workers, tasks) in scenarios(seed).items():
            for sched, make in SCHEDULERS.items():
                samples.setdefault(name, {}).setdefault(sched, []).append(simulate(make, workers, tasks)["makespan_s"])
    return {name: {sched: {"mean_s": round(sum(v) / len(v), 2), "worst_s": round(max(v), 2), "best_s": round(min(v), 2)}
                   for sched, v in by_sched.items()} for name, by_sched in samples.items()}


def main() -> None:
    results = compare()
    for name, by_sched in results.items():
        print(f"\n{name}")
        baseline = by_sched["fifo"]["mean_s"]
        for sched, r in by_sched.items():
            gain = (1 - r["mean_s"] / baseline) * 100
            print(f"  {sched:<16} mean {r['mean_s']:>6.2f} s ({gain:>+6.1f}% vs fifo)   best {r['best_s']:>6.2f}   worst {r['worst_s']:>6.2f}")


if __name__ == "__main__":
    main()
