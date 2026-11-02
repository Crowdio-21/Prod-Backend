"""Bandwidth-aware scheduler.

Every task is placed where its **estimated completion time** is smallest, and that
estimate includes the time to move its data:

    T(task, worker) = rtt + (args + code + model not yet on the worker) / down
                      + compute + expected result bytes / up

This is the latency model of Neurosurgeon (compute + transfer / bandwidth), applied
per task. Batches are mapped with the **Sufferage** heuristic (Maheswaran et al.
1999): the task that would lose the most by missing its best worker chooses first, so
large payloads win the fast links. When tasks share one wireless medium or the
Foreman's own link, concurrent flows are stretched by a max-min fair share and the
mapping is re-run with the stretched costs, as in network-aware HEFT (Yousaf and
Welzl 2014). Bandwidth is tracked as an EWMA (alpha 0.3, after Cotter et al.).

Only ``batch_select_workers`` is used by the dispatcher today; ``select_worker`` and
``select_task`` are kept consistent with it. See docs/bandwidth_aware_scheduling.md.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from foreman.core.payload_store import is_payload_ref, payload_ref_head, payload_ref_size

from .link_estimator import LinkEstimate, LinkEstimator
from .scheduler_interface import Task, TaskScheduler

FRAMING_BYTES = 512  # websocket/JSON envelope around an assign_task message
_MODEL_RE = re.compile(r'"model_path"\s*:\s*"([^"]+)"')
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "tools" / "onnx_partitions_sentiment"


def fair_level(caps: Sequence[float], capacity: float) -> float:
    """Max-min fair per-flow rate: the level L with sum(min(cap, L)) == capacity.

    Returns infinity when the flows fit without sharing.
    """
    ordered = sorted(caps)
    remaining = capacity
    for index, cap in enumerate(ordered):
        share = remaining / (len(ordered) - index)
        if cap <= share:
            remaining -= cap
        else:
            return share
    return math.inf


def default_model_size(name: str) -> Optional[int]:
    """Size of a served model file (``CROWDIO_MODEL_DIR``), None if unknown."""
    root = Path(os.getenv("CROWDIO_MODEL_DIR") or DEFAULT_MODEL_DIR)
    try:
        return (root / Path(name).name).stat().st_size
    except OSError:
        return None


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    in_bytes: int  # args + code + framing, sent Foreman -> worker
    args_bytes: int
    out_bytes: float  # expected result size, worker -> Foreman
    model: Optional[str]
    model_bytes: int
    key: str  # groups tasks that cost about the same to compute (a model or a function)


class BandwidthAwareScheduler(TaskScheduler):
    def __init__(
        self,
        estimator: Optional[LinkEstimator] = None,
        *,
        shared_capacity_bps: Optional[float] = None,
        result_ratio: float = 1.0,
        default_compute_s: float = 1.0,
        max_window: int = 64,
        contention_iterations: int = 3,
        max_wait_s: float = 60.0,
        max_defer_s: float = 30.0,
        model_size: Callable[[str], Optional[int]] = default_model_size,
        clock: Callable[[], float] = time.monotonic,
        inflight_ttl_s: float = 3600.0,
    ) -> None:
        self.estimator = estimator or LinkEstimator(clock=clock)
        self.shared_capacity_bps = shared_capacity_bps
        self.result_ratio = result_ratio
        self.default_compute_s = default_compute_s
        self.max_window = max_window
        self.contention_iterations = max(1, contention_iterations)
        self.max_wait_s = max_wait_s  # aging: a task that has waited this long must start on an idle worker
        self.max_defer_s = max_defer_s  # a task only waits for a worker expected to be free within this long
        self.model_size = model_size
        self.clock = clock
        self.inflight_ttl_s = inflight_ttl_s
        self._cached: Dict[str, Set[str]] = {}  # worker -> model files it has been sent before
        self._out_bytes: Dict[str, float] = {}  # key -> EWMA of observed result bytes
        # task -> (worker, profile, dispatched at, model bytes sent, estimated seconds)
        self._inflight: Dict[str, Tuple[str, TaskProfile, float, int, float]] = {}
        self._first_seen: Dict[str, float] = {}  # task -> when the scheduler first considered it
        self._model_sizes: Dict[str, int] = {}
        self.decisions: Deque[dict] = deque(maxlen=100)

    # ------------------------------------------------------------ task profile
    def profile(self, task: Task) -> TaskProfile:
        args = task.args or ""
        if is_payload_ref(args):
            args_bytes = payload_ref_size(args) or len(args)
            head = payload_ref_head(args) or ""
        else:
            args_bytes = len(args.encode("utf-8"))
            head = args[:4096]
        code = task.stage_func_code or ""
        match = _MODEL_RE.search(head)
        model = Path(match.group(1)).name if match else None
        model_bytes = 0
        if model:
            if model not in self._model_sizes:
                self._model_sizes[model] = self.model_size(model) or 0
            model_bytes = self._model_sizes[model]
        key = model or (hashlib.sha1(code.encode("utf-8")).hexdigest()[:10] if code else "default")
        out_bytes = self._out_bytes.get(key, self.result_ratio * args_bytes)
        return TaskProfile(task.id, args_bytes + len(code.encode("utf-8")) + FRAMING_BYTES, args_bytes,
                           out_bytes, model, model_bytes, key)

    # ------------------------------------------------------------ cost model
    def cost(self, profile: TaskProfile, worker_id: str, est: LinkEstimate,
             level_down: float = math.inf, level_up: float = math.inf) -> float:
        """Estimated seconds from dispatch to result for ``profile`` on ``worker_id``."""
        down = min(est.down_bps, level_down)
        up = min(est.up_bps, level_up)
        missing = profile.model_bytes if profile.model and profile.model not in self._cached.get(worker_id, ()) else 0
        compute = self.estimator.compute_seconds(worker_id, profile.key, self.default_compute_s)
        return est.rtt_s + (profile.in_bytes + missing) / down + compute + profile.out_bytes / up

    def _tie_key(self, worker_id: str, all_workers: Mapping[str, object]) -> Tuple[float, str]:
        worker = all_workers.get(worker_id)
        return (-(getattr(worker, "success_rate", 1.0) or 0.0), worker_id)

    # ------------------------------------------------------------ selection
    async def select_worker(self, task: Task, available_workers: Set[str], all_workers: dict) -> Optional[str]:
        if not available_workers:
            return None
        profile = self.profile(task)
        ests = self.estimator.estimates(available_workers, self.clock())
        best = min(available_workers,
                   key=lambda w: (self.cost(profile, w, ests[w]), *self._tie_key(w, all_workers)))
        self._commit(profile, best, self.cost(profile, best, ests[best]), ests[best])
        return best

    async def select_task(self, pending_tasks: List[Task], worker_id: str) -> Optional[Task]:
        """Pick the pending task for which this worker is closest to the best worker for it."""
        if not pending_tasks:
            return None
        window = pending_tasks[:8]
        known = set(self.estimator.known_workers()) | {worker_id}
        ests = self.estimator.estimates(known, self.clock())
        best_task, best_regret = window[0], math.inf
        for task in window:
            profile = self.profile(task)
            mine = self.cost(profile, worker_id, ests[worker_id])
            regret = mine - min(self.cost(profile, w, ests[w]) for w in known)
            if regret < best_regret - 1e-12:
                best_task, best_regret = task, regret
        return best_task

    async def batch_select_workers(self, tasks: List[Task], available_workers: Set[str],
                                   all_workers: dict) -> List[tuple]:
        """Map tasks to workers by earliest completion time (Sufferage with worker ready times).

        Busy workers are candidates too, at the time they are expected to be free. A task whose
        best slot is on a busy worker waits for it instead of going to a slow idle worker now.
        Waiting is bounded: only for a worker expected free within ``max_defer_s``, never when
        nothing is running (no completion would wake the task up), and a task that has waited
        ``max_wait_s`` (aging) is forced onto an idle worker. Looking at the first ``max_window``
        tasks lets large payloads claim the fast links first.
        """
        if not tasks or not available_workers:
            return []
        now = self.clock()
        self._expire(now)
        idle = sorted(available_workers)
        busy = self._busy_ready(now, set(idle), all_workers)
        window = list(tasks[: self.max_window])
        profiles = [self.profile(t) for t in window]
        ests = self.estimator.estimates(list(idle) + list(busy), now)
        for t in window:
            self._first_seen.setdefault(t.id, now)
        forced = {i for i, t in enumerate(window) if now - self._first_seen[t.id] >= self.max_wait_s}

        level_down = level_up = math.inf
        pairs: List[Tuple[int, str]] = []
        for _ in range(self.contention_iterations if self.shared_capacity_bps else 1):
            pairs = self._map(profiles, idle, busy, ests, now, level_down, level_up, all_workers, forced)
            if not self.shared_capacity_bps:
                break
            flows = [w for _, w in pairs]
            new_down = fair_level([ests[w].down_bps for w in flows], self.shared_capacity_bps)
            new_up = fair_level([ests[w].up_bps for w in flows], self.shared_capacity_bps)
            if (new_down, new_up) == (level_down, level_up):
                break
            level_down, level_up = new_down, new_up
            # Re-map with the stretched costs (network-aware HEFT's iteration).

        result = []
        for i, w in sorted(pairs):
            seconds = self.cost(profiles[i], w, ests[w], level_down, level_up)
            self._commit(profiles[i], w, seconds, ests[w])
            self._first_seen.pop(window[i].id, None)
            result.append((window[i], w))
        return result

    def _busy_ready(self, now: float, idle: Set[str], all_workers: Mapping[str, object]) -> Dict[str, float]:
        """Busy workers worth waiting for: ``{worker: time it is expected to be free}``."""
        ready: Dict[str, float] = {}
        for worker_id, _, started, _, seconds in self._inflight.values():
            worker = all_workers.get(worker_id)
            if worker_id in idle or getattr(worker, "status", "online") == "offline":
                continue
            free_at = started + seconds
            if free_at <= now:  # overdue: probably almost done
                free_at = now + max(0.25 * seconds, 0.1)
            if free_at - now <= self.max_defer_s:
                ready[worker_id] = max(ready.get(worker_id, 0.0), free_at)
        return ready

    def _map(self, profiles: Sequence[TaskProfile], idle: Sequence[str], busy: Mapping[str, float],
             ests: Mapping[str, LinkEstimate], now: float, level_down: float, level_up: float,
             all_workers: Mapping[str, object], forced: Set[int]) -> List[Tuple[int, str]]:
        """Sufferage mapping on completion times. Returns the (task, idle worker) pairs to start now."""
        workers = list(idle) + [w for w in busy if w not in idle]
        ties = {w: self._tie_key(w, all_workers) for w in workers}
        cost = {(i, w): self.cost(p, w, ests[w], level_down, level_up)
                for i, p in enumerate(profiles) for w in workers}
        ready = {w: now for w in idle}
        ready.update({w: busy[w] for w in busy if w not in idle})
        started: Dict[str, int] = {}  # idle worker -> the task it starts now (the first one mapped to it)

        def map_tasks(task_ids: List[int], candidates: Sequence[str]) -> None:
            remaining = list(task_ids)
            while remaining and candidates:
                pick = None  # (sufferage, -task, task, worker)
                for i in remaining:
                    best = second = None
                    for w in candidates:
                        done = (ready[w] + cost[(i, w)], ties[w], w)
                        if best is None or done < best:
                            best, second = done, best
                        elif second is None or done < second:
                            second = done
                    sufferage = (second[0] - best[0]) if second else 0.0
                    if pick is None or (sufferage, -i) > (pick[0], pick[1]):
                        pick = (sufferage, -i, i, best[2])
                _, _, chosen, worker = pick
                if worker in idle and worker not in started and ready[worker] <= now:
                    started[worker] = chosen
                ready[worker] += cost[(chosen, worker)]
                remaining.remove(chosen)

        # Tasks that have already waited long enough must start now: idle workers only.
        map_tasks(sorted(forced), [w for w in idle])
        # Everything else competes for idle and soon-free workers alike.
        map_tasks([i for i in range(len(profiles)) if i not in forced], workers)
        return [(i, w) for w, i in started.items()]

    # ------------------------------------------------------------ feedback
    def _commit(self, profile: TaskProfile, worker_id: str, seconds: float, est: LinkEstimate) -> None:
        now = self.clock()
        cached = self._cached.setdefault(worker_id, set())
        model_sent = profile.model_bytes if profile.model and profile.model not in cached else 0
        self._inflight[profile.task_id] = (worker_id, profile, now, model_sent, seconds)
        if profile.model:
            cached.add(profile.model)
        self.decisions.append({
            "task_id": profile.task_id, "worker_id": worker_id, "estimated_s": round(seconds, 3),
            "send_bytes": profile.in_bytes, "model": profile.model, "imputed_link": est.imputed,
        })

    def _expire(self, now: float) -> None:
        for task_id in [t for t, (_, _, at, _, _) in self._inflight.items() if now - at > self.inflight_ttl_s]:
            del self._inflight[task_id]
        for task_id in [t for t, at in self._first_seen.items() if now - at > self.inflight_ttl_s]:
            del self._first_seen[task_id]

    def observe_link_report(self, worker_id: str, report: dict) -> None:
        self.estimator.observe_report(worker_id, report)

    def observe_completion(self, task_id: str, worker_id: str, result_bytes: int) -> None:
        entry = self._inflight.pop(task_id, None)
        if entry is None or entry[0] != worker_id:
            return
        _, profile, started, model_sent, _ = entry
        now = self.clock()
        est = self.estimator.estimate(worker_id, now)
        # What is left after the predicted transfer is the compute time; transfer is not counted twice.
        predicted = est.rtt_s + (profile.in_bytes + model_sent) / est.down_bps + result_bytes / est.up_bps
        self.estimator.observe_compute(worker_id, profile.key, (now - started) - predicted, now)
        previous = self._out_bytes.get(profile.key)
        a = self.estimator.alpha
        self._out_bytes[profile.key] = result_bytes if previous is None else a * result_bytes + (1 - a) * previous

    def link_snapshot(self) -> dict:
        return {
            "workers": self.estimator.snapshot(self.clock()),
            "shared_capacity_mbps": None if self.shared_capacity_bps is None else self.shared_capacity_bps * 8 / 1e6,
            "result_bytes_by_key": {k: round(v) for k, v in self._out_bytes.items()},
            "tasks_waiting": len(self._first_seen),
            "recent_decisions": list(self.decisions)[-20:],
        }
