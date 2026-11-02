"""Jobs the simulator submits, through the real SDK, to the simulated fleet.

Every task carries a ``sim`` hint (work seconds on the reference device, result size) that a
simulated device in ``model`` mode reads, and that ``sim_task`` honours in ``real`` mode.
Keys are ordered ``sim``, ``model_path``, ``data`` so the hint and model name sit in the first
4 KB of the arguments, where the bandwidth-aware scheduler looks for them.

  mixed    one job of 150 tasks with payloads from 20 KB to 5 MB, 1 to 3 s of work each
  uniform  one job of 120 identical tasks (500 KB in, 50 KB out, 2 s of work)
  dnn3     a three-stage pipeline shaped like the DistilBERT sentiment test (40 inputs): three model
           files, ~197 KB activations between stages (model sizes are scaled by --model-scale)
  stream   10 jobs of 12 tasks, one every 8 virtual seconds, so bandwidth changes between jobs

Tasks are deliberately heavy (seconds of work): the Foreman can dispatch only about 6 to 7 tasks per
second with 15 workers, so shorter tasks would measure the Foreman instead of the devices. Scale
all work with ``work_scale``.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from foreman.core.scheduling.bandwidth_scheduler import default_model_size

from .clock import VirtualClock

KB, MB = 1_000, 1_000_000
DEFAULT_TASKS = {"mixed": 150, "uniform": 120, "dnn3": 40, "stream": 12}


def sim_task(payload):
    """The function the SDK ships to workers (``real`` mode runs it; ``model`` mode only reads its hint)."""
    import time

    spec = payload if isinstance(payload, dict) else {}
    nested = spec.get("original_args") if isinstance(spec.get("original_args"), dict) else {}
    hint = spec.get("sim") or nested.get("sim") or {}
    time.sleep(float(hint.get("work_s", 0.0)))
    return {"sim": True, "pad": "x" * int(hint.get("out_bytes", 0))}


def sim_dnn_stage(payload):
    """Stand-in for an onnxruntime partition stage.

    The Foreman treats a task as DNN work (mobile workers only) when its source mentions onnxruntime,
    so the ``dnn3`` pipeline ships this function: it behaves like ``sim_task``.
    """
    import time

    spec = payload if isinstance(payload, dict) else {}
    nested = spec.get("original_args") if isinstance(spec.get("original_args"), dict) else {}
    hint = spec.get("sim") or nested.get("sim") or {}
    time.sleep(float(hint.get("work_s", 0.0)))
    return {"sim": True, "pad": "x" * int(hint.get("out_bytes", 0))}


def payload(in_bytes: int, work_s: float, out_bytes: int, model: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"sim": {"work_s": round(work_s, 3), "out_bytes": int(out_bytes)}}
    if model:
        body["model_path"] = model
    body["data"] = "x" * int(in_bytes)
    return body


def make_model_size(scale: float = 0.05) -> Callable[[str], Optional[int]]:
    """Size of the model files the workloads name: the real files if present, scaled down.

    ``dnn3`` would otherwise spend minutes of virtual time on a 180 MB model over a 5 Mbit/s link
    (as a real phone would). ``scale`` shrinks that so a run finishes; 1.0 is the real size.
    """
    fallback = {"cell_a.onnx": 180 * MB, "cell_b.onnx": 85 * MB, "cell_c.onnx": 2 * MB}

    def size(name: str) -> Optional[int]:
        real = default_model_size(name) or fallback.get(name)
        return None if real is None else max(1, int(real * scale))

    return size


@dataclass
class WorkloadResult:
    name: str
    jobs: int
    tasks: int
    ok: bool
    started_v: float
    ended_v: float
    job_seconds: List[float] = field(default_factory=list)  # virtual seconds per job
    error: Optional[str] = None

    @property
    def makespan_v(self) -> float:
        return self.ended_v - self.started_v


def _mixed_payloads(rng: random.Random, n: int, work_scale: float = 1.0) -> List[Dict[str, Any]]:
    sizes = [20 * KB, 100 * KB, 500 * KB, 2 * MB, 5 * MB]
    out = []
    for _ in range(n):
        size = rng.choices(sizes, weights=[3, 3, 2, 1, 1])[0]
        out.append(payload(size, rng.uniform(1.0, 3.0) * work_scale, max(KB, size // 20)))
    return out


async def _run_map(clock: VirtualClock, host: str, port: int, payloads: List[Dict[str, Any]]):
    from crowdio.client import CrowdComputeClient

    client = CrowdComputeClient()
    await client.connect(host, port)
    try:
        started = clock.now()
        results = await client.map(sim_task, payloads)
        return results, clock.now() - started
    finally:
        await client.disconnect()


async def run_workload(name: str, clock: VirtualClock, host: str, port: int, *, seed: int = 0,
                       tasks: Optional[int] = None, timeout_s: float = 900.0,
                       work_scale: float = 1.0) -> WorkloadResult:
    n = tasks or DEFAULT_TASKS[name]
    rng = random.Random(f"workload:{seed}:{name}")
    started = clock.now()
    try:
        if name in ("mixed", "uniform"):
            data = (_mixed_payloads(rng, n, work_scale) if name == "mixed"
                    else [payload(500 * KB, 2.0 * work_scale, 50 * KB) for _ in range(n)])
            results, took = await asyncio.wait_for(_run_map(clock, host, port, data), timeout_s)
            return WorkloadResult(name, 1, n, len(results) == n, started, clock.now(), [took])
        if name == "dnn3":
            results = await asyncio.wait_for(_run_pipeline(host, port, n, work_scale), timeout_s)
            return WorkloadResult(name, 1, 3 * n, len(results) == n, started, clock.now(), [clock.now() - started])
        if name == "stream":
            return await asyncio.wait_for(_run_stream(clock, host, port, rng, n, work_scale), timeout_s)
        raise ValueError(f"Unknown workload {name!r}. Available: {', '.join(DEFAULT_TASKS)}")
    except asyncio.TimeoutError:
        return WorkloadResult(name, 0, n, False, started, clock.now(), error=f"timed out after {timeout_s:.0f} s")


async def _run_pipeline(host: str, port: int, n: int, work_scale: float = 1.0):
    from crowdio.client import CrowdComputeClient

    stages = [("cell_a", "cell_a.onnx", 2.0, 1 * KB, 197 * KB), ("cell_b", "cell_b.onnx", 1.5, 0, 197 * KB),
              ("cell_c", "cell_c.onnx", 0.1, 0, KB)]
    definition = [{"name": name, "func": sim_dnn_stage, "pass_upstream_results": i > 0,
                   "args_list": [payload(in_b, work * work_scale, out_b, model) for _ in range(n)]}
                  for i, (name, model, work, in_b, out_b) in enumerate(stages)]
    client = CrowdComputeClient()
    await client.connect(host, port)
    try:
        results = await client.pipeline(definition)
        return results.get("raw_results", results) if isinstance(results, dict) else results
    finally:
        await client.disconnect()


async def _run_stream(clock: VirtualClock, host: str, port: int, rng: random.Random, tasks_per_job: int,
                      work_scale: float = 1.0, jobs: int = 10, every_s: float = 8.0) -> WorkloadResult:
    started = clock.now()

    async def one(index: int):
        await clock.sleep_until(started + index * every_s)
        data = _mixed_payloads(random.Random(f"stream:{index}:{rng.random()}"), tasks_per_job, work_scale)
        results, took = await _run_map(clock, host, port, data)
        return len(results) == tasks_per_job, took

    outcomes = await asyncio.gather(*(one(i) for i in range(jobs)))
    return WorkloadResult("stream", jobs, jobs * tasks_per_job, all(ok for ok, _ in outcomes), started, clock.now(),
                          [took for _, took in outcomes])
