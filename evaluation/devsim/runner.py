"""Assemble a simulated world and run a workload through it.

  run_once   one Foreman + a fleet + a scenario + a workload, with the result written to a folder
  compare    the same world for several schedulers (each in its own process, same bandwidth trace)
  serve      the real Foreman app with the fleet attached, for running your own test scripts

The Foreman code is the real one (dispatcher, schedulers, run log, dashboard); only the devices
and the network are simulated. Simulated runs use a throwaway database, never ``crowdio.db``.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import websockets

from .clock import VirtualClock
from .device import VirtualDevice
from .fleet import MBIT, make_fleet
from .link import LinkEngine
from .scenarios import ScenarioEvent, get_scenario, prepare
from .workloads import DEFAULT_TASKS, make_model_size, run_workload


@dataclass
class SimConfig:
    scenario: str = "congestion"
    devices: int = 15
    pc: int = 0
    scheduler: str = "bandwidth_aware"
    workload: str = "mixed"
    tasks: Optional[int] = None
    seed: int = 1
    speed: float = 1.0
    work_scale: float = 1.0
    mode: str = "model"
    out_dir: str = "devsim_results/run"
    model_scale: float = 0.05
    shared_capacity_mbps: Optional[float] = None
    heartbeat_s: float = 10.0
    horizon_s: float = 3600.0
    timeout_s: float = 900.0
    verbose: bool = False

    def validate(self) -> None:
        get_scenario(self.scenario)
        if self.workload not in DEFAULT_TASKS:
            raise ValueError(f"Unknown workload {self.workload!r}. Available: {', '.join(DEFAULT_TASKS)}")
        if self.speed <= 0 or self.work_scale <= 0:
            raise ValueError("--speed and --work-scale must be positive")
        if self.mode == "real" and self.speed != 1.0:
            raise ValueError("--mode real runs the real task functions, so it needs --speed 1")


# ------------------------------------------------------------------ small helpers
@contextlib.contextmanager
def _env(**values: Optional[str]) -> Iterator[None]:
    """Set environment variables for the duration of a block, then put them back."""
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _quiet(verbose: bool) -> Iterator[None]:
    """The Foreman prints a lot; hide it unless asked."""
    if verbose:
        yield
        return
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            yield
        finally:
            logging.disable(previous)


class _EventLog:
    def __init__(self, path: Path, clock: VirtualClock) -> None:
        self._fh = open(path, "w", encoding="utf-8")
        self._clock = clock

    def __call__(self, event: str, **fields: Any) -> None:
        self._fh.write(json.dumps({"t": round(self._clock.now(), 3), "event": event, **fields}, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))], 2)


async def _play_events(events: Sequence[ScenarioEvent], devices: Dict[str, VirtualDevice], clock: VirtualClock) -> None:
    for event in events:
        await clock.sleep_until(event.at_s)
        device = devices.get(event.device_id)
        if device is not None:
            (device.go_offline if event.action == "leave" else device.go_online)()


async def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


# ------------------------------------------------------------------ summary
def build_summary(cfg: SimConfig, fleet, prepared, devices: Dict[str, VirtualDevice], engine: LinkEngine,
                  result, real_s: float, link_snapshot: Optional[dict]) -> Dict[str, Any]:
    makespan = max(result.makespan_v, 1e-9)
    task_seconds = [s for d in devices.values() for s in d.stats.task_seconds]
    rows = []
    for spec in fleet:
        st, trace = devices[spec.device_id].stats, prepared.traces[spec.device_id]
        rows.append({
            "device": spec.device_id, "kind": spec.kind, "tier": spec.tier, "cpu_factor": spec.cpu_factor,
            "down_mbps": spec.down_mbps, "up_mbps": spec.up_mbps, "rtt_ms": spec.rtt_ms,
            "mean_bandwidth_factor": round(trace.mean(0.0, makespan), 3),
            "min_bandwidth_factor": round(min(trace.values[: int(makespan) + 1]), 3),
            "tasks": st.tasks, "errors": st.errors, "busy_s": round(st.busy_s, 2),
            "utilisation": round(min(1.0, st.busy_s / makespan), 3), "mb_down": round(st.bytes_down / 1e6, 2),
            "mb_up": round(st.bytes_up / 1e6, 2), "model_downloads": st.model_downloads, "connects": st.connects,
        })
    estimates = (link_snapshot or {}).get("workers") or {}
    errors = []
    for spec in fleet:
        est = estimates.get(spec.device_id)
        if est and est.get("samples_down"):
            truth = engine.rate_bps(spec.device_id, "down", makespan) / MBIT
            errors.append(abs(est["down_mbps"] / cfg.speed - truth) / max(truth, 1e-9))
    done = sum(r["tasks"] for r in rows)
    return {
        "config": asdict(cfg),
        "ok": result.ok, "error": result.error,
        "makespan_virtual_s": round(makespan, 2), "makespan_real_s": round(real_s, 2),
        "tasks_expected": result.tasks, "tasks_run": done, "jobs": result.jobs,
        "throughput_tasks_per_s": round(done / makespan, 3),
        "job_seconds": [round(s, 2) for s in result.job_seconds],
        "task_seconds": {"mean": round(sum(task_seconds) / len(task_seconds), 2) if task_seconds else None,
                         "p50": _percentile(task_seconds, 0.5), "p95": _percentile(task_seconds, 0.95),
                         "max": round(max(task_seconds), 2) if task_seconds else None},
        "devices_used": sum(1 for r in rows if r["tasks"]),
        "mean_device_utilisation": round(sum(r["utilisation"] for r in rows) / len(rows), 3),
        "tasks_per_real_second": round(done / max(real_s, 1e-9), 2),
        "link_estimate_error_pct": round(100 * sum(errors) / len(errors), 1) if errors else None,
        "devices": rows,
        "shared_capacity_mbps": cfg.shared_capacity_mbps,
    }


def write_outputs(out: Path, summary: Dict[str, Any], fleet, prepared, engine: LinkEngine, makespan: float) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with open(out / "devices.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary["devices"][0]))
        writer.writeheader()
        writer.writerows(summary["devices"])
    with open(out / "bandwidth_trace.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["device", "t_s", "down_mbps", "up_mbps", "factor"])
        for spec in fleet:
            trace = prepared.traces[spec.device_id]
            for t in range(0, int(math.ceil(makespan)) + 1):
                f = trace.factor(t)
                writer.writerow([spec.device_id, t, round(spec.down_mbps * f, 3), round(spec.up_mbps * f, 3), round(f, 4)])


def format_summary(s: Dict[str, Any]) -> str:
    c = s["config"]
    lines = [
        f"Scenario {c['scenario']} | scheduler {c['scheduler']} | {c['devices']} devices | workload {c['workload']} "
        f"| speed {c['speed']}x | seed {c['seed']}",
        f"  makespan   {s['makespan_virtual_s']:.1f} s simulated ({s['makespan_real_s']:.1f} s real)    "
        f"{'OK' if s['ok'] else 'FAILED: ' + str(s['error'])}",
        f"  tasks      {s['tasks_run']} run for {s['tasks_expected']} expected, {s['throughput_tasks_per_s']} tasks/s, "
        f"{s['devices_used']}/{c['devices']} devices used",
        f"  task time  mean {s['task_seconds']['mean']} s, p95 {s['task_seconds']['p95']} s, max {s['task_seconds']['max']} s",
    ]
    if s.get("link_estimate_error_pct") is not None:
        lines.append(f"  the scheduler's downlink estimates were off by {s['link_estimate_error_pct']}% on average at the end")
    lines += ["", f"  {'device':<14}{'tier':<9}{'cpu':>5}{'down':>7}{'up':>6}{'bw now%':>8}{'tasks':>6}{'busy%':>7}{'MB dn':>8}{'MB up':>7}"]
    for r in sorted(s["devices"], key=lambda r: -r["tasks"]):
        lines.append(f"  {r['device']:<14}{r['tier']:<9}{r['cpu_factor']:>5.1f}{r['down_mbps']:>7.1f}{r['up_mbps']:>6.1f}"
                     f"{100 * r['mean_bandwidth_factor']:>8.0f}{r['tasks']:>6}{100 * r['utilisation']:>7.0f}"
                     f"{r['mb_down']:>8.1f}{r['mb_up']:>7.1f}")
    lines.append("\n  (down/up are nominal Mbit/s; 'bw now%' is the mean share of nominal bandwidth during the run)")
    if c["speed"] > 1 and s["mean_device_utilisation"] < 0.25:
        lines.append(f"\n  WARNING: devices were idle {100 * (1 - s['mean_device_utilisation']):.0f}% of the time. At {c['speed']}x the Foreman's "
                     f"own processing (about {s['tasks_per_real_second']} tasks per real second here) is probably the bottleneck, "
                     f"not the devices. Use --speed 1, or heavier tasks (--work-scale).")
    return "\n".join(lines)


# ------------------------------------------------------------------ the world
async def _start_fleet(cfg: SimConfig, fleet, clock, engine, url: str, events, model_size):
    devices = {spec.device_id: VirtualDevice(spec, url, clock, engine, mode=cfg.mode, model_size=model_size,
                                             seed=cfg.seed, heartbeat_s=cfg.heartbeat_s, events=events)
               for spec in fleet}
    tasks = [asyncio.create_task(d.run()) for d in devices.values()]
    return devices, tasks


async def _stop_fleet(devices, tasks, engine) -> None:
    for d in devices.values():
        await d.close()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await engine.close()


async def run_once(cfg: SimConfig) -> Dict[str, Any]:
    cfg.validate()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fleet = make_fleet(cfg.devices, seed=cfg.seed, pc=cfg.pc)
    prepared = prepare(get_scenario(cfg.scenario), fleet, seed=cfg.seed, horizon_s=cfg.horizon_s)
    shared = cfg.shared_capacity_mbps if cfg.shared_capacity_mbps is not None else prepared.shared_capacity_mbps
    cfg.shared_capacity_mbps = shared
    clock = VirtualClock(cfg.speed)
    engine = LinkEngine(clock, {d.device_id: d for d in fleet}, prepared.traces, shared_capacity_mbps=shared)
    events = _EventLog(out / "events.jsonl", clock)
    model_size = make_model_size(cfg.model_scale)

    with _env(CROWDIO_RUN_LOG_DIR=str(out / "run_logs"), CROWDIO_SCHEDULER=cfg.scheduler), _quiet(cfg.verbose):
        from foreman.core.scheduling import BandwidthAwareScheduler, create_scheduler
        from foreman.core.ws_manager import WebSocketManager
        from foreman.db.base import engine as db_engine
        from foreman.db.base import init_db

        await init_db()
        options = {"shared_capacity_bps": shared * MBIT} if shared and cfg.scheduler == "bandwidth_aware" else {}
        scheduler = create_scheduler(cfg.scheduler, **options)
        if isinstance(scheduler, BandwidthAwareScheduler):
            scheduler.model_size = model_size
        manager = WebSocketManager(scheduler=scheduler)
        server = await websockets.serve(manager.handle_connection, "127.0.0.1", 0, max_size=None)
        port = server.sockets[0].getsockname()[1]
        devices, device_tasks = await _start_fleet(cfg, fleet, clock, engine, f"ws://127.0.0.1:{port}", events, model_size)
        player = None
        try:
            await _wait_for(lambda: len(manager.connection_manager.get_all_worker_ids()) >= len(fleet), 60, "devices to connect")
            await asyncio.sleep(0.3)  # first heartbeats (link measurements) arrive
            clock.reset()  # virtual time 0 = the moment the workload starts, so scenario times line up
            real_start = time.monotonic()
            player = asyncio.create_task(_play_events(prepared.events, devices, clock))
            result = await run_workload(cfg.workload, clock, "127.0.0.1", port, seed=cfg.seed, tasks=cfg.tasks,
                                        timeout_s=cfg.timeout_s, work_scale=cfg.work_scale)
            real_s = time.monotonic() - real_start
            await asyncio.sleep(0.3)
            snapshot = scheduler.link_snapshot()
        finally:
            if player is not None:
                player.cancel()
                await asyncio.gather(player, return_exceptions=True)
            await _stop_fleet(devices, device_tasks, engine)
            server.close()
            await server.wait_closed()
            events.close()
            await db_engine.dispose()

    summary = build_summary(cfg, fleet, prepared, devices, engine, result, real_s, snapshot)
    write_outputs(out, summary, fleet, prepared, engine, summary["makespan_virtual_s"])
    return summary


# ------------------------------------------------------------------ compare
def compare(cfg: SimConfig, schedulers: Sequence[str], repeats: int = 1) -> Dict[str, Any]:
    """Run the same world for each scheduler, each in a fresh process, and tabulate the results."""
    base = Path(cfg.out_dir)
    results: Dict[str, List[Dict[str, Any]]] = {s: [] for s in schedulers}
    for scheduler in schedulers:
        for repeat in range(repeats):
            seed = cfg.seed + repeat
            out = base / scheduler / f"seed{seed}"
            command = [sys.executable, "-m", "evaluation.devsim", "run", "--quiet", "--scenario", cfg.scenario,
                       "--devices", str(cfg.devices), "--pc", str(cfg.pc), "--scheduler", scheduler,
                       "--workload", cfg.workload, "--seed", str(seed), "--speed", str(cfg.speed),
                       "--work-scale", str(cfg.work_scale),
                       "--mode", cfg.mode, "--model-scale", str(cfg.model_scale), "--out", str(out),
                       "--timeout", str(cfg.timeout_s)]
            if cfg.tasks:
                command += ["--tasks", str(cfg.tasks)]
            if cfg.shared_capacity_mbps is not None:
                command += ["--shared-capacity-mbps", str(cfg.shared_capacity_mbps)]
            print(f"  running {scheduler} (seed {seed}) ...", flush=True)
            subprocess.run(command, check=False, timeout=cfg.timeout_s + 120)
            summary_file = out / "summary.json"
            if summary_file.exists():
                results[scheduler].append(json.loads(summary_file.read_text(encoding="utf-8")))
            else:
                print(f"    no result for {scheduler} seed {seed} (see {out})")
    table = {}
    for scheduler, runs in results.items():
        good = [r for r in runs if r["ok"]]
        if not runs:
            continue
        table[scheduler] = {
            "runs": len(runs), "completed": len(good),
            "makespan_s": round(sum(r["makespan_virtual_s"] for r in good) / len(good), 2) if good else None,
            "tasks_per_s": round(sum(r["throughput_tasks_per_s"] for r in good) / len(good), 3) if good else None,
            "task_p95_s": round(sum(r["task_seconds"]["p95"] for r in good) / len(good), 2) if good else None,
            "devices_used": round(sum(r["devices_used"] for r in good) / len(good), 1) if good else None,
        }
    (base / "comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table


def format_comparison(table: Dict[str, Any], cfg: SimConfig, repeats: int) -> str:
    first = next((v["makespan_s"] for v in table.values() if v["makespan_s"]), None)
    lines = [f"Scenario {cfg.scenario} | {cfg.devices} devices | workload {cfg.workload} | speed {cfg.speed}x | "
             f"{repeats} run(s) each, same bandwidth for every scheduler",
             f"  {'scheduler':<17}{'makespan s':>11}{'vs first':>10}{'tasks/s':>9}{'task p95 s':>12}{'devices used':>14}{'done':>7}"]
    for name, v in table.items():
        gain = f"{100 * (1 - v['makespan_s'] / first):+.1f}%" if v["makespan_s"] and first else "-"
        lines.append(f"  {name:<17}{v['makespan_s'] if v['makespan_s'] is not None else '-':>11}{gain:>10}"
                     f"{v['tasks_per_s'] if v['tasks_per_s'] is not None else '-':>9}"
                     f"{v['task_p95_s'] if v['task_p95_s'] is not None else '-':>12}"
                     f"{v['devices_used'] if v['devices_used'] is not None else '-':>14}{str(v['completed']) + '/' + str(v['runs']):>7}")
    return "\n".join(lines)


# ------------------------------------------------------------------ serve
async def serve(cfg: SimConfig, duration_s: Optional[float] = None) -> None:
    """The real Foreman app on :8000/:9000 with the fleet attached. Stop with Ctrl+C."""
    import uvicorn

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fleet = make_fleet(cfg.devices, seed=cfg.seed, pc=cfg.pc)
    prepared = prepare(get_scenario(cfg.scenario), fleet, seed=cfg.seed, horizon_s=cfg.horizon_s)
    shared = cfg.shared_capacity_mbps if cfg.shared_capacity_mbps is not None else prepared.shared_capacity_mbps
    clock = VirtualClock(cfg.speed)
    engine = LinkEngine(clock, {d.device_id: d for d in fleet}, prepared.traces, shared_capacity_mbps=shared)
    events = _EventLog(out / "events.jsonl", clock)
    model_size = make_model_size(cfg.model_scale)
    with _env(CROWDIO_RUN_LOG_DIR=str(out / "run_logs"), CROWDIO_SCHEDULER=cfg.scheduler):
        server = uvicorn.Server(uvicorn.Config("foreman.main:app", host="0.0.0.0", port=8000, log_level="warning"))
        serving = asyncio.create_task(server.serve())
        await _wait_for(lambda: server.started or serving.done(), 60, "the Foreman to start")
        if serving.done():
            serving.result()  # raises, e.g. when port 8000 is already in use
        import foreman.main as foreman_main
        from foreman.core.scheduling import BandwidthAwareScheduler

        scheduler = foreman_main.ws_manager.task_dispatcher.scheduler
        if isinstance(scheduler, BandwidthAwareScheduler):
            scheduler.model_size = model_size
        devices, device_tasks = await _start_fleet(cfg, fleet, clock, engine, "ws://127.0.0.1:9000", events, model_size)
        player = None
        try:
            await _wait_for(lambda: len(foreman_main.ws_manager.connection_manager.get_all_worker_ids()) >= len(fleet),
                            60, "devices to connect")
            clock.reset()
            player = asyncio.create_task(_play_events(prepared.events, devices, clock))
            print(f"\n{len(fleet)} simulated devices connected to the Foreman (scheduler: {type(scheduler).__name__}, "
                  f"scenario: {cfg.scenario}, speed {cfg.speed}x)")
            print("  Dashboard   http://localhost:8000")
            print("  WebSocket   ws://localhost:9000   (run your test scripts against this)")
            print("  Press Ctrl+C to stop.\n", flush=True)
            await (asyncio.sleep(duration_s) if duration_s else asyncio.Event().wait())
        finally:
            if player is not None:
                player.cancel()
                await asyncio.gather(player, return_exceptions=True)
            await _stop_fleet(devices, device_tasks, engine)
            events.close()
            server.should_exit = True
            await asyncio.gather(serving, return_exceptions=True)
