"""Experiment harness CLI.

  python -m evaluation.reconfig.run sim     --config C --spec S [--out LOG] [--oracle]
  python -m evaluation.reconfig.run compare --config C --spec S --policies B0,B2,S-aware [--outdir D]
  python -m evaluation.reconfig.run notax   --config C --spec S [--outdir D]
  python -m evaluation.reconfig.run metrics LOG [--oracle LOG]
  python -m evaluation.reconfig.run replay  LOG
  python -m evaluation.reconfig.run dp-gap  [--instances 200] [--workers 5] [--seed 0]
  python -m evaluation.reconfig.run live    --config C --host H --port 9000 --inputs N [--foreman-pid PID]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from foreman.planning.config import RunConfig
from foreman.planning.eventlog import EventLog, read_events
from foreman.planning.flags import FeatureFlags

from .metrics import compute_metrics
from .replay import replay
from .scenario import LiveScenarioRunner, load_scenario
from .sim import Simulator
from .spec import load_spec


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _simulate(config: RunConfig, spec_path: str, out: Optional[str], oracle: bool = False):
    spec = load_spec(spec_path, seq_tokens=config.cost.seq_len * config.cost.batch)
    sim = Simulator(spec, config, load_scenario(config.scenario), oracle=oracle, log_path=out)
    return sim.run()


def cmd_sim(args) -> None:
    config = RunConfig.load(args.config)
    result = _simulate(config, args.spec, args.out, args.oracle)
    oracle_total = None
    if not args.oracle and args.with_oracle:
        oracle_total = _simulate(config, args.spec, None, oracle=True).total_time
    _print(compute_metrics(result.log.events, switch_window_s=config.controller.switch_window_s,
                           oracle_total=oracle_total))


def cmd_compare(args) -> None:
    base = RunConfig.load(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    oracle = _simulate(base, args.spec, str(outdir / "oracle.jsonl"), oracle=True)
    rows = {"oracle": compute_metrics(oracle.log.events)}
    for name in args.policies.split(","):
        flags = base.flags if name not in ("B0", "B1") else FeatureFlags()
        config = replace(base, policy=name, flags=flags)
        result = _simulate(config, args.spec, str(outdir / f"{name}.jsonl"))
        rows[name] = compute_metrics(result.log.events, switch_window_s=config.controller.switch_window_s,
                                     oracle_total=oracle.total_time)
    keys = ["total_completion_s", "regret_s", "reconfiguration_s", "switches", "wasted_switches", "stall_s",
            "steady_period_s", "bytes_moved", "planner_time_s", "deadline_hit_rate", "control_bytes"]
    _print({name: {k: row.get(k) for k in keys} for name, row in rows.items()})


def cmd_notax(args) -> None:
    """Healthy run (no scenario): B0 with flags off vs S-aware with every flag on."""
    base = replace(RunConfig.load(args.config), scenario=[])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for label, config in (("B0_flags_off", replace(base, policy="B0", flags=FeatureFlags())),
                          ("all_features_on", replace(base, policy="S-aware", flags=FeatureFlags.all_on()))):
        result = _simulate(config, args.spec, str(outdir / f"notax_{label}.jsonl"))
        m = compute_metrics(result.log.events)
        rows[label] = {k: m.get(k) for k in ("steady_period_s", "throughput_rps", "total_completion_s", "switches",
                                             "control_bytes", "control_bytes_per_request", "control_bytes_per_min",
                                             "planner_time_s", "planner_calls", "bytes_moved")}
    rows["note"] = ("Worker CPU/battery and Foreman CPU/bandwidth are measured in live runs "
                    "(`run live --foreman-pid`); the simulator reports control-plane bytes and planner time.")
    _print(rows)


def cmd_metrics(args) -> None:
    events = read_events(args.log)
    oracle_total = None
    if args.oracle:
        oracle_total = compute_metrics(read_events(args.oracle))["total_completion_s"]
    _print(compute_metrics(events, oracle_total=oracle_total))


def cmd_replay(args) -> None:
    result = replay(read_events(args.log))
    _print(result)
    if not result["identical"]:
        sys.exit(1)


def cmd_dp_gap(args) -> None:
    from foreman.planning.cost_model import CostOptions
    from foreman.planning.planner import dp_gap

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.planning.fixtures import opts, random_snapshot

    rng = random.Random(args.seed)
    gaps = {"scalar_dp": [], "pareto_dp": []}
    for _ in range(args.instances):
        snap = random_snapshot(rng, n_workers=args.workers)
        n = rng.randint(1, 200)
        for method in gaps:
            exact, approx, gap = dp_gap(snap, n, opts(), method)
            if math.isfinite(exact):
                gaps[method].append(gap)
    _print({m: {"instances": len(g), "max_gap": max(g, default=0.0), "mean_gap": sum(g) / max(1, len(g)),
                "exact_share": sum(1 for x in g if abs(x) < 1e-9) / max(1, len(g))} for m, g in gaps.items()})


def cmd_live(args) -> None:
    """Submit planned inference to a running Foreman and collect live metrics."""
    import websockets

    config = RunConfig.load(args.config)
    log = EventLog(args.out, run_id=config.run_id, seed=config.seed, policy=config.policy,
                   flags=config.flags.to_dict())
    scenario = LiveScenarioRunner(load_scenario(config.scenario), log=log.emit)
    process = None
    if args.foreman_pid:
        import psutil

        process = psutil.Process(args.foreman_pid)
        cpu_start = sum(process.cpu_times()[:2])

    async def run() -> None:
        async with websockets.connect(f"ws://{args.host}:{args.port}", max_size=None) as ws:
            inputs = [{"value": i} for i in range(args.inputs)] if not args.inputs_file else \
                json.loads(Path(args.inputs_file).read_text())
            started = time.monotonic()
            await ws.send(json.dumps({"protocol": "crowdio.reconfiguration", "version": 2, "type": "submit_inference",
                                      "data": {"model_id": args.model_id, "inputs": inputs,
                                               "deadline_s": config.deadline_s}}))
            while True:
                scenario.poll()
                try:
                    message = json.loads(await asyncio.wait_for(ws.recv(), 1.0))
                except asyncio.TimeoutError:
                    continue
                if message.get("type") == "inference_result":
                    log.emit("run_end", wall_s=time.monotonic() - started,
                             results=len(message["data"]["results"]),
                             foreman_cpu_s=(sum(process.cpu_times()[:2]) - cpu_start) if process else None)
                    return

    asyncio.run(run())
    log.close()
    print(f"client log written to {args.out}; merge with the Foreman's CROWDIO log for full metrics")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="evaluation.reconfig.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sim")
    p.add_argument("--config", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--out")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--with-oracle", action="store_true", help="also run the oracle and report regret")
    p.set_defaults(func=cmd_sim)
    p = sub.add_parser("compare")
    p.add_argument("--config", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--policies", default="B0,B1,B2,B3,B4,B5a,B5b,S-aware-placement,S-aware")
    p.add_argument("--outdir", default="evaluation_results/reconfig")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("notax")
    p.add_argument("--config", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--outdir", default="evaluation_results/reconfig")
    p.set_defaults(func=cmd_notax)
    p = sub.add_parser("metrics")
    p.add_argument("log")
    p.add_argument("--oracle")
    p.set_defaults(func=cmd_metrics)
    p = sub.add_parser("replay")
    p.add_argument("log")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("dp-gap")
    p.add_argument("--instances", type=int, default=200)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_dp_gap)
    p = sub.add_parser("live")
    p.add_argument("--config", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--inputs", type=int, default=100)
    p.add_argument("--inputs-file", help="JSON list of serialized input tensors")
    p.add_argument("--model-id", default="distilbert-base-uncased-finetuned-sst-2-english")
    p.add_argument("--foreman-pid", type=int)
    p.add_argument("--out", default="evaluation_results/reconfig/live_client.jsonl")
    p.set_defaults(func=cmd_live)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
