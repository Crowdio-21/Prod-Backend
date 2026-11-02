"""Command line for the device simulator.

  python -m evaluation.devsim scenarios
  python -m evaluation.devsim trace   --scenario drop --seconds 120
  python -m evaluation.devsim run     --scenario drop --scheduler bandwidth_aware --workload mixed
  python -m evaluation.devsim compare --scenario drop --schedulers fifo,round_robin,bandwidth_aware --repeats 3
  python -m evaluation.devsim serve   --scenario congestion        (then run your own tests against it)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

RAMP = " .:-=+*#%@"  # ASCII only: Windows consoles choke on block characters


def _config(args):
    from .runner import SimConfig

    return SimConfig(
        scenario=args.scenario, devices=args.devices, pc=args.pc, scheduler=getattr(args, "scheduler", "bandwidth_aware"),
        workload=getattr(args, "workload", "mixed"), tasks=getattr(args, "tasks", None), seed=args.seed, speed=args.speed,
        work_scale=getattr(args, "work_scale", 1.0),
        mode=getattr(args, "mode", "model"), out_dir=args.out, model_scale=getattr(args, "model_scale", 0.05),
        shared_capacity_mbps=getattr(args, "shared_capacity_mbps", None), timeout_s=getattr(args, "timeout", 900.0),
        verbose=getattr(args, "verbose", False))


def _isolate_database(out_dir: str) -> None:
    """Simulated devices must never register in the real crowdio.db; give the run its own database."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    db = directory / "devsim.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    os.environ["CROWDIO_DATABASE_URL"] = "sqlite+aiosqlite:///" + str(db.resolve()).replace("\\", "/")


def cmd_scenarios(args) -> None:
    from .scenarios import SCENARIOS

    for scenario in SCENARIOS.values():
        print(f"  {scenario.name:<12}{scenario.description}")
    from .workloads import DEFAULT_TASKS

    print("\nworkloads: " + ", ".join(f"{k} ({v} tasks)" for k, v in DEFAULT_TASKS.items()))


def cmd_trace(args) -> None:
    from .fleet import make_fleet
    from .scenarios import get_scenario, prepare

    fleet = make_fleet(args.devices, seed=args.seed, pc=args.pc)
    prepared = prepare(get_scenario(args.scenario), fleet, seed=args.seed, horizon_s=args.seconds)
    width = 60
    print(f"{args.scenario}: bandwidth as a share of nominal over {args.seconds:.0f} s "
          f"(one column = {args.seconds / width:.1f} s;  ' ' ~0%  '@' >=120%)\n")
    for spec in fleet:
        trace = prepared.traces[spec.device_id]
        cols = []
        for c in range(width):
            mean = trace.mean(c * args.seconds / width, (c + 1) * args.seconds / width)
            cols.append(RAMP[min(len(RAMP) - 1, int(max(0.0, mean) / 1.2 * (len(RAMP) - 1) + 0.5))])
        print(f"  {spec.device_id:<14}{spec.tier:<9}{spec.down_mbps:>6.1f} Mbit/s |{''.join(cols)}| mean {100 * trace.mean(0, args.seconds):>3.0f}%")
    for e in prepared.events:
        print(f"  t={e.at_s:>6.1f} s  {e.device_id} {e.action}s")
    if prepared.shared_capacity_mbps:
        print(f"  all devices share {prepared.shared_capacity_mbps:.0f} Mbit/s")


def cmd_run(args) -> int:
    _isolate_database(args.out)
    from .runner import format_summary, run_once

    cfg = _config(args)
    print(f"Running {cfg.devices} simulated devices, scenario {cfg.scenario}, scheduler {cfg.scheduler} ...", flush=True)
    summary = asyncio.run(run_once(cfg))
    if not args.quiet:
        print(format_summary(summary))
        print(f"\nResults in {cfg.out_dir}  (summary.json, devices.csv, bandwidth_trace.csv, events.jsonl, run_logs/)")
    return 0 if summary["ok"] else 1


def cmd_compare(args) -> int:
    from .runner import compare, format_comparison

    cfg = _config(args)
    schedulers = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    print(f"Comparing {', '.join(schedulers)} on scenario {cfg.scenario} ({args.repeats} run(s) each)\n")
    table = compare(cfg, schedulers, args.repeats)
    print("\n" + format_comparison(table, cfg, args.repeats))
    print(f"\nPer-run results in {cfg.out_dir}/<scheduler>/seed<N>/")
    return 0


def cmd_serve(args) -> int:
    _isolate_database(args.out)
    from .runner import serve

    try:
        asyncio.run(serve(_config(args), args.duration))
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.devsim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, workload=True):
        p.add_argument("--scenario", default="congestion")
        p.add_argument("--devices", type=int, default=15)
        p.add_argument("--pc", type=int, default=0, help="how many of the devices are PCs (the rest are phones)")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--speed", type=float, default=1.0,
                       help="simulated seconds per real second (default 1; above ~2 the Foreman's own processing time, "
                            "which is not sped up, starts to distort results)")
        p.add_argument("--out", default="devsim_results/run")
        p.add_argument("--shared-capacity-mbps", type=float, default=None, help="one medium all devices share")
        if workload:
            p.add_argument("--workload", default="mixed", help="mixed, uniform, dnn3 or stream")
            p.add_argument("--tasks", type=int, default=None, help="tasks (inputs for dnn3, per job for stream)")
            p.add_argument("--mode", choices=("model", "real"), default="model",
                           help="model: tasks take modelled time; real: really run the task function (speed 1)")
            p.add_argument("--model-scale", type=float, default=0.05, help="shrink model files for dnn3 (1.0 = real size)")
            p.add_argument("--work-scale", type=float, default=1.0, help="multiply every task's compute time")
            p.add_argument("--timeout", type=float, default=900.0, help="give up on a run after this many real seconds")

    sub.add_parser("scenarios", help="list scenarios and workloads").set_defaults(func=cmd_scenarios)
    p = sub.add_parser("trace", help="show a scenario's bandwidth over time without running anything")
    common(p, workload=False)
    p.add_argument("--seconds", type=float, default=120.0)
    p.set_defaults(func=cmd_trace)
    p = sub.add_parser("run", help="run one scheduler through a scenario")
    common(p)
    p.add_argument("--scheduler", default="bandwidth_aware")
    p.add_argument("--quiet", action="store_true", help="write the result files only")
    p.add_argument("--verbose", action="store_true", help="show the Foreman's own output")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("compare", help="run several schedulers through the same scenario and compare")
    common(p)
    p.add_argument("--schedulers", default="fifo,round_robin,bandwidth_aware")
    p.add_argument("--repeats", type=int, default=1, help="runs per scheduler, with seeds seed, seed+1, ...")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("serve", help="run the real Foreman (:8000, :9000) with the fleet attached, for your own tests")
    common(p, workload=False)
    p.add_argument("--scheduler", default="bandwidth_aware")
    p.add_argument("--model-scale", type=float, default=0.05)
    p.add_argument("--duration", type=float, default=None, help="stop after this many real seconds")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
