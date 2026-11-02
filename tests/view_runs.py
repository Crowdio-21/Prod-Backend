#!/usr/bin/env python3
"""Show what happened in past runs (written by the Foreman to logs/runs/).

  python tests/view_runs.py                 table of the latest runs
  python tests/view_runs.py --last          full report of the latest run
  python tests/view_runs.py --job 3f2a      full report of the run whose job id starts with 3f2a
  python tests/view_runs.py --json --last   the raw record
  python tests/view_runs.py --dir other/dir read another log directory
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from foreman.core.run_log import find_run, read_runs, render_text  # noqa: E402


def table(records) -> str:
    header = f"{'started':<20}{'job':<10}{'status':<20}{'tasks':>11}{'workers':>8}{'duration':>10}{'tasks/s':>9}{'fail':>5}  scope"
    rows = [header, "-" * len(header)]
    for r in records:
        tasks = r["tasks"]
        workers = len([w for w in r["workers"] if w["worker_id"] != "unassigned"])
        duration = r["duration_seconds"]
        rate = r["throughput_tasks_per_s"]
        rows.append(
            f"{(r['started_at'] or '-')[:19].replace('T', ' '):<20}{str(r['run_id'])[:8]:<10}{r['status']:<20}"
            f"{str(tasks['completed']) + '/' + str(tasks['total']):>11}{workers:>8}"
            f"{('-' if duration is None else f'{duration:.1f}s'):>10}{('-' if rate is None else f'{rate:.2f}'):>9}"
            f"{r['failures']['count']:>5}  {r.get('dnn_scope') or '-'}"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", help="run log directory (default: CROWDIO_RUN_LOG_DIR or logs/runs)")
    parser.add_argument("--limit", type=int, default=10, help="runs to list (default 10)")
    parser.add_argument("--job", help="job id (or its first characters) to show in full")
    parser.add_argument("--last", action="store_true", help="show the latest run in full")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of text")
    args = parser.parse_args()

    if args.job or args.last:
        record = find_run(args.job, args.dir) if args.job else (read_runs(args.dir, 1) or [None])[0]
        if record is None:
            sys.exit("No matching run found. Runs are written to the log directory when a job finishes.")
        print(json.dumps(record, indent=2) if args.json else render_text(record))
        return
    records = read_runs(args.dir, args.limit)
    if not records:
        sys.exit("No runs logged yet. Run a job against the Foreman first (logs/runs/runs.jsonl).")
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        print(table(list(reversed(records))))
        print(f"\n{len(records)} run(s) shown. Details: python tests/view_runs.py --job <id>  or  --last")


if __name__ == "__main__":
    main()
