"""Run the DistilBERT phone job under a labelled condition and record one results row.

The conditions follow the experiment factors: distance (RSSI), network load, number of devices,
type of traffic and mobility. Labels you pass are written as given; what the phones *measured*
(RSSI, link speed, free RAM) is joined in from the Foreman's heartbeat log
(``CROWDIO_EXPERIMENT_LOG``), so the row says both what you intended and what the radio did.

  python -m evaluation.experiments.run_condition --sentences 30 --distance 2m --load none \
      --traffic dnn-distilbert --mobility static

With ``--load-level light|heavy`` the phones generate the background load themselves
(see ``android_load.py``) for the duration of the run, and the load label is filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests" / "dnn_partitioning" / "sentiment_onnx_pipeline" / "multi_input_test.py"
DATASET = ROOT / "data" / "sentences.csv"
RESULTS = ROOT / "logs" / "experiments" / "results.csv"
HEARTBEATS = ROOT / "logs" / "experiments" / "heartbeats.jsonl"
FIELDS = ["when", "sentences", "distance", "load", "traffic", "mobility", "devices_online", "notes",
          "seconds", "sentences_per_s", "returned", "accuracy_pct", "failures", "run_id",
          "rssi_mean_dbm", "rssi_min_dbm", "rssi_max_dbm", "link_mbps_mean", "bw_down_mbps_mean",
          "free_ram_min_mb", "tasks_per_phone"]


def api(host: str, path: str):
    return json.load(urllib.request.urlopen(f"http://{host}:8000{path}", timeout=10))


def heartbeat_stats(start: float, end: float) -> Dict[str, Optional[float]]:
    rows: List[dict] = []
    if HEARTBEATS.exists():
        for line in HEARTBEATS.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if start <= row.get("t", 0) <= end:
                rows.append(row)

    def values(key: str) -> List[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    rssi, link, bw, ram = values("rssi_dbm"), values("link_mbps"), values("bw_down_bps"), values("free_ram_mb")
    mean = lambda v: round(statistics.mean(v), 2) if v else None
    return {"rssi_mean_dbm": mean(rssi), "rssi_min_dbm": min(rssi) if rssi else None,
            "rssi_max_dbm": max(rssi) if rssi else None, "link_mbps_mean": mean(link),
            "bw_down_mbps_mean": round(statistics.mean(bw) / 1e6, 2) if bw else None,
            "free_ram_min_mb": min(ram) if ram else None}


def accuracy(predictions: Path, n: int) -> Optional[float]:
    try:
        pred = list(csv.DictReader(open(predictions, encoding="utf-8")))
        gold = list(csv.DictReader(open(DATASET, encoding="utf-8")))[:n]
        hits = sum((p["label"] == "POSITIVE") == (g["label"] == "1") for p, g in zip(pred, gold))
        return round(100.0 * hits / len(pred), 1)
    except (OSError, KeyError, ZeroDivisionError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sentences", type=int, default=30)
    p.add_argument("--distance", default="unspecified", help="label, e.g. 2m, 10m, wall")
    p.add_argument("--load", default="none", help="label for background network load / interference")
    p.add_argument("--traffic", default="dnn-distilbert", help="label for the traffic type")
    p.add_argument("--mobility", default="static", help="label, e.g. static, walking, vehicular")
    p.add_argument("--notes", default="")
    p.add_argument("--host", default="localhost")
    p.add_argument("--load-level", choices=["light", "heavy"],
                   help="start phone-side background load (android_load.py) for the run")
    p.add_argument("--load-serial", action="append",
                   help="adb serial of the phone(s) that generate the load (default: every attached phone)")
    p.add_argument("--isolate", action="store_true", help="force-stop the worker app on the loading phone(s)")
    p.add_argument("--foreman-url", default=os.getenv("CROWDIO_PUBLIC_URL", "http://192.168.1.238:8000"),
                   help="Foreman address as the phones see it (for --load-level)")
    args = p.parse_args()

    loaders: List[str] = []
    if args.load_level:
        from . import android_load

        loaders = args.load_serial or android_load.attached()
        if not loaders:
            print("--load-level needs an adb-attached phone; none found")
            return 1
        for serial in loaders:
            if not android_load.start(serial, args.foreman_url, args.load_level, 600, args.isolate):
                for started in loaders:
                    android_load.stop(started)
                return 1
        if args.load == "none":
            args.load = f"android-{args.load_level} x{len(loaders)}" + (" (isolated)" if args.isolate else "")
        time.sleep(5)  # let the streams reach steady state before timing the job

    try:
        devices = sum(1 for w in api(args.host, "/api/workers") if w.get("status") == "online")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        start = time.time()
        proc = subprocess.run([sys.executable, str(TEST), "--dataset", str(DATASET), "--text-column", "text",
                               "--limit", str(args.sentences), "--host", args.host],
                              capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    finally:
        if loaders:
            from . import android_load

            for serial in loaders:
                android_load.stop(serial)
    end = time.time()
    out = proc.stdout + proc.stderr
    done = re.search(r"completed in ([\d.]+)s; got (\d+) results", out)
    saved = re.search(r"to (\S*sentiment_[\w-]+\.csv)", out)
    seconds = float(done.group(1)) if done else None
    row = {"when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)), "sentences": args.sentences,
           "distance": args.distance, "load": args.load, "traffic": args.traffic, "mobility": args.mobility,
           "devices_online": devices, "notes": args.notes, "seconds": seconds,
           "sentences_per_s": round(args.sentences / seconds, 3) if seconds else None,
           "returned": int(done.group(2)) if done else 0,
           "accuracy_pct": accuracy(ROOT / saved.group(1), args.sentences) if saved else None}
    run = None
    runs = ROOT / "logs" / "runs" / "runs.jsonl"
    if runs.exists():
        lines = runs.read_text(encoding="utf-8").splitlines()
        run = json.loads(lines[-1]) if lines else None
    if run:
        row["run_id"] = run["run_id"][:8]
        row["failures"] = (run.get("failures") or {}).get("count")
        row["tasks_per_phone"] = json.dumps({w["worker_id"][-6:]: w["tasks"] for w in run.get("workers", [])})
    row.update(heartbeat_stats(start, end))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in FIELDS})
    for k in FIELDS:
        print(f"{k:>18}: {row.get(k)}")
    if proc.returncode != 0 or not done:
        print("\nthe job did not complete:\n" + out[-1500:])
        return 1
    print(f"\nrow appended to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
