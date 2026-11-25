"""Regime sweep: when does network-aware scheduling help, and when does it not?

Every condition is run on 30 paired seeds (all schedulers see the identical world, tasks and report
noise). Conditions vary one axis at a time around a base case, plus a transfer-share x heterogeneity
grid. Gains are paired per seed against FIFO; a bootstrap over seeds gives the confidence interval of
the median.

  python -m evaluation.netsim.sweep run                  # about 15-20 minutes on 8 cores
  python -m evaluation.netsim.sweep run --quick          # 4 seeds, a handful of conditions
  python -m evaluation.netsim.sweep summarize            # tables + figures from the saved runs

Results are appended to ``evaluation_results/netsim/runs.csv`` as they finish, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import NetParams
from .sim import SCHEDULER_NAMES, Reporting, Workload, calibrate_curve, run_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation_results" / "netsim"
FIGS = ROOT / "reports" / "network_study"
SEEDS = 30
FIELDS = ["cond_id", "seed", "scheduler", "makespan_s", "est_error", "mean_task_s", "imputed_share"]
MBPS_LABEL = {"inf": "radio only"}

HETERO = {  # (area radius m, ring, backhaul spread)
    "identical": dict(area_m=12.0, ring=True, spread=1.0),
    "moderate": dict(area_m=40.0, ring=False, spread=5.0),
    "extreme": dict(area_m=80.0, ring=False, spread=50.0),
}
RATIOS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


LABEL_ORDER: Dict[str, List[str]] = defaultdict(list)  # axis -> labels in the order they were defined


class Cond:
    def __init__(self, params: NetParams, workload: Workload, rep: Reporting, seeds: int = SEEDS) -> None:
        self.params, self.workload, self.rep, self.seeds = params, workload, rep, seeds
        parts = [asdict(params), asdict(workload), asdict(rep)]
        if parts[1].get("sched_window") == 24:
            del parts[1]["sched_window"]  # the default: keeps the ids of conditions run before the field existed
        self.cond_id = hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:8]
        self.tags: List[Tuple[str, str]] = []  # (axis, label) pairs that point at this condition


BASE_PARAMS = NetParams(devices=12, speed_mps=1.4, backhaul_mbps=20.0, contention=0.0, **HETERO["moderate"])
BASE_WORKLOAD = Workload(tc_ratio=0.5)
BASE_REP = Reporting(bw_report="heartbeat:10")


def build_conditions(quick: bool = False) -> List[Cond]:
    by_id: Dict[str, Cond] = {}
    LABEL_ORDER.clear()

    def add(axis: str, label: str, *, params=None, workload=None, rep=None, seeds=SEEDS) -> None:
        LABEL_ORDER[axis].append(label)
        cond = Cond(params or BASE_PARAMS, workload or BASE_WORKLOAD, rep or BASE_REP, seeds)
        cond = by_id.setdefault(cond.cond_id, cond)
        cond.seeds = min(cond.seeds, seeds)
        cond.tags.append((axis, label))

    P, W, R = BASE_PARAMS, BASE_WORKLOAD, BASE_REP
    for r in RATIOS:
        add("transfer/compute ratio", f"{r:g}", workload=replace(W, tc_ratio=r))
    for label, size in (("20 KB", 20e3), ("200 KB", 200e3), ("2 MB", 2e6), ("20 MB", 20e6)):
        add("payload size (ratio fixed at 0.5)", label, workload=replace(W, payload_bytes=size))
    for k in (1, 2, 4, 8, 16):
        add("tasks per device (queue depth)", str(k), workload=replace(W, tasks_per_device=float(k)))
    for name, h in HETERO.items():
        add("heterogeneity", name, params=replace(P, **h))
    for label, speed in (("static", 0.0), ("walking 1.4 m/s", 1.4), ("vehicular 20 m/s", 20.0)):
        add("mobility", label, params=replace(P, speed_mps=speed))
    for label, rep in (("every 1 s", "heartbeat:1"), ("every 10 s", "heartbeat:10"), ("every 60 s", "heartbeat:60"),
                       ("after transfers only", "transfers")):
        add("bandwidth report freshness", label, rep=Reporting(bw_report=rep))
    for k in (3, 10, 30, 100):
        add("fleet size (window 24)", str(k), params=replace(P, devices=k), seeds=10 if k >= 100 else SEEDS)
    for k in (3, 10, 30, 100):  # the scheduler maps `window` tasks per call: let it see as many tasks as there are phones
        add("fleet size (window = fleet)", str(k), params=replace(P, devices=k), workload=replace(W, sched_window=max(24, k)),
            seeds=5 if k >= 100 else SEEDS)  # a 100-device run takes minutes: 5 paired seeds are enough to see a 19% gap move
    for k in (1, 2, 4, 8):  # no bandwidth measurement exists yet: the one case where RSSI should help
        add("cold start (reports only after transfers)", f"{k} tasks/device", workload=replace(W, tasks_per_device=float(k)),
            rep=Reporting(bw_report="transfers"))
    for label, c in (("none", 0.0), ("moderate (30% airtime)", 0.3), ("heavy (60% airtime)", 0.6)):
        add("contention from other devices", label, params=replace(P, contention=c))
    for label, b in (("5 Mbit/s", 5.0), ("20 Mbit/s", 20.0), ("100 Mbit/s", 100.0), ("none (radio only)", math.inf)):
        add("backhaul cap (median)", label, params=replace(P, backhaul_mbps=b))
    for label, ev in (("none", {}), ("handoff outages 2/min", dict(outage_per_min=2.0)),
                      ("interference bursts 2/min", dict(burst_per_min=2.0)),
                      ("25% of phones throttled once", dict(drop_frac=0.25))):
        add("network events", label, params=replace(P, **ev))
    for hname, h in HETERO.items():  # the regime map: transfer share x heterogeneity
        for r in RATIOS:
            add("regime map", f"{hname}|{r:g}", params=replace(P, **h), workload=replace(W, tc_ratio=r))
    conds = list(by_id.values())
    if quick:
        keep = {"transfer/compute ratio", "bandwidth report freshness"}
        conds = [c for c in conds if any(a in keep for a, _ in c.tags)][:6]
        for c in conds:
            c.seeds = 4
    return conds


# ------------------------------------------------------------------ running
def _job(args) -> List[dict]:
    cond_json, seed, schedulers, curve = args
    cond = _from_json(cond_json)
    results = run_seed(cond.params, cond.workload, cond.rep, seed, schedulers, curve)
    return [{"cond_id": cond.cond_id, "seed": seed, "scheduler": r.scheduler, "makespan_s": round(r.makespan_s, 4),
             "est_error": "" if math.isnan(r.est_error) else round(r.est_error, 4),
             "mean_task_s": round(r.mean_task_s, 4),
             "imputed_share": "" if math.isnan(r.imputed_share) else round(r.imputed_share, 4)} for r in results]


def _to_json(cond: Cond) -> dict:
    return {"params": asdict(cond.params), "workload": asdict(cond.workload), "rep": asdict(cond.rep), "seeds": cond.seeds}


def _from_json(data: dict) -> Cond:
    return Cond(NetParams(**data["params"]), Workload(**data["workload"]), Reporting(**data["rep"]), data["seeds"])


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {(r["cond_id"], int(r["seed"])) for r in csv.DictReader(handle)}


def run_all(conds: Sequence[Cond], workers: int, schedulers: Sequence[str], out: Path = OUT,
            register: Optional[Sequence[Cond]] = None) -> None:
    """Run ``conds``; ``register`` (default: ``conds``) is the full list written to conditions.json, so a filtered run keeps the rest."""
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONHASHSEED"] = "0"  # worker processes inherit it: FIFO's set order is then reproducible
    curve = calibrate_curve(BASE_PARAMS, seeds=range(9000, 9005))
    (out / "conditions.json").write_text(json.dumps(
        [{"cond_id": c.cond_id, "tags": c.tags, "seeds": c.seeds, **_to_json(c)} for c in (register or conds)], indent=1, default=str),
        encoding="utf-8")
    (out / "rssi_curve.json").write_text(json.dumps(asdict(curve), indent=1), encoding="utf-8")
    runs = out / "runs.csv"
    done = _done(runs)
    jobs = [(_to_json(c), s, tuple(schedulers), curve) for c in conds for s in range(c.seeds)
            if (c.cond_id, s) not in done]
    print(f"{len(conds)} conditions, {len(jobs)} paired runs to do ({len(done)} already done), curve {curve.db_per_decade:.1f} dB/decade "
          f"from {curve.samples} samples")
    new_file = not runs.exists()
    started = time.time()
    with open(runs, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, j) for j in jobs]
            for n, future in enumerate(as_completed(futures), 1):
                writer.writerows(future.result())
                handle.flush()
                if n % 25 == 0 or n == len(futures):
                    rate = n / (time.time() - started)
                    print(f"  {n}/{len(futures)} done, {rate:.2f}/s, about {(len(futures) - n) / max(rate, 1e-9) / 60:.1f} min left",
                          flush=True)


# ------------------------------------------------------------------ statistics
def bootstrap_median_ci(values: Sequence[float], n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """(median, lo, hi): percentile bootstrap over the paired per-seed values."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    return float(np.median(x)), float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def load_runs(path: Path = OUT / "runs.csv") -> Dict[str, Dict[str, Dict[int, dict]]]:
    data: Dict[str, Dict[str, Dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            data[r["cond_id"]][r["scheduler"]][int(r["seed"])] = {
                "makespan": float(r["makespan_s"]), "err": float(r["est_error"]) if r["est_error"] else float("nan"),
                "imputed": float(r["imputed_share"]) if r["imputed_share"] else float("nan")}
    return data


def _paired(a: Dict[int, dict], b: Dict[int, dict], key: str = "makespan") -> List[float]:
    seeds = sorted(set(a) & set(b))
    return [1.0 - a[s][key] / b[s][key] for s in seeds]  # share of b's time saved by a


def summarise(runs: Dict[str, Dict[str, Dict[int, dict]]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for cond_id, by_sched in runs.items():
        fifo = by_sched.get("fifo", {})
        entry = {"n_seeds": len(fifo), "schedulers": {}}
        for sched, per_seed in by_sched.items():
            med, lo, hi = bootstrap_median_ci(_paired(per_seed, fifo)) if sched != "fifo" else (0.0, 0.0, 0.0)
            gap = [per_seed[s]["makespan"] / by_sched["oracle"][s]["makespan"] - 1.0 for s in per_seed
                   if s in by_sched.get("oracle", {})]
            entry["schedulers"][sched] = {
                "median_makespan": float(np.median([v["makespan"] for v in per_seed.values()])),
                "gain_vs_fifo": {"median": med, "lo": lo, "hi": hi},
                "win_rate_vs_fifo": float(np.mean([g > 0 for g in _paired(per_seed, fifo)])) if sched != "fifo" and fifo else None,
                "gap_to_oracle_median": float(np.median(gap)) if gap else None,
                "est_error_median": float(np.nanmedian([v["err"] for v in per_seed.values()]))
                if any(not math.isnan(v["err"]) for v in per_seed.values()) else None,
                "imputed_share_mean": float(np.nanmean([v["imputed"] for v in per_seed.values()]))
                if any(not math.isnan(v["imputed"]) for v in per_seed.values()) else None,
            }
        if "bandwidth_aware" in by_sched and "rssi_aware_cal" in by_sched:
            m, lo, hi = bootstrap_median_ci(_paired(by_sched["rssi_aware_cal"], by_sched["bandwidth_aware"]))
            entry["rssi_cal_vs_bandwidth"] = {"median": m, "lo": lo, "hi": hi}
        if "bandwidth_aware" in by_sched and "rssi_aware" in by_sched:
            m, lo, hi = bootstrap_median_ci(_paired(by_sched["rssi_aware"], by_sched["bandwidth_aware"]))
            entry["rssi_default_vs_bandwidth"] = {"median": m, "lo": lo, "hi": hi}
        out[cond_id] = entry
    return out


# ------------------------------------------------------------------ report
def _fmt(g: dict) -> str:
    sig = "" if g["lo"] <= 0 <= g["hi"] else "*"
    return f"{g['median'] * 100:+.1f}% [{g['lo'] * 100:+.1f}, {g['hi'] * 100:+.1f}]{sig}"


def ordered_axes(conds: List[dict]) -> Dict[str, List[Tuple[str, str]]]:
    """axis -> [(label, cond_id)] in the order the labels were defined (not the order they were first run)."""
    build_conditions()  # refills LABEL_ORDER
    axes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for c in conds:
        for axis, label in c["tags"]:
            axes[axis].append((label, c["cond_id"]))
    for axis, rows in axes.items():
        order = LABEL_ORDER.get(axis, [])
        rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else len(order))
    return dict(sorted(axes.items(), key=lambda kv: list(LABEL_ORDER).index(kv[0]) if kv[0] in LABEL_ORDER else 99))


def write_tables(conds: List[dict], summary: Dict[str, dict], path: Path) -> None:
    axes = ordered_axes(conds)
    cols = ["round_robin", "least_loaded", "bandwidth_aware", "adaptive", "rssi_aware", "rssi_aware_cal", "oracle"]
    lines = ["# Network-awareness sweep: tables (generated)\n",
             "Median paired makespan gain over FIFO per seed, with a 95% bootstrap CI of the median. Positive = shorter than FIFO. "
             "`*` marks a CI that excludes zero. Rows with fewer than 30 seeds say so.\n"]
    for axis, rows in axes.items():
        if axis == "regime map":
            continue
        lines += [f"\n## {axis}\n", "| condition | seeds | FIFO makespan (s) | " + " | ".join(cols) + " |",
                  "| --- | --- | --- | " + " | ".join("---" for _ in cols) + " |"]
        for label, cid in rows:
            s = summary.get(cid)
            if not s:
                continue
            lines.append(f"| {label} | {s['n_seeds']} | {s['schedulers']['fifo']['median_makespan']:.1f} | " + " | ".join(
                _fmt(s["schedulers"][c]["gain_vs_fifo"]) if c in s["schedulers"] else "" for c in cols) + " |")
    lines += ["\n## RSSI vs bandwidth-only (paired, per condition)\n",
              "Median paired gain of the RSSI-aware scheduler over bandwidth-aware.\n",
              "| axis | condition | default curve | calibrated curve | bandwidth est. error | RSSI-cal est. error |", "| --- | --- | --- | --- | --- | --- |"]
    for axis, rows in axes.items():
        if axis == "regime map":
            continue
        for label, cid in rows:
            s = summary.get(cid)
            if not s or "rssi_cal_vs_bandwidth" not in s:
                continue
            bw, rc = s["schedulers"]["bandwidth_aware"], s["schedulers"]["rssi_aware_cal"]
            lines.append(f"| {axis} | {label} | {_fmt(s['rssi_default_vs_bandwidth'])} | {_fmt(s['rssi_cal_vs_bandwidth'])} | "
                         f"{bw['est_error_median']:.2f} | {rc['est_error_median']:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_all(conds: List[dict], summary: Dict[str, dict], figs: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    colours = {"round_robin": "#999999", "least_loaded": "#bbbbbb", "bandwidth_aware": "#0072B2", "adaptive": "#56B4E9",
               "rssi_aware": "#E69F00", "rssi_aware_cal": "#D55E00", "oracle": "#009E73"}
    names = {"round_robin": "round-robin", "least_loaded": "least-loaded", "bandwidth_aware": "bandwidth-aware",
             "adaptive": "adaptive", "rssi_aware": "RSSI (default curve)", "rssi_aware_cal": "RSSI (calibrated)", "oracle": "oracle"}
    axes = ordered_axes(conds)
    written: List[Path] = []

    # ---- regime map
    grid = {tuple(label.split("|")): cid for label, cid in axes.get("regime map", [])}
    hetero_names, ratios = list(HETERO), [f"{r:g}" for r in RATIOS]
    if grid:
        panels = ["bandwidth_aware", "rssi_aware_cal", "oracle"]
        fig, ax = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.6), sharey=True)
        for a, sched in zip(ax, panels):
            z = np.full((len(hetero_names), len(ratios)), np.nan)
            sig = np.zeros_like(z, dtype=bool)
            for i, h in enumerate(hetero_names):
                for j, r in enumerate(ratios):
                    cid = grid.get((h, r))
                    if cid in summary:
                        g = summary[cid]["schedulers"][sched]["gain_vs_fifo"]
                        z[i, j], sig[i, j] = g["median"] * 100, not (g["lo"] <= 0 <= g["hi"])
            im = a.imshow(z, cmap="RdBu", vmin=-30, vmax=30, aspect="auto")
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if not np.isnan(z[i, j]):
                        a.text(j, i, f"{z[i, j]:+.0f}" + ("" if sig[i, j] else "\nn.s."), ha="center", va="center", fontsize=8)
            a.set_xticks(range(len(ratios)), ratios)
            a.set_yticks(range(len(hetero_names)), hetero_names)
            a.set_xlabel("transfer time / compute time")
            a.set_title(names[sched])
        fig.suptitle("Median makespan gain over FIFO (%), paired over 30 seeds; n.s. = 95% CI includes 0", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, label="gain (%)")
        path = figs / "regime_map.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # ---- one-at-a-time axes
    names_axes = [a for a in axes if a != "regime map"]
    cols = 3
    rows_n = math.ceil(len(names_axes) / cols)
    fig, ax = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 3.4 * rows_n))
    ax = np.atleast_2d(ax)
    for k, axis in enumerate(names_axes):
        a = ax[k // cols][k % cols]
        pts = [(label, cid) for label, cid in axes[axis] if cid in summary]
        x = np.arange(len(pts))
        for sched in colours:
            if sched in ("adaptive", "least_loaded"):
                continue  # identical to bandwidth-aware / FIFO in this simulator (no failures, battery or load signals)
            y = np.array([summary[cid]["schedulers"][sched]["gain_vs_fifo"]["median"] * 100 for _, cid in pts])
            lo = np.array([summary[cid]["schedulers"][sched]["gain_vs_fifo"]["lo"] * 100 for _, cid in pts])
            hi = np.array([summary[cid]["schedulers"][sched]["gain_vs_fifo"]["hi"] * 100 for _, cid in pts])
            style = "--" if sched in ("round_robin", "least_loaded") else "-"
            a.plot(x, y, style, marker="o", ms=3, color=colours[sched], label=names[sched], lw=1.4)
            if sched in ("bandwidth_aware", "rssi_aware_cal", "oracle"):
                a.fill_between(x, lo, hi, color=colours[sched], alpha=0.15, lw=0)
        a.axhline(0, color="black", lw=0.6)
        a.set_xticks(x, [label for label, _ in pts], rotation=20, ha="right", fontsize=7)
        a.set_title(axis, fontsize=9)
        a.set_ylabel("gain over FIFO (%)", fontsize=8)
        a.tick_params(axis="y", labelsize=7)
    for k in range(len(names_axes), rows_n * cols):
        ax[k // cols][k % cols].axis("off")
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = figs / "axes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # ---- estimate error vs report freshness
    if "bandwidth report freshness" in axes:
        pts = [(label, cid) for label, cid in axes["bandwidth report freshness"] if cid in summary]
        fig, ax2 = plt.subplots(figsize=(6.4, 3.4))
        width = 0.2
        for n, sched in enumerate(["bandwidth_aware", "rssi_aware", "rssi_aware_cal", "adaptive"]):
            vals = [summary[cid]["schedulers"][sched]["est_error_median"] or 0.0 for _, cid in pts]
            ax2.bar(np.arange(len(pts)) + (n - 1.5) * width, vals, width, color=colours[sched], label=names[sched])
        ax2.set_xticks(range(len(pts)), [label for label, _ in pts], fontsize=8)
        ax2.set_ylabel("median |ln(estimated / true downlink)|", fontsize=8)
        ax2.set_title("How wrong the scheduler's downlink estimate is at assignment", fontsize=9)
        ax2.legend(fontsize=7, frameon=False)
        path = figs / "estimate_error.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def summarize(out: Path = OUT, figs: Path = FIGS) -> None:
    conds = json.loads((out / "conditions.json").read_text(encoding="utf-8"))
    runs = load_runs(out / "runs.csv")
    summary = summarise(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_tables(conds, summary, out / "tables.md")
    paths = plot_all(conds, summary, figs)
    print(f"summary of {len(summary)} conditions -> {out / 'summary.json'}, {out / 'tables.md'}; figures: " + ", ".join(p.name for p in paths))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    run.add_argument("--quick", action="store_true", help="4 seeds, a few conditions (smoke test; writes to a separate folder)")
    run.add_argument("--schedulers", default=",".join(SCHEDULER_NAMES))
    run.add_argument("--axis", default="", help="only run conditions whose axis name contains this text")
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "run":
        out = OUT.with_name("netsim_quick") if args.quick else OUT
        everything = build_conditions(args.quick)
        conds = [c for c in everything if not args.axis or any(args.axis in a for a, _ in c.tags)]
        run_all(conds, args.workers, args.schedulers.split(","), out, register=everything)
        summarize(out, FIGS.with_name("network_study_quick") if args.quick else FIGS)
    else:
        summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
