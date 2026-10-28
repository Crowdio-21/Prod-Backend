"""Metrics from JSONL event logs (simulated or live).

Reports: total completion time (reconfiguration and inference separately),
steady-state latency/period, switch count, stall time, wasted switches
(A -> B -> A inside the controller window), regret versus an oracle log,
route shares and flips, bytes moved, planner time, deadline hit rate, and the
control-plane overhead used by no-fault tax mode.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))]


def compute_metrics(events: Iterable[Dict[str, Any]], *, switch_window_s: float = 300.0,
                    oracle_total: Optional[float] = None) -> Dict[str, Any]:
    events = list(events)
    start = next((e["ts"] for e in events if e["event"] == "run_start"), events[0]["ts"] if events else 0.0)
    ends = [e for e in events if e["event"] == "request_end"]
    starts = {e.get("request_id"): e["ts"] for e in events if e["event"] == "request_start"}
    completion_ts = sorted(e["ts"] for e in ends)
    switch_starts = [e for e in events if e["event"] == "switch_start" and e.get("reason") != "initial"]
    switch_ends = [e for e in events if e["event"] == "switch_end"]
    reconfig_s = sum(float(e.get("seconds") or 0.0) for e in switch_ends if not e.get("asynchronous"))
    initial = [e for e in events if e["event"] == "switch_start" and e.get("reason") == "initial"]
    if initial and switch_ends:
        reconfig_s -= float(switch_ends[0].get("seconds") or 0.0) if switch_ends[0]["ts"] >= initial[0]["ts"] else 0.0
    total = (completion_ts[-1] - start) if completion_ts else None
    latencies = [e["ts"] - starts[e["request_id"]] for e in ends if e.get("request_id") in starts]
    gaps = [b - a for a, b in zip(completion_ts, completion_ts[1:])]
    # A switch is wasted when it returns to the assignment held two switches earlier
    # within the controller window (A -> B -> A).
    all_switches = [e for e in events if e["event"] == "switch_start"]
    targets = [tuple(map(tuple, e.get("target") or [])) for e in all_switches]
    wasted = sum(
        1 for i in range(2, len(targets))
        if targets[i] == targets[i - 2] and all_switches[i]["ts"] - all_switches[i - 2]["ts"] <= switch_window_s
    )
    routes: Dict[str, int] = {}
    for e in events:
        if e["event"] in ("route_choice", "route_flip"):
            routes[e.get("mode", "?")] = routes.get(e.get("mode", "?"), 0) + 1
    flips = sum(1 for e in events if e["event"] == "route_flip")
    planner = [float(e.get("planner_s") or 0.0) for e in events if e["event"] == "planner_call"]
    bytes_moved = 0
    for e in events:
        if e["event"] == "prefetch" and e.get("status") == "done":
            bytes_moved += int(e.get("bytes") or 0)
        if e.get("kind") == "load_result":
            bytes_moved += int((e.get("timings") or {}).get("bytes_fetched") or 0)
    run_end = next((e for e in reversed(events) if e["event"] == "run_end"), {})
    if "bytes_moved" in run_end:
        bytes_moved = int(run_end["bytes_moved"])
    duration_min = max(1e-9, ((events[-1]["ts"] - start) if events else 0.0) / 60.0)
    control_bytes = run_end.get("control_bytes")
    heartbeats = [e for e in events if e["event"] == "heartbeat"]
    cpu = [float((e.get("state") or {}).get("cpu") or 0.0) for e in heartbeats]
    battery_by_worker: Dict[str, List[float]] = {}
    for e in heartbeats:
        b = (e.get("state") or {}).get("battery")
        if b is not None:
            battery_by_worker.setdefault(e.get("worker"), []).append(float(b))
    metrics = {
        "total_completion_s": total,
        "reconfiguration_s": max(0.0, reconfig_s),
        "inference_s": None if total is None else max(0.0, total - max(0.0, reconfig_s)),
        "requests_completed": len(ends),
        "steady_period_s": statistics.median(gaps) if gaps else None,
        "throughput_rps": (1.0 / statistics.median(gaps)) if gaps and statistics.median(gaps) > 0 else None,
        "latency_p50_s": _pct(latencies, 0.5),
        "latency_p95_s": _pct(latencies, 0.95),
        "switches": run_end.get("switches", len(switch_starts)),
        "stall_s": run_end.get("stall_time", reconfig_s),
        "wasted_switches": run_end.get("wasted_switches", wasted),
        "route_shares": run_end.get("route_shares") or {k: v / max(1, sum(routes.values())) for k, v in routes.items()},
        "route_flips": run_end.get("route_flips", flips),
        "bytes_moved": bytes_moved,
        "planner_calls": len(planner),
        "planner_time_s": sum(planner),
        "planner_p95_s": _pct(planner, 0.95),
        "deadline_hit_rate": run_end.get("deadline_hit_rate"),
        "control_bytes": control_bytes,
        "control_bytes_per_request": (control_bytes / len(ends)) if control_bytes and ends else None,
        "control_bytes_per_min": (control_bytes / duration_min) if control_bytes else None,
        "worker_cpu_mean": statistics.mean(cpu) if cpu else None,
        "worker_battery_drop": {w: b[0] - b[-1] for w, b in battery_by_worker.items() if len(b) > 1} or None,
        "foreman_cpu_s": run_end.get("foreman_cpu_s"),
        "prefetch_issued": sum(1 for e in events if e["event"] == "prefetch" and e.get("status") in ("issued", "done")),
        "prefetch_used": sum(1 for e in events if e["event"] == "prefetch_used"),
    }
    if oracle_total is not None and total is not None:
        metrics["regret_s"] = total - oracle_total
    return metrics
