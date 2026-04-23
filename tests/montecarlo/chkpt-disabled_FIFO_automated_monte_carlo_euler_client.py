#!/usr/bin/env python3

import asyncio
import sys
import os
import json
import time
import csv
from datetime import datetime

# Add root directory to Python path (go up two levels from tests/montecarlo/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crowdio import crowdio_connect, crowdio_map, crowdio_disconnect, CROWDio


@CROWDio.task(
    checkpoint=False,
    checkpoint_interval=5.0,  # Checkpoint every 5 seconds
    checkpoint_state=[
        "trials_completed",
        "total_count",
        "estimated_e",
        "progress_percent",
    ],
)
def monte_carlo_euler_worker(num_trials):
    """
    Worker function to perform Monte Carlo trials for estimating e
    WITH DECLARATIVE CHECKPOINTING - PURE LOGIC, NO RESUME CODE!

    Args:
        num_trials: Number of simulation trials to run

    Returns:
        Dictionary containing trial results and statistics

    Note:
        The @CROWDio.task decorator enables automatic checkpointing:
        - State variables are captured automatically via frame introspection
        - TRANSPARENT RESUME - framework handles everything automatically!
        - Just write your pure algorithm logic
        - Include 'progress_percent' in checkpoint_state for progress tracking

        DEVELOPER WRITES PURE LOGIC - NO RESUME CODE NEEDED!
        The framework automatically:
        1. Captures checkpoint_state variables during execution
        2. On resume, injects saved values into variables
        3. Adjusts loop ranges to continue from checkpoint position
    """
    import random
    import time

    start = time.time()
    trials_completed = 0
    total_count = 0
    estimated_e = 0.0
    progress_percent = 0.0  # Include in checkpoint_state for progress tracking

    random.seed()  # Ensure different seeds on different workers

    try:
        # Simple loop - framework automatically adjusts range on resume!
        for i in range(num_trials):
            random_sum = 0.0
            count = 0

            # Keep adding random numbers until sum exceeds 1
            while random_sum < 1.0:
                random_sum += random.random()
                count += 1

            total_count += count
            trials_completed = i + 1

            # Update progress and estimate - these are captured automatically!
            progress_percent = (trials_completed / num_trials) * 100
            estimated_e = (
                total_count / trials_completed if trials_completed > 0 else 0.0
            )

        latency_ms = int((time.time() - start) * 1000)

        result = {
            "num_trials": num_trials,
            "total_count": total_count,
            "estimated_e": round(estimated_e, 6),
            "latency_ms": latency_ms,
            "status": "success",
        }

        print(
            f"[Worker] Completed {num_trials:,} trials | e ≈ {estimated_e:.6f} | Total time: {latency_ms/1000:.1f}s"
        )
        return result

    except Exception as e:
        import traceback

        traceback.print_exc()

        latency_ms = int((time.time() - start) * 1000)

        return {
            "num_trials": num_trials,
            "estimated_e": 0.0,
            "latency_ms": latency_ms,
            "status": "error",
            "error": str(e),
        }


# =========================================================
# 📊 RESULT AGGREGATION
# =========================================================
def aggregate_monte_carlo_results(results):
    """
    Aggregate results from distributed Monte Carlo workers

    Args:
        results: List of worker results

    Returns:
        Dictionary with aggregated statistics
    """
    parsed = []

    for r in results:
        if isinstance(r, dict):
            parsed.append(r)
        elif isinstance(r, str):
            try:
                parsed.append(json.loads(r))
            except:
                try:
                    import ast

                    parsed.append(ast.literal_eval(r))
                except:
                    parsed.append({"status": "error", "error": "Unparseable result"})
        else:
            parsed.append({"status": "error", "error": "Unknown result type"})

    valid = [r for r in parsed if r.get("status") == "success"]

    if not valid:
        return {
            "final_estimate": 0.0,
            "total_trials": 0,
            "worker_count": 0,
            "error_count": len(results),
            "error": "All results failed or could not be parsed",
            "worker_results": [],
        }

    # Calculate aggregate statistics
    total_count = sum(r["total_count"] for r in valid)
    total_trials = sum(r["num_trials"] for r in valid)

    # Overall estimate of e
    final_estimate = total_count / total_trials if total_trials > 0 else 0.0

    # Calculate average of worker estimates
    worker_estimates = [r["estimated_e"] for r in valid]
    avg_worker_estimate = sum(worker_estimates) / len(worker_estimates)

    # Calculate standard deviation
    mean = avg_worker_estimate
    variance = sum((x - mean) ** 2 for x in worker_estimates) / len(worker_estimates)
    std_dev = variance**0.5

    # Latency statistics
    latencies = [r["latency_ms"] for r in valid]

    # Calculate error from true value of e
    true_e = 2.718281828459045
    error = abs(final_estimate - true_e)
    error_percentage = (error / true_e) * 100

    return {
        "final_estimate": round(final_estimate, 10),
        "true_e": round(true_e, 10),
        "absolute_error": round(error, 10),
        "error_percentage": round(error_percentage, 6),
        "avg_worker_estimate": round(avg_worker_estimate, 10),
        "std_dev": round(std_dev, 6),
        "min_estimate": round(min(worker_estimates), 6),
        "max_estimate": round(max(worker_estimates), 6),
        "total_trials": total_trials,
        "worker_count": len(valid),
        "error_count": len(results) - len(valid),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "total_latency_ms": sum(latencies),
        "worker_results": valid,
    }


# =========================================================
# 🚀 DISTRIBUTED EXECUTION
# =========================================================
async def run_distributed_monte_carlo_euler(
    total_trials, num_workers=None, foreman_host="localhost"
):
    """
    Run distributed Monte Carlo simulation to estimate Euler's number.
    Returns a metrics dict with job/task timing, throughput, and per-worker details.
    """
    await crowdio_connect(foreman_host, 9000)

    if num_workers is None:
        num_workers = 4

    trials_per_worker = total_trials // num_workers
    task_inputs = [trials_per_worker] * num_workers

    remainder = total_trials % num_workers
    if remainder > 0:
        task_inputs[-1] += remainder

    start_time = time.time()
    results = await crowdio_map(monte_carlo_euler_worker, task_inputs)
    job_exec_time_s = time.time() - start_time

    aggregated = aggregate_monte_carlo_results(results)

    # Per-worker task times (seconds)
    worker_task_times_ms = [
        r["latency_ms"] for r in aggregated.get("worker_results", [])
    ]
    worker_task_times_s = [t / 1000.0 for t in worker_task_times_ms]
    avg_task_time_s = (
        (sum(worker_task_times_s) / len(worker_task_times_s))
        if worker_task_times_s
        else 0.0
    )
    throughput = total_trials / job_exec_time_s if job_exec_time_s > 0 else 0.0

    await crowdio_disconnect()

    return {
        "total_trials": total_trials,
        "num_workers": num_workers,
        "job_exec_time_s": round(job_exec_time_s, 3),
        "worker_task_times_s": [round(t, 3) for t in worker_task_times_s],
        "avg_task_time_s": round(avg_task_time_s, 3),
        "throughput": round(throughput, 2),
        "aggregated": aggregated,
    }


# =========================================================
# 📈 BENCHMARK GRID RUNNER
# =========================================================
def format_trials(n):
    """Human-readable trial count label."""
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    elif n >= 1_000_000:
        return f"{n // 1_000_000}M"
    elif n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def print_separator(width=140):
    print("=" * width)


def print_run_result(metrics):
    """Pretty-print a single run's metrics."""
    total = metrics["total_trials"]
    nw = metrics["num_workers"]
    jet = metrics["job_exec_time_s"]
    att = metrics["avg_task_time_s"]
    tp = metrics["throughput"]
    wtt = metrics["worker_task_times_s"]

    print(
        f"\n  Trials: {format_trials(total):>5s}  |  Workers: {nw:>2d}  |  "
        f"Job Exec Time: {jet:>9.3f}s  |  Avg Task Time: {att:>9.3f}s  |  "
        f"Throughput: {tp:>14,.2f} trials/s"
    )
    # Per-worker breakdown
    worker_strs = [f"W{i+1}={t:.3f}s" for i, t in enumerate(wtt)]
    # Print in rows of 6
    for row_start in range(0, len(worker_strs), 6):
        chunk = worker_strs[row_start : row_start + 6]
        print(f"    Task times: {', '.join(chunk)}")


def print_summary_table(all_metrics):
    """Print a summary table of all benchmark runs."""
    print("\n")
    print_separator()
    print("  BENCHMARK SUMMARY TABLE")
    print_separator()
    header = (
        f"{'Trials':>8s}  {'Workers':>7s}  {'Job Exec (s)':>13s}  "
        f"{'Avg Task (s)':>13s}  {'Min Task (s)':>13s}  {'Max Task (s)':>13s}  "
        f"{'Throughput (trials/s)':>22s}"
    )
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        wtt = m["worker_task_times_s"]
        min_t = min(wtt) if wtt else 0.0
        max_t = max(wtt) if wtt else 0.0
        print(
            f"{format_trials(m['total_trials']):>8s}  {m['num_workers']:>7d}  "
            f"{m['job_exec_time_s']:>13.3f}  {m['avg_task_time_s']:>13.3f}  "
            f"{min_t:>13.3f}  {max_t:>13.3f}  {m['throughput']:>22,.2f}"
        )

    print_separator()


def print_aggregate_averages(all_metrics):
    """Print averages grouped by trials and by workers."""
    from collections import defaultdict

    # --- Averages grouped by num_workers ---
    by_workers = defaultdict(list)
    for m in all_metrics:
        by_workers[m["num_workers"]].append(m)

    print("\n  AVERAGES BY NUM WORKERS")
    print("-" * 90)
    header = f"{'Workers':>7s}  {'Avg Job Exec (s)':>17s}  {'Avg Task Time (s)':>18s}  {'Avg Throughput (trials/s)':>26s}"
    print(header)
    print("-" * len(header))
    for nw in sorted(by_workers):
        runs = by_workers[nw]
        avg_jet = sum(r["job_exec_time_s"] for r in runs) / len(runs)
        avg_att = sum(r["avg_task_time_s"] for r in runs) / len(runs)
        avg_tp = sum(r["throughput"] for r in runs) / len(runs)
        print(f"{nw:>7d}  {avg_jet:>17.3f}  {avg_att:>18.3f}  {avg_tp:>26,.2f}")

    # --- Averages grouped by total_trials ---
    by_trials = defaultdict(list)
    for m in all_metrics:
        by_trials[m["total_trials"]].append(m)

    print(f"\n  AVERAGES BY NUM TRIALS")
    print("-" * 90)
    header = f"{'Trials':>8s}  {'Avg Job Exec (s)':>17s}  {'Avg Task Time (s)':>18s}  {'Avg Throughput (trials/s)':>26s}"
    print(header)
    print("-" * len(header))
    for nt in sorted(by_trials):
        runs = by_trials[nt]
        avg_jet = sum(r["job_exec_time_s"] for r in runs) / len(runs)
        avg_att = sum(r["avg_task_time_s"] for r in runs) / len(runs)
        avg_tp = sum(r["throughput"] for r in runs) / len(runs)
        print(
            f"{format_trials(nt):>8s}  {avg_jet:>17.3f}  {avg_att:>18.3f}  {avg_tp:>26,.2f}"
        )

    # --- Grand averages ---
    grand_jet = sum(m["job_exec_time_s"] for m in all_metrics) / len(all_metrics)
    grand_att = sum(m["avg_task_time_s"] for m in all_metrics) / len(all_metrics)
    grand_tp = sum(m["throughput"] for m in all_metrics) / len(all_metrics)
    print(f"\n  GRAND AVERAGES ({len(all_metrics)} runs)")
    print(f"    Avg Job Exec Time : {grand_jet:.3f}s")
    print(f"    Avg Task Time     : {grand_att:.3f}s")
    print(f"    Avg Throughput    : {grand_tp:,.2f} trials/s")
    print_separator()


def export_results_csv(all_metrics, filepath):
    """Export all benchmark metrics to a CSV file."""
    fieldnames = [
        "trials",
        "trials_label",
        "num_workers",
        "job_exec_time_s",
        "avg_task_time_s",
        "min_task_time_s",
        "max_task_time_s",
        "throughput_trials_per_s",
        "worker_task_times_s",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            wtt = m["worker_task_times_s"]
            writer.writerow(
                {
                    "trials": m["total_trials"],
                    "trials_label": format_trials(m["total_trials"]),
                    "num_workers": m["num_workers"],
                    "job_exec_time_s": m["job_exec_time_s"],
                    "avg_task_time_s": m["avg_task_time_s"],
                    "min_task_time_s": min(wtt) if wtt else 0.0,
                    "max_task_time_s": max(wtt) if wtt else 0.0,
                    "throughput_trials_per_s": m["throughput"],
                    "worker_task_times_s": json.dumps(wtt),
                }
            )
    print(f"\n  Results exported to {filepath}")


async def run_benchmark_grid(foreman_host="localhost"):
    """
    Run the full benchmark grid:
      Trials  : 1M, 2M, 5M, 10M, 50M, 100M, 1B
      Workers : 6, 12, 18
    Collects job exec time, per-worker task times, averages, and throughput.
    """
    trial_counts = [
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        50_000_000, 
    ]
    worker_counts = [1]

    total_runs = len(trial_counts) * len(worker_counts)
    all_metrics = []

    print_separator()
    print(f"  AUTOMATED MONTE CARLO EULER BENCHMARK")
    print(f"  Trials grid : {', '.join(format_trials(t) for t in trial_counts)}")
    print(f"  Workers grid: {', '.join(str(w) for w in worker_counts)}")
    print(f"  Total runs  : {total_runs}")
    print(f"  Foreman     : {foreman_host}:9000")
    print(f"  Started at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    run_idx = 0
    for trials in trial_counts:
        for workers in worker_counts:
            run_idx += 1
            print(
                f"\n>>> Run {run_idx}/{total_runs}: "
                f"{format_trials(trials)} trials, {workers} workers"
            )

            try:
                metrics = await run_distributed_monte_carlo_euler(
                    total_trials=trials,
                    num_workers=workers,
                    foreman_host=foreman_host,
                )
                all_metrics.append(metrics)
                print_run_result(metrics)
            except Exception as e:
                print(f"  ERROR: {e}")
                all_metrics.append(
                    {
                        "total_trials": trials,
                        "num_workers": workers,
                        "job_exec_time_s": 0.0,
                        "worker_task_times_s": [],
                        "avg_task_time_s": 0.0,
                        "throughput": 0.0,
                        "aggregated": {},
                    }
                )

            # Small cooldown between runs to let resources settle
            if run_idx < total_runs:
                print("  Cooldown 3s...")
                await asyncio.sleep(3)

    # ---- Print summary tables ----
    successful = [m for m in all_metrics if m["job_exec_time_s"] > 0]
    if successful:
        print_summary_table(successful)
        print_aggregate_averages(successful)

        # Export CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            os.path.dirname(__file__), f"benchmark_results_{timestamp}.csv"
        )
        export_results_csv(successful, csv_path)
    else:
        print("\n  No successful runs to summarize.")

    print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    return all_metrics


# =========================================================
# 🏁 MAIN
# =========================================================
async def main():
    """
    Entry point: runs the full automated benchmark grid.
    Optional CLI args:
        python automated_monte_carlo_euler_client.py [foreman_host]
    """
    foreman_host = "localhost"
    if len(sys.argv) > 1:
        foreman_host = sys.argv[1]

    await run_benchmark_grid(foreman_host=foreman_host)


if __name__ == "__main__":
    asyncio.run(main())
