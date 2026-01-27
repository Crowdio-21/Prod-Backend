#!/usr/bin/env python3
"""
Monte Carlo Simulation Client for Estimating Euler's Number (e)

This script demonstrates:
1. Distributed Monte Carlo simulation using worker devices
2. Estimating Euler's number using random sampling
3. Task offloading to worker devices
4. Result aggregation and statistical analysis

Mathematical Background:
If we repeatedly generate random numbers uniformly between 0 and 1,
the expected number of random numbers needed until their sum exceeds 1
is Euler's number e ≈ 2.71828...
"""

import asyncio
import sys
import os
import json
import time

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from developer_sdk import connect, map as distributed_map, disconnect


def monte_carlo_euler_worker(num_trials):
    """
    Worker function to perform Monte Carlo trials for estimating e
    
    Args:
        num_trials: Number of simulation trials to run
    
    Returns:
        Dictionary containing trial results and statistics
    """
    import random
    import time
    
    start = time.time()
    
    try:
        random.seed()  # Ensure different seeds on different workers
        
        total_count = 0
        
        # Run Monte Carlo trials
        for _ in range(num_trials):
            random_sum = 0.0
            count = 0
            
            # Keep adding random numbers until sum exceeds 1
            while random_sum < 1.0:
                random_sum += random.random()
                count += 1
            
            total_count += count
        
        # Calculate average count (estimate of e)
        estimated_e = total_count / num_trials
        
        latency_ms = int((time.time() - start) * 1000)
        
        result = {
            "num_trials": num_trials,
            "total_count": total_count,
            "estimated_e": round(estimated_e, 6),
            "latency_ms": latency_ms,
            "status": "success"
        }
        
        print(f"[Worker] Completed {num_trials} trials | e ≈ {estimated_e:.6f}")
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
            "error": str(e)
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
            "worker_results": []
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
    std_dev = variance ** 0.5
    
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
        "worker_results": valid
    }


# =========================================================
# 🚀 DISTRIBUTED EXECUTION
# =========================================================
async def run_distributed_monte_carlo_euler(total_trials, num_workers=None, foreman_host="localhost"):
    """
    Run distributed Monte Carlo simulation to estimate Euler's number
    
    Args:
        total_trials: Total number of Monte Carlo trials
        num_workers: Number of workers (if None, determined by available workers)
        foreman_host: Hostname/IP of foreman server
    """
    print("\n" + "=" * 70)
    print("🎲 DISTRIBUTED MONTE CARLO ESTIMATION OF EULER'S NUMBER (e)")
    print("=" * 70)
    
    print(f"\n📡 Connecting to foreman at {foreman_host}:9000...")
    await connect(foreman_host, 9000)
    print("✅ Connected")
    
    print(f"\n🎯 Total Monte Carlo trials: {total_trials:,}")
    
    # Determine number of workers and trials per worker
    if num_workers is None:
        num_workers = 4  # Default number of task batches
    
    trials_per_worker = total_trials // num_workers
    task_inputs = [trials_per_worker] * num_workers
    
    # Handle remainder
    remainder = total_trials % num_workers
    if remainder > 0:
        task_inputs[-1] += remainder
    
    print(f"👥 Distributing across {num_workers} tasks")
    print(f"📊 Trials per task: {trials_per_worker:,}")
    
    start_time = time.time()
    
    print("\n⏳ Dispatching tasks to workers...")
    results = await distributed_map(monte_carlo_euler_worker, task_inputs)
    
    execution_time = time.time() - start_time
    
    aggregated = aggregate_monte_carlo_results(results)
    
    print("\n" + "-" * 70)
    print("📊 RESULTS")
    print("-" * 70)
    
    print(f"\n🎯 Final Estimate of e: {aggregated['final_estimate']}")
    print(f"📐 True value of e:     {aggregated['true_e']}")
    print(f"❌ Absolute Error:      {aggregated['absolute_error']}")
    print(f"📉 Error Percentage:    {aggregated['error_percentage']}%")
    
    print(f"\n📊 Statistical Summary:")
    print(f"   Workers completed:   {aggregated['worker_count']}")
    print(f"   Total trials:        {aggregated['total_trials']:,}")
    print(f"   Avg worker estimate: {aggregated['avg_worker_estimate']}")
    print(f"   Std deviation:       {aggregated['std_dev']}")
    print(f"   Min estimate:        {aggregated['min_estimate']}")
    print(f"   Max estimate:        {aggregated['max_estimate']}")
    
    print(f"\n⚡ Performance:")
    print(f"   Total execution time: {execution_time:.2f} seconds")
    print(f"   Avg worker latency:   {aggregated['avg_latency_ms']:.1f} ms")
    print(f"   Throughput:           {aggregated['total_trials'] / execution_time:,.0f} trials/sec")
    
    if aggregated.get("error_count", 0) > 0:
        print(f"\n⚠️  Failed workers: {aggregated['error_count']}")
    
    print("\n📋 Worker-level Results:")
    for i, r in enumerate(aggregated.get("worker_results", []), 1):
        print(f"{i}. e ≈ {r['estimated_e']:.6f} | {r['num_trials']:,} trials | {r['latency_ms']} ms")
    
    await disconnect()
    
    return aggregated


# =========================================================
# 🏁 MAIN
# =========================================================
async def main():
    """
    Main entry point for Monte Carlo Euler estimation
    """
    # Parse command line arguments
    total_trials = 1000000  # Default: 1 million trials
    num_workers = 8  # Default: 8 workers
    foreman_host = "localhost"
    
    if len(sys.argv) > 1:
        try:
            total_trials = int(sys.argv[1])
        except ValueError:
            foreman_host = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            num_workers = int(sys.argv[2])
        except ValueError:
            pass
    
    if len(sys.argv) > 3:
        foreman_host = sys.argv[3]
    
    print(f"\n📝 Configuration:")
    print(f"   Total trials: {total_trials:,}")
    print(f"   Workers: {num_workers}")
    print(f"   Foreman: {foreman_host}")
    
    await run_distributed_monte_carlo_euler(total_trials, num_workers, foreman_host)


if __name__ == "__main__":
    asyncio.run(main())
