"""
Generate visualizations from real CROWDio metrics.
Run this after collecting real metrics with real_metrics_collector.py
"""

import json
from pathlib import Path
from evaluation.visualization import EvaluationVisualizer


def generate_real_charts(metrics_file: str = "evaluation_results/real_metrics.json"):
    """Generate charts from real metrics JSON file."""
    
    # Load metrics
    with open(metrics_file) as f:
        data = json.load(f)
    
    output_dir = Path(metrics_file).parent / "charts"
    viz = EvaluationVisualizer(output_dir)
    
    print("Generating charts from real metrics...")
    print(f"Output directory: {output_dir}")
    
    charts_created = []
    
    # 1. Load Distribution Chart
    load_dist = data.get("load_balancing", {}).get("distribution", {})
    if load_dist:
        worker_loads = {w: [c] for w, c in load_dist.items()}
        path = viz.plot_load_distribution(
            worker_loads, 
            title="Real Load Distribution Across Workers",
            save_name="real_load_distribution"
        )
        if path:
            charts_created.append(path)
            print(f"  ✓ Load distribution: {path}")
    
    # 2. Job Performance Comparison
    jobs = data.get("jobs", [])
    if jobs:
        # Job throughput comparison
        job_metrics = {}
        for j in jobs:
            name = j["job_id"][:8]
            job_metrics[name] = {
                "throughput": j.get("throughput_tasks_per_sec", 0),
                "completion_rate": j.get("completion_rate", 0) * 100,
            }
        
        path = viz.plot_experiment_comparison(
            job_metrics,
            "throughput",
            title="Job Throughput Comparison (tasks/sec)",
            save_name="real_job_throughput"
        )
        if path:
            charts_created.append(path)
            print(f"  ✓ Job throughput: {path}")
        
        # Execution time comparison
        completed_jobs = [j for j in jobs if j.get("total_execution_time_sec")]
        if len(completed_jobs) >= 2:
            exec_metrics = {
                j["job_id"][:8]: {"execution_time": j["total_execution_time_sec"]}
                for j in completed_jobs
            }
            path = viz.plot_experiment_comparison(
                exec_metrics,
                "execution_time",
                title="Job Execution Times (seconds)",
                save_name="real_job_execution_time"
            )
            if path:
                charts_created.append(path)
                print(f"  ✓ Job execution times: {path}")
    
    # 3. Worker Performance
    workers = data.get("workers", [])
    if workers:
        device_metrics = {}
        for w in workers:
            name = w["worker_id"][:8]
            device_metrics[name] = {
                "tasks_completed": w.get("total_tasks_completed", 0),
                "failure_rate_pct": w.get("failure_rate", 0) * 100,
            }
        
        path = viz.plot_device_performance(
            device_metrics,
            title="Worker Performance",
            save_name="real_worker_performance"
        )
        if path:
            charts_created.append(path)
            print(f"  ✓ Worker performance: {path}")
    
    # Print summary
    print(f"\n{'=' * 50}")
    print("REAL METRICS SUMMARY")
    print(f"{'=' * 50}")
    
    summary = data.get("summary", {})
    perf = data.get("performance", {})
    lb = data.get("load_balancing", {})
    
    print(f"\nJobs:")
    print(f"  Total: {summary.get('total_jobs', 0)}")
    print(f"  Completed: {summary.get('completed_jobs', 0)}")
    print(f"  Failed: {summary.get('failed_jobs', 0)}")
    
    print(f"\nTasks:")
    print(f"  Total: {summary.get('total_tasks', 0)}")
    print(f"  Completed: {summary.get('completed_tasks', 0)}")
    print(f"  Pending: {summary.get('pending_tasks', 0)}")
    
    print(f"\nWorkers:")
    print(f"  Total: {summary.get('total_workers', 0)}")
    print(f"  Online: {summary.get('online_workers', 0)}")
    
    print(f"\nPerformance:")
    print(f"  Throughput: {perf.get('overall_throughput_tasks_per_sec', 0):.2f} tasks/sec")
    print(f"  Avg job time: {perf.get('avg_job_completion_time_sec', 0):.2f}s")
    print(f"  Avg task time: {perf.get('avg_task_execution_time_sec', 0):.2f}s")
    
    print(f"\nLoad Balancing:")
    print(f"  Jain's Fairness Index: {lb.get('jains_fairness_index', 0):.3f}")
    print(f"  Coefficient of Variation: {lb.get('coefficient_of_variation', 0):.3f}")
    
    print(f"\n{len(charts_created)} charts created in {output_dir}")
    
    return charts_created


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate charts from real metrics")
    parser.add_argument("--input", default="evaluation_results/real_metrics.json",
                       help="Path to real_metrics.json file")
    args = parser.parse_args()
    
    generate_real_charts(args.input)
