"""
Real Evaluation Runner for CROWDio.

Runs actual jobs through the system and collects real metrics.
This requires the foreman and workers to be running.
"""

from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from developer_sdk.client import CrowdComputeClient
from .real_metrics_collector import RealMetricsCollector, RealSystemMetrics, collect_real_evaluation
from .visualization import EvaluationVisualizer


@dataclass
class WorkloadConfig:
    """Configuration for a test workload."""
    name: str
    num_tasks: int
    task_function: Callable
    task_args_generator: Callable[[int], List[Any]]
    description: str = ""


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""
    name: str
    foreman_host: str = "localhost"
    foreman_port: int = 9000
    output_dir: str = "evaluation_results"
    workloads: List[WorkloadConfig] = field(default_factory=list)
    iterations: int = 1
    warmup_seconds: float = 5.0
    cooldown_seconds: float = 5.0


@dataclass
class WorkloadResult:
    """Result from running a single workload."""
    workload_name: str
    iteration: int
    job_id: str
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float
    num_tasks: int
    num_completed: int
    throughput_tasks_per_sec: float
    results: List[Any] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    config_name: str
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    workload_results: List[WorkloadResult] = field(default_factory=list)
    system_metrics: Optional[RealSystemMetrics] = None
    summary: Dict[str, Any] = field(default_factory=dict)


class RealEvaluationRunner:
    """
    Runs real evaluation by submitting jobs to CROWDio and collecting metrics.
    
    This requires:
    1. Foreman to be running (python -m foreman.main)
    2. Workers to be connected (python -m pc_worker.main)
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = CrowdComputeClient()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_collector = RealMetricsCollector(config.output_dir)
        self.visualizer = EvaluationVisualizer(self.output_dir / "charts")
        self.job_ids: List[str] = []
    
    async def run(self) -> EvaluationResult:
        """Run the complete evaluation."""
        print(f"\n{'=' * 60}")
        print(f"STARTING EVALUATION: {self.config.name}")
        print(f"{'=' * 60}")
        
        result = EvaluationResult(
            config_name=self.config.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration_seconds=0,
        )
        
        try:
            # Connect to foreman
            print(f"\nConnecting to foreman at {self.config.foreman_host}:{self.config.foreman_port}...")
            await self.client.connect(self.config.foreman_host, self.config.foreman_port)
            print("Connected successfully!")
            
            # Warmup
            if self.config.warmup_seconds > 0:
                print(f"\nWarmup period: {self.config.warmup_seconds}s...")
                await asyncio.sleep(self.config.warmup_seconds)
            
            # Run workloads
            for workload in self.config.workloads:
                for iteration in range(1, self.config.iterations + 1):
                    print(f"\n{'=' * 40}")
                    print(f"Workload: {workload.name} (iteration {iteration}/{self.config.iterations})")
                    print(f"Tasks: {workload.num_tasks}")
                    print(f"{'=' * 40}")
                    
                    workload_result = await self._run_workload(workload, iteration)
                    result.workload_results.append(workload_result)
                    
                    # Cooldown between iterations
                    if iteration < self.config.iterations:
                        print(f"Cooldown: {self.config.cooldown_seconds}s...")
                        await asyncio.sleep(self.config.cooldown_seconds)
            
            # Cooldown
            if self.config.cooldown_seconds > 0:
                print(f"\nFinal cooldown: {self.config.cooldown_seconds}s...")
                await asyncio.sleep(self.config.cooldown_seconds)
            
        finally:
            await self.client.disconnect()
        
        result.end_time = datetime.now()
        result.total_duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        # Collect real metrics from database
        print("\nCollecting real metrics from database...")
        result.system_metrics = await self.metrics_collector.collect_all_metrics(
            job_ids=self.job_ids
        )
        
        # Generate summary
        result.summary = self._generate_summary(result)
        
        # Save results
        self._save_results(result)
        
        # Generate visualizations
        self._generate_visualizations(result)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    async def _run_workload(self, workload: WorkloadConfig, iteration: int) -> WorkloadResult:
        """Run a single workload and return metrics."""
        start_time = datetime.now()
        
        # Generate task arguments
        task_args = workload.task_args_generator(workload.num_tasks)
        
        try:
            # Submit job via client.map()
            print(f"Submitting {workload.num_tasks} tasks...")
            submit_time = time.time()
            
            results = await self.client.map(workload.task_function, task_args)
            
            end_time_epoch = time.time()
            execution_time = end_time_epoch - submit_time
            
            # Get job ID from client (it's in pending_jobs or we can track it)
            # For now, we'll track completed jobs
            job_id = f"workload_{workload.name}_{iteration}_{int(submit_time)}"
            self.job_ids.append(job_id)
            
            end_time = datetime.now()
            
            return WorkloadResult(
                workload_name=workload.name,
                iteration=iteration,
                job_id=job_id,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                num_tasks=workload.num_tasks,
                num_completed=len(results) if results else 0,
                throughput_tasks_per_sec=workload.num_tasks / execution_time if execution_time > 0 else 0,
                results=results if results else [],
                success=True,
            )
            
        except Exception as e:
            end_time = datetime.now()
            return WorkloadResult(
                workload_name=workload.name,
                iteration=iteration,
                job_id="",
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=(end_time - start_time).total_seconds(),
                num_tasks=workload.num_tasks,
                num_completed=0,
                throughput_tasks_per_sec=0,
                success=False,
                error=str(e),
            )
    
    def _generate_summary(self, result: EvaluationResult) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful_runs = [r for r in result.workload_results if r.success]
        
        if not successful_runs:
            return {"error": "No successful workload runs"}
        
        total_tasks = sum(r.num_tasks for r in successful_runs)
        total_completed = sum(r.num_completed for r in successful_runs)
        total_time = sum(r.execution_time_seconds for r in successful_runs)
        
        summary = {
            "total_workloads_run": len(result.workload_results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(result.workload_results) - len(successful_runs),
            "total_tasks_submitted": total_tasks,
            "total_tasks_completed": total_completed,
            "completion_rate": total_completed / total_tasks if total_tasks > 0 else 0,
            "total_execution_time_sec": total_time,
            "avg_throughput_tasks_per_sec": total_tasks / total_time if total_time > 0 else 0,
            "workload_summaries": {},
        }
        
        # Per-workload summaries
        for workload in self.config.workloads:
            workload_runs = [r for r in successful_runs if r.workload_name == workload.name]
            if workload_runs:
                summary["workload_summaries"][workload.name] = {
                    "iterations": len(workload_runs),
                    "avg_execution_time_sec": sum(r.execution_time_seconds for r in workload_runs) / len(workload_runs),
                    "avg_throughput": sum(r.throughput_tasks_per_sec for r in workload_runs) / len(workload_runs),
                    "total_tasks": sum(r.num_tasks for r in workload_runs),
                    "total_completed": sum(r.num_completed for r in workload_runs),
                }
        
        # Add system metrics summary if available
        if result.system_metrics:
            metrics = result.system_metrics
            summary["system_metrics"] = {
                "workers_used": metrics.total_workers,
                "online_workers": metrics.online_workers,
                "jains_fairness_index": metrics.jains_fairness_index,
                "coefficient_of_variation": metrics.coefficient_of_variation,
                "total_worker_failures": metrics.total_worker_failures,
                "checkpoint_recovery_rate": metrics.checkpoint_recovery_rate,
                "load_distribution": metrics.load_distribution,
            }
        
        return summary
    
    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to files."""
        # Save main results
        results_file = self.output_dir / f"evaluation_{self.config.name}_{int(time.time())}.json"
        
        results_dict = {
            "config_name": result.config_name,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "total_duration_seconds": result.total_duration_seconds,
            "summary": result.summary,
            "workload_results": [
                {
                    "workload_name": wr.workload_name,
                    "iteration": wr.iteration,
                    "job_id": wr.job_id,
                    "start_time": wr.start_time.isoformat(),
                    "end_time": wr.end_time.isoformat(),
                    "execution_time_seconds": wr.execution_time_seconds,
                    "num_tasks": wr.num_tasks,
                    "num_completed": wr.num_completed,
                    "throughput_tasks_per_sec": wr.throughput_tasks_per_sec,
                    "success": wr.success,
                    "error": wr.error,
                }
                for wr in result.workload_results
            ],
        }
        
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save system metrics separately
        if result.system_metrics:
            self.metrics_collector.export_to_json(
                result.system_metrics,
                f"system_metrics_{self.config.name}_{int(time.time())}.json"
            )
    
    def _generate_visualizations(self, result: EvaluationResult) -> None:
        """Generate visualization charts from results."""
        if not result.workload_results or not result.system_metrics:
            return
        
        try:
            # Throughput over iterations
            for workload in self.config.workloads:
                workload_runs = [r for r in result.workload_results 
                                if r.workload_name == workload.name and r.success]
                if workload_runs:
                    iterations = list(range(1, len(workload_runs) + 1))
                    throughputs = [r.throughput_tasks_per_sec for r in workload_runs]
                    self.visualizer.plot_throughput_scaling(
                        iterations, throughputs,
                        title=f"Throughput: {workload.name}",
                        save_name=f"throughput_{workload.name}"
                    )
            
            # Load distribution
            if result.system_metrics.load_distribution:
                worker_loads = {
                    w: [result.system_metrics.load_distribution.get(w, 0)]
                    for w in result.system_metrics.load_distribution.keys()
                }
                self.visualizer.plot_load_distribution(
                    worker_loads,
                    title=f"Load Distribution: {self.config.name}",
                    save_name=f"load_distribution_{self.config.name}"
                )
            
            print(f"Charts saved to: {self.output_dir / 'charts'}")
            
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    def _print_summary(self, result: EvaluationResult) -> None:
        """Print evaluation summary to console."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATION COMPLETE: {self.config.name}")
        print(f"{'=' * 60}")
        print(f"\nDuration: {result.total_duration_seconds:.2f} seconds")
        
        summary = result.summary
        print(f"\nWorkloads: {summary.get('successful_runs', 0)} successful, {summary.get('failed_runs', 0)} failed")
        print(f"Tasks: {summary.get('total_tasks_completed', 0)}/{summary.get('total_tasks_submitted', 0)} completed ({summary.get('completion_rate', 0):.1%})")
        print(f"Throughput: {summary.get('avg_throughput_tasks_per_sec', 0):.2f} tasks/sec")
        
        if "system_metrics" in summary:
            sm = summary["system_metrics"]
            print(f"\nWorkers: {sm.get('online_workers', 0)}/{sm.get('workers_used', 0)} online")
            print(f"Load Fairness (Jain's): {sm.get('jains_fairness_index', 0):.3f}")
            print(f"Failures: {sm.get('total_worker_failures', 0)}")
        
        if "workload_summaries" in summary:
            print("\nPer-Workload Results:")
            for name, ws in summary["workload_summaries"].items():
                print(f"  {name}:")
                print(f"    - Avg time: {ws.get('avg_execution_time_sec', 0):.2f}s")
                print(f"    - Avg throughput: {ws.get('avg_throughput', 0):.2f} tasks/sec")


# ==================== Pre-defined Workloads ====================

def cpu_intensive_task(x):
    """CPU-intensive task for testing."""
    result = 0
    for i in range(100000):
        result += i * x
    return result


def io_simulation_task(x):
    """Simulated I/O task."""
    import time
    time.sleep(0.1)  # Simulate I/O wait
    return x * 2


def mixed_workload_task(x):
    """Mixed CPU and I/O task."""
    import time
    # CPU work
    result = sum(i * x for i in range(10000))
    # I/O simulation
    time.sleep(0.05)
    return result


def generate_simple_args(num_tasks: int) -> List[int]:
    """Generate simple integer arguments."""
    return list(range(1, num_tasks + 1))


# Pre-defined workload configs
CPU_WORKLOAD = WorkloadConfig(
    name="cpu_intensive",
    num_tasks=50,
    task_function=cpu_intensive_task,
    task_args_generator=generate_simple_args,
    description="CPU-intensive computation workload"
)

IO_WORKLOAD = WorkloadConfig(
    name="io_simulation",
    num_tasks=50,
    task_function=io_simulation_task,
    task_args_generator=generate_simple_args,
    description="I/O simulation workload"
)

MIXED_WORKLOAD = WorkloadConfig(
    name="mixed",
    num_tasks=50,
    task_function=mixed_workload_task,
    task_args_generator=generate_simple_args,
    description="Mixed CPU and I/O workload"
)


async def run_quick_real_evaluation(
    host: str = "localhost",
    port: int = 9000,
    output_dir: str = "evaluation_results"
) -> EvaluationResult:
    """Run a quick real evaluation with default workloads."""
    config = EvaluationConfig(
        name="quick_real_eval",
        foreman_host=host,
        foreman_port=port,
        output_dir=output_dir,
        workloads=[CPU_WORKLOAD, IO_WORKLOAD],
        iterations=1,
        warmup_seconds=2.0,
        cooldown_seconds=2.0,
    )
    
    runner = RealEvaluationRunner(config)
    return await runner.run()


async def run_full_real_evaluation(
    host: str = "localhost",
    port: int = 9000,
    output_dir: str = "evaluation_results"
) -> EvaluationResult:
    """Run a comprehensive real evaluation."""
    config = EvaluationConfig(
        name="full_real_eval",
        foreman_host=host,
        foreman_port=port,
        output_dir=output_dir,
        workloads=[CPU_WORKLOAD, IO_WORKLOAD, MIXED_WORKLOAD],
        iterations=3,
        warmup_seconds=5.0,
        cooldown_seconds=5.0,
    )
    
    runner = RealEvaluationRunner(config)
    return await runner.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run real CROWDio evaluation")
    parser.add_argument("--host", default="localhost", help="Foreman host")
    parser.add_argument("--port", type=int, default=9000, help="Foreman port")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation")
    args = parser.parse_args()
    
    if args.quick:
        result = asyncio.run(run_quick_real_evaluation(args.host, args.port, args.output))
    else:
        result = asyncio.run(run_full_real_evaluation(args.host, args.port, args.output))
