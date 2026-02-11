"""
Scalability Experiment

Tests system performance as the number of devices/workers increases.

Scenarios:
- Linear scaling: 1, 2, 4, 8, 16, 32 workers
- Workload scaling: Fixed workers, increasing tasks
- Combined scaling: Both workers and tasks increase
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    WorkloadConfig,
    WorkerSimulationConfig,
    DeviceCategory,
    WorkloadType,
)


@dataclass
class ScalabilityConfig(ExperimentConfig):
    """Configuration for scalability experiments"""
    
    # Worker scaling
    worker_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    # Task scaling
    task_counts: List[int] = field(default_factory=lambda: [100])
    
    # Scaling mode
    scale_workers: bool = True
    scale_tasks: bool = False
    
    # Worker distribution (if heterogeneous)
    device_distribution: Dict[str, float] = field(default_factory=lambda: {
        "high_end_pc": 1.0  # 100% high-end PCs by default
    })


class ScalabilityExperiment(Experiment):
    """
    Experiment to test system scalability.
    
    Measures:
    - Throughput vs number of workers
    - Task completion time vs worker count
    - Speedup and efficiency metrics
    - Load balancing at different scales
    """
    
    def __init__(self, config: ScalabilityConfig):
        super().__init__(config)
        self.scalability_config = config
        
        # Results per scale point
        self.scale_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Simulated workers
        self.simulated_workers: List[Dict[str, Any]] = []
    
    async def setup(self) -> None:
        """Initialize experiment"""
        print(f"ScalabilityExperiment: Setting up with worker counts {self.scalability_config.worker_counts}")
        
        # Initialize tracking (in production, these would be real tracker instances)
        self.scale_results = {}
    
    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run scaling tests"""
        all_results = []
        
        # Determine what to scale
        if self.scalability_config.scale_workers:
            worker_counts = self.scalability_config.worker_counts
        else:
            worker_counts = [self.config.num_workers]
        
        if self.scalability_config.scale_tasks:
            task_counts = self.scalability_config.task_counts
        else:
            task_counts = [self.config.workload.num_tasks]
        
        # Run tests for each scale point
        for num_workers in worker_counts:
            for num_tasks in task_counts:
                scale_key = f"w{num_workers}_t{num_tasks}"
                print(f"  Testing: {num_workers} workers, {num_tasks} tasks")
                
                result = await self._run_scale_point(
                    num_workers=num_workers,
                    num_tasks=num_tasks,
                    iteration=iteration
                )
                
                if scale_key not in self.scale_results:
                    self.scale_results[scale_key] = []
                self.scale_results[scale_key].append(result)
                
                all_results.append(result)
                
                # Brief pause between scale points
                await asyncio.sleep(0.5)
        
        # Aggregate iteration results
        return self._aggregate_iteration(all_results)
    
    async def _run_scale_point(
        self,
        num_workers: int,
        num_tasks: int,
        iteration: int
    ) -> Dict[str, Any]:
        """Run experiment at a specific scale point"""
        
        # Create simulated workers based on distribution
        workers = self._create_simulated_workers(num_workers)
        
        # Generate tasks
        original_num_tasks = self.config.workload.num_tasks
        self.config.workload.num_tasks = num_tasks
        tasks = self.generate_workload()
        self.config.workload.num_tasks = original_num_tasks
        
        # Simulate task execution
        start_time = time.time()
        results = await self._simulate_execution(tasks, workers)
        end_time = time.time()
        
        # Calculate metrics
        total_duration = end_time - start_time
        completed_tasks = sum(1 for r in results if r["success"])
        failed_tasks = sum(1 for r in results if not r["success"])
        
        task_times = [r["duration"] for r in results if r["success"]]
        
        # Calculate speedup (compared to single worker baseline)
        single_worker_time = sum(t["expected_duration"] for t in tasks)
        actual_time = total_duration
        speedup = single_worker_time / actual_time if actual_time > 0 else 0
        efficiency = speedup / num_workers if num_workers > 0 else 0
        
        # Calculate load balance
        tasks_per_worker = {}
        for r in results:
            worker_id = r["worker_id"]
            tasks_per_worker[worker_id] = tasks_per_worker.get(worker_id, 0) + 1
        
        task_counts_list = list(tasks_per_worker.values())
        if task_counts_list:
            avg_tasks = statistics.mean(task_counts_list)
            if avg_tasks > 0:
                cv = statistics.stdev(task_counts_list) / avg_tasks if len(task_counts_list) > 1 else 0
            else:
                cv = 0
            
            # Jain's fairness index
            sum_x = sum(task_counts_list)
            sum_x_sq = sum(x**2 for x in task_counts_list)
            n = len(task_counts_list)
            jfi = (sum_x ** 2) / (n * sum_x_sq) if sum_x_sq > 0 else 1.0
        else:
            cv = 0
            jfi = 1.0
        
        return {
            "scale_point": f"w{num_workers}_t{num_tasks}",
            "num_workers": num_workers,
            "num_tasks": num_tasks,
            "iteration": iteration,
            "total_duration": total_duration,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "throughput": completed_tasks / total_duration if total_duration > 0 else 0,
            "avg_task_time": statistics.mean(task_times) if task_times else None,
            "p95_task_time": sorted(task_times)[int(len(task_times) * 0.95)] if task_times else None,
            "speedup": speedup,
            "efficiency": efficiency,
            "jains_fairness_index": jfi,
            "coefficient_of_variation": cv,
            "tasks_per_worker": tasks_per_worker,
        }
    
    def _create_simulated_workers(self, num_workers: int) -> List[Dict[str, Any]]:
        """Create simulated workers based on device distribution"""
        workers = []
        distribution = self.scalability_config.device_distribution
        
        # Normalize distribution
        total = sum(distribution.values())
        normalized = {k: v/total for k, v in distribution.items()}
        
        # Create workers
        for i in range(num_workers):
            # Pick device category based on distribution
            r = random.random()
            cumulative = 0
            selected_category = DeviceCategory.MID_RANGE_PC
            
            for cat_name, prob in normalized.items():
                cumulative += prob
                if r <= cumulative:
                    selected_category = DeviceCategory(cat_name)
                    break
            
            config = WorkerSimulationConfig.from_category(selected_category)
            
            workers.append({
                "worker_id": f"worker_{i}",
                "config": config,
                "current_tasks": 0,
                "total_completed": 0,
            })
        
        return workers
    
    async def _simulate_execution(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate distributed task execution"""
        results = []
        task_queue = list(tasks)
        
        # Simple round-robin assignment simulation
        worker_idx = 0
        
        while task_queue:
            task = task_queue.pop(0)
            worker = workers[worker_idx]
            
            # Simulate execution
            duration, success, error = self.simulate_task_execution(
                task, worker["config"]
            )
            
            results.append({
                "task_id": task["task_id"],
                "worker_id": worker["worker_id"],
                "duration": duration,
                "success": success,
                "error": error,
            })
            
            worker["total_completed"] += 1
            
            # Move to next worker
            worker_idx = (worker_idx + 1) % len(workers)
            
            # Small delay to simulate actual execution
            await asyncio.sleep(0.001)
        
        return results
    
    def _aggregate_iteration(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all scale points in an iteration"""
        if not results:
            return {}
        
        return {
            "scale_points": len(results),
            "total_tasks": sum(r["completed_tasks"] + r["failed_tasks"] for r in results),
            "completed_tasks": sum(r["completed_tasks"] for r in results),
            "failed_tasks": sum(r["failed_tasks"] for r in results),
            "avg_throughput": statistics.mean(r["throughput"] for r in results),
            "max_throughput": max(r["throughput"] for r in results),
            "avg_speedup": statistics.mean(r["speedup"] for r in results),
            "avg_efficiency": statistics.mean(r["efficiency"] for r in results),
            "avg_jains_fairness_index": statistics.mean(r["jains_fairness_index"] for r in results),
            "scale_results": results,
        }
    
    async def teardown(self) -> None:
        """Clean up"""
        print("ScalabilityExperiment: Teardown complete")
    
    def get_scalability_analysis(self) -> Dict[str, Any]:
        """
        Analyze scalability characteristics.
        
        Returns:
            Analysis including speedup curves, efficiency, and bottlenecks
        """
        analysis = {
            "worker_scaling": [],
            "ideal_speedup": [],
            "actual_speedup": [],
            "efficiency": [],
            "throughput_curve": [],
        }
        
        # Collect data for each worker count
        for scale_key, results in sorted(self.scale_results.items()):
            if not results:
                continue
            
            # Average across iterations
            num_workers = results[0]["num_workers"]
            avg_speedup = statistics.mean(r["speedup"] for r in results)
            avg_efficiency = statistics.mean(r["efficiency"] for r in results)
            avg_throughput = statistics.mean(r["throughput"] for r in results)
            
            analysis["worker_scaling"].append(num_workers)
            analysis["ideal_speedup"].append(num_workers)
            analysis["actual_speedup"].append(avg_speedup)
            analysis["efficiency"].append(avg_efficiency)
            analysis["throughput_curve"].append(avg_throughput)
        
        # Calculate scaling efficiency
        if len(analysis["worker_scaling"]) >= 2:
            # Check if sub-linear scaling
            speedups = analysis["actual_speedup"]
            workers = analysis["worker_scaling"]
            
            if workers[-1] > workers[0]:
                expected_ratio = workers[-1] / workers[0]
                actual_ratio = speedups[-1] / speedups[0] if speedups[0] > 0 else 0
                
                analysis["scaling_factor"] = actual_ratio / expected_ratio if expected_ratio > 0 else 0
                
                if analysis["scaling_factor"] >= 0.9:
                    analysis["scaling_quality"] = "Near-linear (excellent)"
                elif analysis["scaling_factor"] >= 0.7:
                    analysis["scaling_quality"] = "Good scaling"
                elif analysis["scaling_factor"] >= 0.5:
                    analysis["scaling_quality"] = "Moderate scaling"
                else:
                    analysis["scaling_quality"] = "Poor scaling - bottleneck detected"
        
        return analysis
