"""
Heterogeneity Experiment

Tests system performance with diverse device capabilities and configurations.

Scenarios:
- Mixed PC and mobile devices
- Varying processing speeds
- Different network conditions
- Mixed battery/plugged-in devices
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
class HeterogeneityConfig(ExperimentConfig):
    """Configuration for heterogeneity experiments"""
    
    # Device mix scenarios
    scenarios: List[Dict[str, float]] = field(default_factory=lambda: [
        # Scenario 1: Homogeneous (all mid-range PCs)
        {"mid_range_pc": 1.0},
        
        # Scenario 2: Mixed PCs
        {"high_end_pc": 0.2, "mid_range_pc": 0.5, "low_end_pc": 0.3},
        
        # Scenario 3: PC + Mobile mix
        {"mid_range_pc": 0.5, "high_end_mobile": 0.3, "mid_range_mobile": 0.2},
        
        # Scenario 4: Highly heterogeneous
        {"high_end_pc": 0.1, "mid_range_pc": 0.2, "low_end_pc": 0.1,
         "high_end_mobile": 0.2, "mid_range_mobile": 0.2, "low_end_mobile": 0.1,
         "edge_device": 0.1},
    ])
    
    # Workload matching strategies
    test_workload_matching: bool = True  # Test performance-based task assignment
    
    # Network heterogeneity
    vary_network_conditions: bool = True
    
    # Compare scheduling strategies
    compare_schedulers: bool = True


class HeterogeneityExperiment(Experiment):
    """
    Experiment to test system with heterogeneous devices.
    
    Measures:
    - Performance impact of device diversity
    - Effectiveness of load balancing with varied capabilities
    - Task allocation efficiency
    - Impact on energy consumption
    """
    
    def __init__(self, config: HeterogeneityConfig):
        super().__init__(config)
        self.heterogeneity_config = config
        
        # Results per scenario
        self.scenario_results: Dict[str, List[Dict[str, Any]]] = {}
    
    async def setup(self) -> None:
        """Initialize experiment"""
        print(f"HeterogeneityExperiment: Setting up with {len(self.heterogeneity_config.scenarios)} scenarios")
        self.scenario_results = {}
    
    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run heterogeneity tests"""
        all_results = []
        
        for idx, scenario in enumerate(self.heterogeneity_config.scenarios):
            scenario_name = self._get_scenario_name(scenario)
            print(f"  Testing scenario {idx + 1}: {scenario_name}")
            
            # Run with round-robin scheduling
            result_rr = await self._run_scenario(
                scenario=scenario,
                scenario_name=f"{scenario_name}_rr",
                scheduler="round_robin",
                iteration=iteration
            )
            
            if scenario_name not in self.scenario_results:
                self.scenario_results[scenario_name] = []
            self.scenario_results[scenario_name].append(result_rr)
            all_results.append(result_rr)
            
            # Also test with performance-based scheduling if configured
            if self.heterogeneity_config.test_workload_matching:
                result_perf = await self._run_scenario(
                    scenario=scenario,
                    scenario_name=f"{scenario_name}_perf",
                    scheduler="performance_based",
                    iteration=iteration
                )
                all_results.append(result_perf)
            
            await asyncio.sleep(0.5)
        
        return self._aggregate_iteration(all_results)
    
    def _get_scenario_name(self, scenario: Dict[str, float]) -> str:
        """Generate a name for a scenario"""
        if len(scenario) == 1:
            return f"homogeneous_{list(scenario.keys())[0]}"
        elif all("pc" in k for k in scenario.keys()):
            return "mixed_pcs"
        elif any("mobile" in k for k in scenario.keys()):
            if any("pc" in k for k in scenario.keys()):
                return "pc_mobile_mix"
            return "mobile_only"
        return "highly_heterogeneous"
    
    async def _run_scenario(
        self,
        scenario: Dict[str, float],
        scenario_name: str,
        scheduler: str,
        iteration: int
    ) -> Dict[str, Any]:
        """Run a single heterogeneity scenario"""
        
        # Create workers with the specified distribution
        workers = self._create_heterogeneous_workers(
            scenario=scenario,
            num_workers=self.config.num_workers
        )
        
        # Generate tasks
        tasks = self.generate_workload()
        
        # Run simulation
        start_time = time.time()
        
        if scheduler == "performance_based":
            results = await self._simulate_performance_based(tasks, workers)
        else:
            results = await self._simulate_round_robin(tasks, workers)
        
        end_time = time.time()
        
        # Analyze results
        return self._analyze_scenario_results(
            scenario_name=scenario_name,
            scenario=scenario,
            scheduler=scheduler,
            workers=workers,
            results=results,
            duration=end_time - start_time,
            iteration=iteration
        )
    
    def _create_heterogeneous_workers(
        self,
        scenario: Dict[str, float],
        num_workers: int
    ) -> List[Dict[str, Any]]:
        """Create workers with heterogeneous capabilities"""
        workers = []
        
        # Normalize distribution
        total = sum(scenario.values())
        normalized = {k: v/total for k, v in scenario.items()}
        
        # Calculate how many of each type
        type_counts = {}
        remaining = num_workers
        
        for device_type, fraction in normalized.items():
            count = int(num_workers * fraction)
            type_counts[device_type] = count
            remaining -= count
        
        # Distribute remaining workers
        for device_type in normalized.keys():
            if remaining <= 0:
                break
            type_counts[device_type] = type_counts.get(device_type, 0) + 1
            remaining -= 1
        
        # Create workers
        worker_idx = 0
        for device_type, count in type_counts.items():
            category = DeviceCategory(device_type)
            
            for i in range(count):
                config = WorkerSimulationConfig.from_category(category)
                
                # Add some variation within category
                config.processing_speed_factor *= random.uniform(0.9, 1.1)
                
                if self.heterogeneity_config.vary_network_conditions:
                    config.latency_mean_ms *= random.uniform(0.8, 1.5)
                    config.bandwidth_mbps *= random.uniform(0.7, 1.3)
                
                workers.append({
                    "worker_id": f"worker_{worker_idx}",
                    "device_type": device_type,
                    "config": config,
                    "current_tasks": 0,
                    "total_completed": 0,
                    "total_execution_time": 0.0,
                })
                worker_idx += 1
        
        return workers
    
    async def _simulate_round_robin(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate with round-robin scheduling"""
        results = []
        worker_idx = 0
        
        for task in tasks:
            worker = workers[worker_idx]
            duration, success, error = self.simulate_task_execution(task, worker["config"])
            
            results.append({
                "task_id": task["task_id"],
                "worker_id": worker["worker_id"],
                "device_type": worker["device_type"],
                "duration": duration,
                "success": success,
                "error": error,
            })
            
            worker["total_completed"] += 1
            worker["total_execution_time"] += duration
            
            worker_idx = (worker_idx + 1) % len(workers)
            await asyncio.sleep(0.001)
        
        return results
    
    async def _simulate_performance_based(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate with performance-based scheduling"""
        results = []
        
        # Sort workers by processing speed (fastest first)
        sorted_workers = sorted(
            workers,
            key=lambda w: w["config"].processing_speed_factor,
            reverse=True
        )
        
        # Assign more tasks to faster workers
        task_idx = 0
        while task_idx < len(tasks):
            for worker in sorted_workers:
                if task_idx >= len(tasks):
                    break
                
                task = tasks[task_idx]
                duration, success, error = self.simulate_task_execution(task, worker["config"])
                
                results.append({
                    "task_id": task["task_id"],
                    "worker_id": worker["worker_id"],
                    "device_type": worker["device_type"],
                    "duration": duration,
                    "success": success,
                    "error": error,
                })
                
                worker["total_completed"] += 1
                worker["total_execution_time"] += duration
                task_idx += 1
                
                await asyncio.sleep(0.001)
        
        return results
    
    def _analyze_scenario_results(
        self,
        scenario_name: str,
        scenario: Dict[str, float],
        scheduler: str,
        workers: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        duration: float,
        iteration: int
    ) -> Dict[str, Any]:
        """Analyze results from a scenario run"""
        
        completed_tasks = sum(1 for r in results if r["success"])
        failed_tasks = sum(1 for r in results if not r["success"])
        
        task_times = [r["duration"] for r in results if r["success"]]
        
        # Per-device-type analysis
        by_device_type: Dict[str, Dict[str, Any]] = {}
        for worker in workers:
            dtype = worker["device_type"]
            if dtype not in by_device_type:
                by_device_type[dtype] = {
                    "worker_count": 0,
                    "tasks_completed": 0,
                    "total_time": 0.0,
                }
            by_device_type[dtype]["worker_count"] += 1
            by_device_type[dtype]["tasks_completed"] += worker["total_completed"]
            by_device_type[dtype]["total_time"] += worker["total_execution_time"]
        
        # Calculate throughput contribution per device type
        for dtype, stats in by_device_type.items():
            if stats["total_time"] > 0:
                stats["throughput"] = stats["tasks_completed"] / duration
                stats["avg_task_time"] = stats["total_time"] / stats["tasks_completed"] if stats["tasks_completed"] > 0 else 0
        
        # Load balance analysis
        tasks_per_worker = [w["total_completed"] for w in workers]
        if tasks_per_worker:
            avg_tasks = statistics.mean(tasks_per_worker)
            cv = statistics.stdev(tasks_per_worker) / avg_tasks if avg_tasks > 0 and len(tasks_per_worker) > 1 else 0
            
            sum_x = sum(tasks_per_worker)
            sum_x_sq = sum(x**2 for x in tasks_per_worker)
            n = len(tasks_per_worker)
            jfi = (sum_x ** 2) / (n * sum_x_sq) if sum_x_sq > 0 else 1.0
        else:
            cv = 0
            jfi = 1.0
        
        # Heterogeneity impact score
        # Compare fastest vs slowest worker completion rates
        if workers:
            fastest = max(workers, key=lambda w: w["config"].processing_speed_factor)
            slowest = min(workers, key=lambda w: w["config"].processing_speed_factor)
            speed_ratio = fastest["config"].processing_speed_factor / slowest["config"].processing_speed_factor if slowest["config"].processing_speed_factor > 0 else 1.0
        else:
            speed_ratio = 1.0
        
        return {
            "scenario_name": scenario_name,
            "scenario": scenario,
            "scheduler": scheduler,
            "iteration": iteration,
            "num_workers": len(workers),
            "device_distribution": by_device_type,
            "total_duration": duration,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "throughput": completed_tasks / duration if duration > 0 else 0,
            "avg_task_time": statistics.mean(task_times) if task_times else None,
            "p95_task_time": sorted(task_times)[int(len(task_times) * 0.95)] if task_times else None,
            "jains_fairness_index": jfi,
            "coefficient_of_variation": cv,
            "speed_heterogeneity_ratio": speed_ratio,
            "tasks_per_worker": {w["worker_id"]: w["total_completed"] for w in workers},
        }
    
    def _aggregate_iteration(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all scenarios"""
        if not results:
            return {}
        
        return {
            "num_scenarios": len(results),
            "total_tasks": sum(r["completed_tasks"] + r["failed_tasks"] for r in results),
            "completed_tasks": sum(r["completed_tasks"] for r in results),
            "failed_tasks": sum(r["failed_tasks"] for r in results),
            "avg_throughput": statistics.mean(r["throughput"] for r in results),
            "avg_jains_fairness_index": statistics.mean(r["jains_fairness_index"] for r in results),
            "scenario_results": results,
        }
    
    async def teardown(self) -> None:
        """Clean up"""
        print("HeterogeneityExperiment: Teardown complete")
    
    def get_heterogeneity_analysis(self) -> Dict[str, Any]:
        """
        Analyze impact of device heterogeneity.
        
        Returns:
            Analysis comparing homogeneous vs heterogeneous setups
        """
        analysis = {
            "scenarios": {},
            "scheduler_comparison": {},
            "recommendations": [],
        }
        
        # Analyze each scenario
        homogeneous_throughput = None
        
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            
            avg_throughput = statistics.mean(r["throughput"] for r in results)
            avg_jfi = statistics.mean(r["jains_fairness_index"] for r in results)
            avg_speed_ratio = statistics.mean(r["speed_heterogeneity_ratio"] for r in results)
            
            analysis["scenarios"][scenario_name] = {
                "avg_throughput": avg_throughput,
                "avg_fairness": avg_jfi,
                "heterogeneity_ratio": avg_speed_ratio,
            }
            
            if "homogeneous" in scenario_name:
                homogeneous_throughput = avg_throughput
        
        # Compare to homogeneous baseline
        if homogeneous_throughput:
            for scenario_name, stats in analysis["scenarios"].items():
                if homogeneous_throughput > 0:
                    stats["relative_performance"] = stats["avg_throughput"] / homogeneous_throughput
        
        # Generate recommendations
        if analysis["scenarios"]:
            best_scenario = max(
                analysis["scenarios"].items(),
                key=lambda x: x[1]["avg_throughput"]
            )
            analysis["recommendations"].append(
                f"Best performing configuration: {best_scenario[0]} with "
                f"throughput {best_scenario[1]['avg_throughput']:.2f} tasks/sec"
            )
            
            # Check if heterogeneity hurts performance
            for name, stats in analysis["scenarios"].items():
                if stats.get("relative_performance", 1.0) < 0.8:
                    analysis["recommendations"].append(
                        f"Significant performance degradation in {name} - "
                        f"consider performance-aware scheduling"
                    )
        
        return analysis
