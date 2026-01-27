"""
Failure Simulation Experiment

Tests system resilience to various failure scenarios.

Scenarios:
- Worker disconnections (graceful and abrupt)
- Task timeouts
- Network failures
- Cascading failures
- Recovery with and without checkpoints
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .base import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    WorkloadConfig,
    WorkerSimulationConfig,
    DeviceCategory,
    WorkloadType,
)


class FailureScenarioType(Enum):
    """Types of failure scenarios"""
    SINGLE_WORKER_FAILURE = "single_worker_failure"
    MULTIPLE_WORKER_FAILURE = "multiple_worker_failure"
    RANDOM_FAILURES = "random_failures"
    NETWORK_PARTITION = "network_partition"
    CASCADING_FAILURE = "cascading_failure"
    INTERMITTENT_CONNECTIVITY = "intermittent_connectivity"


@dataclass
class FailureScenarioConfig:
    """Configuration for a failure scenario"""
    scenario_type: FailureScenarioType
    
    # Failure rate (for random failures)
    failure_probability: float = 0.1  # Per-task probability
    
    # Worker failure configuration
    workers_to_fail: int = 1  # Number of workers to fail
    failure_timing: str = "random"  # "start", "middle", "end", "random"
    
    # Recovery configuration
    enable_checkpointing: bool = True
    checkpoint_interval: float = 5.0  # seconds
    recovery_delay_mean: float = 2.0  # seconds before recovery starts
    recovery_delay_std: float = 0.5
    
    # Reconnection behavior
    allow_reconnection: bool = True
    reconnection_delay_mean: float = 5.0
    reconnection_delay_std: float = 2.0
    reconnection_success_rate: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type.value,
            "failure_probability": self.failure_probability,
            "workers_to_fail": self.workers_to_fail,
            "enable_checkpointing": self.enable_checkpointing,
            "allow_reconnection": self.allow_reconnection,
        }


@dataclass
class FailureSimulationConfig(ExperimentConfig):
    """Configuration for failure simulation experiments"""
    
    # Failure scenarios to test
    scenarios: List[FailureScenarioConfig] = field(default_factory=lambda: [
        FailureScenarioConfig(
            scenario_type=FailureScenarioType.SINGLE_WORKER_FAILURE,
            workers_to_fail=1,
            enable_checkpointing=True,
        ),
        FailureScenarioConfig(
            scenario_type=FailureScenarioType.SINGLE_WORKER_FAILURE,
            workers_to_fail=1,
            enable_checkpointing=False,
        ),
        FailureScenarioConfig(
            scenario_type=FailureScenarioType.RANDOM_FAILURES,
            failure_probability=0.05,
            enable_checkpointing=True,
        ),
        FailureScenarioConfig(
            scenario_type=FailureScenarioType.MULTIPLE_WORKER_FAILURE,
            workers_to_fail=3,
            enable_checkpointing=True,
        ),
    ])
    
    # Baseline comparison (no failures)
    include_baseline: bool = True


class FailureSimulationExperiment(Experiment):
    """
    Experiment to test failure handling and recovery.
    
    Measures:
    - Failure detection time
    - Recovery time
    - Checkpoint effectiveness
    - System availability
    - Task completion rate under failures
    """
    
    def __init__(self, config: FailureSimulationConfig):
        super().__init__(config)
        self.failure_config = config
        
        # Results per scenario
        self.scenario_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Failure tracking
        self.failure_events: List[Dict[str, Any]] = []
        self.recovery_events: List[Dict[str, Any]] = []
    
    async def setup(self) -> None:
        """Initialize experiment"""
        print(f"FailureSimulationExperiment: Setting up with {len(self.failure_config.scenarios)} scenarios")
        self.scenario_results = {}
        self.failure_events = []
        self.recovery_events = []
    
    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run failure simulation tests"""
        all_results = []
        
        # Run baseline if configured
        if self.failure_config.include_baseline:
            print("  Testing baseline (no failures)...")
            baseline_result = await self._run_baseline(iteration)
            all_results.append(baseline_result)
            
            if "baseline" not in self.scenario_results:
                self.scenario_results["baseline"] = []
            self.scenario_results["baseline"].append(baseline_result)
        
        # Run each failure scenario
        for idx, scenario in enumerate(self.failure_config.scenarios):
            scenario_name = f"{scenario.scenario_type.value}_cp{scenario.enable_checkpointing}"
            print(f"  Testing scenario {idx + 1}: {scenario_name}")
            
            result = await self._run_failure_scenario(
                scenario=scenario,
                scenario_name=scenario_name,
                iteration=iteration
            )
            
            if scenario_name not in self.scenario_results:
                self.scenario_results[scenario_name] = []
            self.scenario_results[scenario_name].append(result)
            all_results.append(result)
            
            await asyncio.sleep(0.5)
        
        return self._aggregate_iteration(all_results)
    
    async def _run_baseline(self, iteration: int) -> Dict[str, Any]:
        """Run baseline test with no failures"""
        workers = self._create_workers(self.config.num_workers)
        tasks = self.generate_workload()
        
        start_time = time.time()
        results = await self._simulate_execution_no_failures(tasks, workers)
        end_time = time.time()
        
        return self._analyze_results(
            scenario_name="baseline",
            scenario_config=None,
            workers=workers,
            results=results,
            duration=end_time - start_time,
            iteration=iteration,
            failure_events=[],
            recovery_events=[]
        )
    
    async def _run_failure_scenario(
        self,
        scenario: FailureScenarioConfig,
        scenario_name: str,
        iteration: int
    ) -> Dict[str, Any]:
        """Run a single failure scenario"""
        
        workers = self._create_workers(self.config.num_workers)
        tasks = self.generate_workload()
        
        # Add checkpoint info to tasks if enabled
        if scenario.enable_checkpointing:
            for task in tasks:
                task["enable_checkpointing"] = True
                task["checkpoint_interval"] = scenario.checkpoint_interval
        
        start_time = time.time()
        
        # Run simulation with failures
        results, failures, recoveries = await self._simulate_with_failures(
            tasks=tasks,
            workers=workers,
            scenario=scenario
        )
        
        end_time = time.time()
        
        return self._analyze_results(
            scenario_name=scenario_name,
            scenario_config=scenario,
            workers=workers,
            results=results,
            duration=end_time - start_time,
            iteration=iteration,
            failure_events=failures,
            recovery_events=recoveries
        )
    
    def _create_workers(self, num_workers: int) -> List[Dict[str, Any]]:
        """Create simulated workers"""
        workers = []
        for i in range(num_workers):
            config = WorkerSimulationConfig.from_category(DeviceCategory.MID_RANGE_PC)
            workers.append({
                "worker_id": f"worker_{i}",
                "config": config,
                "status": "active",
                "current_task": None,
                "total_completed": 0,
                "failed_tasks": 0,
                "current_checkpoint": None,
                "checkpoint_progress": 0.0,
            })
        return workers
    
    async def _simulate_execution_no_failures(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate execution without any failures"""
        results = []
        worker_idx = 0
        
        for task in tasks:
            worker = workers[worker_idx]
            duration, success, error = self.simulate_task_execution(task, worker["config"])
            
            results.append({
                "task_id": task["task_id"],
                "worker_id": worker["worker_id"],
                "duration": duration,
                "success": success,
                "error": error,
                "recovered_from_checkpoint": False,
                "retry_count": 0,
            })
            
            if success:
                worker["total_completed"] += 1
            
            worker_idx = (worker_idx + 1) % len(workers)
            await asyncio.sleep(0.001)
        
        return results
    
    async def _simulate_with_failures(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]],
        scenario: FailureScenarioConfig
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Simulate execution with failures based on scenario"""
        
        results = []
        failures = []
        recoveries = []
        
        # Determine which workers/tasks will fail
        failure_points = self._plan_failures(scenario, len(tasks), len(workers))
        
        worker_idx = 0
        task_idx = 0
        
        for task in tasks:
            worker = workers[worker_idx]
            
            # Check if this is a failure point
            should_fail = (
                task_idx in failure_points.get("task_failures", set()) or
                worker["worker_id"] in failure_points.get("worker_failures", set())
            )
            
            # Simulate checkpoint progress (simplified)
            checkpoint_progress = random.uniform(0.3, 0.8) if scenario.enable_checkpointing else 0
            
            if should_fail and worker["status"] == "active":
                # Record failure
                failure_time = time.time()
                failure_event = {
                    "task_id": task["task_id"],
                    "worker_id": worker["worker_id"],
                    "failure_time": failure_time,
                    "checkpoint_available": scenario.enable_checkpointing,
                    "checkpoint_progress": checkpoint_progress if scenario.enable_checkpointing else 0,
                }
                failures.append(failure_event)
                self.failure_events.append(failure_event)
                
                # Simulate recovery
                recovery_delay = max(0.1, random.gauss(
                    scenario.recovery_delay_mean,
                    scenario.recovery_delay_std
                ))
                await asyncio.sleep(recovery_delay * 0.01)  # Scale down for simulation
                
                recovery_time = time.time()
                
                # Attempt recovery
                if scenario.enable_checkpointing:
                    # Recover from checkpoint
                    remaining_work = 1 - checkpoint_progress
                    duration, success, error = self.simulate_task_execution(task, worker["config"])
                    duration *= remaining_work  # Only do remaining work
                    
                    recovery_event = {
                        "task_id": task["task_id"],
                        "recovery_time": recovery_time - failure_time,
                        "progress_restored": checkpoint_progress * 100,
                        "success": success,
                    }
                    recoveries.append(recovery_event)
                    self.recovery_events.append(recovery_event)
                    
                    results.append({
                        "task_id": task["task_id"],
                        "worker_id": worker["worker_id"],
                        "duration": duration + recovery_delay,
                        "success": success,
                        "error": error,
                        "recovered_from_checkpoint": True,
                        "checkpoint_progress_restored": checkpoint_progress,
                        "retry_count": 1,
                    })
                else:
                    # Restart from beginning
                    duration, success, error = self.simulate_task_execution(task, worker["config"])
                    
                    recovery_event = {
                        "task_id": task["task_id"],
                        "recovery_time": recovery_time - failure_time,
                        "progress_restored": 0,
                        "success": success,
                    }
                    recoveries.append(recovery_event)
                    
                    results.append({
                        "task_id": task["task_id"],
                        "worker_id": worker["worker_id"],
                        "duration": duration + recovery_delay,
                        "success": success,
                        "error": error,
                        "recovered_from_checkpoint": False,
                        "checkpoint_progress_restored": 0,
                        "retry_count": 1,
                    })
                
                if success:
                    worker["total_completed"] += 1
                else:
                    worker["failed_tasks"] += 1
            else:
                # Normal execution
                duration, success, error = self.simulate_task_execution(task, worker["config"])
                
                results.append({
                    "task_id": task["task_id"],
                    "worker_id": worker["worker_id"],
                    "duration": duration,
                    "success": success,
                    "error": error,
                    "recovered_from_checkpoint": False,
                    "retry_count": 0,
                })
                
                if success:
                    worker["total_completed"] += 1
            
            worker_idx = (worker_idx + 1) % len(workers)
            task_idx += 1
            await asyncio.sleep(0.001)
        
        return results, failures, recoveries
    
    def _plan_failures(
        self,
        scenario: FailureScenarioConfig,
        num_tasks: int,
        num_workers: int
    ) -> Dict[str, Any]:
        """Plan when and where failures will occur"""
        failure_points = {
            "task_failures": set(),
            "worker_failures": set(),
        }
        
        if scenario.scenario_type == FailureScenarioType.SINGLE_WORKER_FAILURE:
            # Fail one worker's tasks at a specific point
            if scenario.failure_timing == "start":
                failure_points["task_failures"].add(0)
            elif scenario.failure_timing == "middle":
                failure_points["task_failures"].add(num_tasks // 2)
            elif scenario.failure_timing == "end":
                failure_points["task_failures"].add(num_tasks - 1)
            else:  # random
                failure_points["task_failures"].add(random.randint(0, num_tasks - 1))
        
        elif scenario.scenario_type == FailureScenarioType.RANDOM_FAILURES:
            # Random task failures
            for i in range(num_tasks):
                if random.random() < scenario.failure_probability:
                    failure_points["task_failures"].add(i)
        
        elif scenario.scenario_type == FailureScenarioType.MULTIPLE_WORKER_FAILURE:
            # Multiple tasks fail (simulating multiple workers failing)
            num_failures = min(scenario.workers_to_fail * 2, num_tasks)
            failure_indices = random.sample(range(num_tasks), num_failures)
            failure_points["task_failures"].update(failure_indices)
        
        return failure_points
    
    def _analyze_results(
        self,
        scenario_name: str,
        scenario_config: Optional[FailureScenarioConfig],
        workers: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        duration: float,
        iteration: int,
        failure_events: List[Dict[str, Any]],
        recovery_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze scenario results"""
        
        completed_tasks = sum(1 for r in results if r["success"])
        failed_tasks = sum(1 for r in results if not r["success"])
        recovered_tasks = sum(1 for r in results if r.get("recovered_from_checkpoint", False))
        
        task_times = [r["duration"] for r in results if r["success"]]
        recovery_times = [re["recovery_time"] for re in recovery_events]
        
        # Calculate metrics
        analysis = {
            "scenario_name": scenario_name,
            "scenario_config": scenario_config.to_dict() if scenario_config else None,
            "iteration": iteration,
            "duration": duration,
            "total_tasks": len(results),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": completed_tasks / len(results) * 100 if results else 0,
            "throughput": completed_tasks / duration if duration > 0 else 0,
            "avg_task_time": statistics.mean(task_times) if task_times else None,
            "p95_task_time": sorted(task_times)[int(len(task_times) * 0.95)] if task_times else None,
            
            # Failure metrics
            "total_failures": len(failure_events),
            "recovered_tasks": recovered_tasks,
            "recovery_rate": recovered_tasks / len(failure_events) * 100 if failure_events else 100,
            
            # Recovery metrics
            "avg_recovery_time": statistics.mean(recovery_times) if recovery_times else None,
            "min_recovery_time": min(recovery_times) if recovery_times else None,
            "max_recovery_time": max(recovery_times) if recovery_times else None,
            
            # Checkpoint metrics
            "checkpoint_enabled": scenario_config.enable_checkpointing if scenario_config else False,
            "avg_progress_restored": statistics.mean(
                [r.get("checkpoint_progress_restored", 0) * 100 for r in results if r.get("recovered_from_checkpoint")]
            ) if any(r.get("recovered_from_checkpoint") for r in results) else 0,
            
            # Retry metrics
            "tasks_retried": sum(1 for r in results if r.get("retry_count", 0) > 0),
            "avg_retries": statistics.mean([r.get("retry_count", 0) for r in results]),
        }
        
        return analysis
    
    def _aggregate_iteration(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all scenarios"""
        if not results:
            return {}
        
        return {
            "num_scenarios": len(results),
            "total_tasks": sum(r["total_tasks"] for r in results),
            "completed_tasks": sum(r["completed_tasks"] for r in results),
            "failed_tasks": sum(r["failed_tasks"] for r in results),
            "total_failures": sum(r["total_failures"] for r in results),
            "avg_recovery_time": statistics.mean(
                [r["avg_recovery_time"] for r in results if r["avg_recovery_time"]]
            ) if any(r["avg_recovery_time"] for r in results) else None,
            "scenario_results": results,
        }
    
    async def teardown(self) -> None:
        """Clean up"""
        print("FailureSimulationExperiment: Teardown complete")
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """
        Analyze failure handling effectiveness.
        
        Returns:
            Analysis comparing checkpoint vs non-checkpoint recovery
        """
        analysis = {
            "checkpoint_comparison": {},
            "recovery_statistics": {},
            "recommendations": [],
        }
        
        # Compare scenarios with and without checkpointing
        checkpoint_scenarios = []
        no_checkpoint_scenarios = []
        
        for name, results in self.scenario_results.items():
            if not results:
                continue
            
            avg_recovery_time = statistics.mean(
                [r["avg_recovery_time"] for r in results if r["avg_recovery_time"]]
            ) if any(r["avg_recovery_time"] for r in results) else None
            
            avg_progress_restored = statistics.mean(
                [r["avg_progress_restored"] for r in results]
            )
            
            avg_success_rate = statistics.mean([r["success_rate"] for r in results])
            
            scenario_stats = {
                "avg_recovery_time": avg_recovery_time,
                "avg_progress_restored": avg_progress_restored,
                "avg_success_rate": avg_success_rate,
            }
            
            if "cpTrue" in name:
                checkpoint_scenarios.append((name, scenario_stats))
            elif "cpFalse" in name:
                no_checkpoint_scenarios.append((name, scenario_stats))
        
        # Calculate checkpoint benefit
        if checkpoint_scenarios and no_checkpoint_scenarios:
            cp_recovery_times = [s[1]["avg_recovery_time"] for s in checkpoint_scenarios if s[1]["avg_recovery_time"]]
            no_cp_recovery_times = [s[1]["avg_recovery_time"] for s in no_checkpoint_scenarios if s[1]["avg_recovery_time"]]
            
            if cp_recovery_times and no_cp_recovery_times:
                avg_cp_time = statistics.mean(cp_recovery_times)
                avg_no_cp_time = statistics.mean(no_cp_recovery_times)
                
                analysis["checkpoint_comparison"] = {
                    "with_checkpoint_recovery_time": avg_cp_time,
                    "without_checkpoint_recovery_time": avg_no_cp_time,
                    "time_saved": avg_no_cp_time - avg_cp_time,
                    "speedup_factor": avg_no_cp_time / avg_cp_time if avg_cp_time > 0 else 1.0,
                }
                
                analysis["recommendations"].append(
                    f"Checkpointing reduces recovery time by {((avg_no_cp_time - avg_cp_time) / avg_no_cp_time * 100):.1f}%"
                )
        
        # Recovery statistics
        all_recovery_times = []
        for name, results in self.scenario_results.items():
            for r in results:
                if r["avg_recovery_time"]:
                    all_recovery_times.append(r["avg_recovery_time"])
        
        if all_recovery_times:
            analysis["recovery_statistics"] = {
                "overall_avg_recovery_time": statistics.mean(all_recovery_times),
                "min_recovery_time": min(all_recovery_times),
                "max_recovery_time": max(all_recovery_times),
            }
        
        return analysis
