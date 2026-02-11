"""
Energy Constraint Experiment

Tests system behavior and performance with energy-constrained devices.

Scenarios:
- All devices on battery
- Mixed charging/battery states
- Low battery scenarios
- Energy-aware task scheduling
- Battery drain patterns
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


class EnergyScenarioType(Enum):
    """Types of energy scenarios"""
    ALL_CHARGING = "all_charging"
    ALL_BATTERY = "all_battery"
    MIXED_POWER = "mixed_power"
    LOW_BATTERY = "low_battery"
    CRITICAL_BATTERY = "critical_battery"
    PROGRESSIVE_DRAIN = "progressive_drain"


@dataclass
class EnergyScenarioConfig:
    """Configuration for an energy scenario"""
    scenario_type: EnergyScenarioType
    
    # Battery levels (percentage)
    initial_battery_min: float = 50.0
    initial_battery_max: float = 100.0
    
    # Charging distribution
    charging_probability: float = 0.0  # Probability a device is charging
    
    # Energy consumption
    energy_per_task_mean: float = 5.0  # mWh
    energy_per_task_std: float = 1.0
    
    # Low battery behavior
    low_battery_threshold: float = 20.0
    critical_battery_threshold: float = 10.0
    pause_on_low_battery: bool = True
    
    # Task rejection
    reject_tasks_below_battery: float = 10.0  # Reject if below this level
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type.value,
            "initial_battery_min": self.initial_battery_min,
            "initial_battery_max": self.initial_battery_max,
            "charging_probability": self.charging_probability,
            "energy_per_task_mean": self.energy_per_task_mean,
            "low_battery_threshold": self.low_battery_threshold,
        }


@dataclass
class EnergyConstraintConfig(ExperimentConfig):
    """Configuration for energy constraint experiments"""
    
    # Energy scenarios to test
    scenarios: List[EnergyScenarioConfig] = field(default_factory=lambda: [
        EnergyScenarioConfig(
            scenario_type=EnergyScenarioType.ALL_CHARGING,
            charging_probability=1.0,
        ),
        EnergyScenarioConfig(
            scenario_type=EnergyScenarioType.ALL_BATTERY,
            charging_probability=0.0,
            initial_battery_min=80.0,
            initial_battery_max=100.0,
        ),
        EnergyScenarioConfig(
            scenario_type=EnergyScenarioType.MIXED_POWER,
            charging_probability=0.5,
            initial_battery_min=50.0,
            initial_battery_max=100.0,
        ),
        EnergyScenarioConfig(
            scenario_type=EnergyScenarioType.LOW_BATTERY,
            charging_probability=0.0,
            initial_battery_min=15.0,
            initial_battery_max=30.0,
        ),
        EnergyScenarioConfig(
            scenario_type=EnergyScenarioType.PROGRESSIVE_DRAIN,
            charging_probability=0.0,
            initial_battery_min=60.0,
            initial_battery_max=80.0,
            energy_per_task_mean=10.0,  # Higher consumption
        ),
    ])
    
    # Device mix (affects energy profiles)
    device_mix: Dict[str, float] = field(default_factory=lambda: {
        "mid_range_mobile": 0.5,
        "high_end_mobile": 0.3,
        "low_end_mobile": 0.2,
    })
    
    # Energy-aware scheduling
    test_energy_aware_scheduling: bool = True


class EnergyConstraintExperiment(Experiment):
    """
    Experiment to test system with energy-constrained devices.
    
    Measures:
    - Task completion under energy constraints
    - Battery drain patterns
    - Energy-aware scheduling effectiveness
    - Worker dropout due to low battery
    - Total energy consumption
    """
    
    def __init__(self, config: EnergyConstraintConfig):
        super().__init__(config)
        self.energy_config = config
        
        # Results per scenario
        self.scenario_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Energy tracking
        self.energy_events: List[Dict[str, Any]] = []
    
    async def setup(self) -> None:
        """Initialize experiment"""
        print(f"EnergyConstraintExperiment: Setting up with {len(self.energy_config.scenarios)} scenarios")
        self.scenario_results = {}
        self.energy_events = []
    
    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run energy constraint tests"""
        all_results = []
        
        for idx, scenario in enumerate(self.energy_config.scenarios):
            scenario_name = scenario.scenario_type.value
            print(f"  Testing scenario {idx + 1}: {scenario_name}")
            
            # Run with standard scheduling
            result_std = await self._run_energy_scenario(
                scenario=scenario,
                scenario_name=f"{scenario_name}_std",
                energy_aware=False,
                iteration=iteration
            )
            all_results.append(result_std)
            
            if scenario_name not in self.scenario_results:
                self.scenario_results[scenario_name] = []
            self.scenario_results[scenario_name].append(result_std)
            
            # Also test with energy-aware scheduling if configured
            if self.energy_config.test_energy_aware_scheduling:
                result_ea = await self._run_energy_scenario(
                    scenario=scenario,
                    scenario_name=f"{scenario_name}_energy_aware",
                    energy_aware=True,
                    iteration=iteration
                )
                all_results.append(result_ea)
            
            await asyncio.sleep(0.5)
        
        return self._aggregate_iteration(all_results)
    
    async def _run_energy_scenario(
        self,
        scenario: EnergyScenarioConfig,
        scenario_name: str,
        energy_aware: bool,
        iteration: int
    ) -> Dict[str, Any]:
        """Run a single energy scenario"""
        
        # Create workers with energy state
        workers = self._create_energy_workers(scenario)
        
        # Generate tasks
        tasks = self.generate_workload()
        
        # Run simulation
        start_time = time.time()
        
        results, energy_data = await self._simulate_with_energy(
            tasks=tasks,
            workers=workers,
            scenario=scenario,
            energy_aware=energy_aware
        )
        
        end_time = time.time()
        
        return self._analyze_energy_results(
            scenario_name=scenario_name,
            scenario=scenario,
            energy_aware=energy_aware,
            workers=workers,
            results=results,
            energy_data=energy_data,
            duration=end_time - start_time,
            iteration=iteration
        )
    
    def _create_energy_workers(
        self,
        scenario: EnergyScenarioConfig
    ) -> List[Dict[str, Any]]:
        """Create workers with energy state based on scenario"""
        workers = []
        device_mix = self.energy_config.device_mix
        
        # Normalize device mix
        total = sum(device_mix.values())
        normalized = {k: v/total for k, v in device_mix.items()}
        
        for i in range(self.config.num_workers):
            # Select device type
            r = random.random()
            cumulative = 0
            device_type = "mid_range_mobile"
            
            for dtype, prob in normalized.items():
                cumulative += prob
                if r <= cumulative:
                    device_type = dtype
                    break
            
            category = DeviceCategory(device_type)
            config = WorkerSimulationConfig.from_category(category)
            
            # Set battery state based on scenario
            is_charging = random.random() < scenario.charging_probability
            
            if is_charging:
                battery_level = 100.0
            else:
                battery_level = random.uniform(
                    scenario.initial_battery_min,
                    scenario.initial_battery_max
                )
            
            workers.append({
                "worker_id": f"worker_{i}",
                "device_type": device_type,
                "config": config,
                "is_charging": is_charging,
                "battery_level": battery_level,
                "initial_battery": battery_level,
                "battery_capacity_mwh": config.battery_capacity_mwh,
                "total_energy_consumed": 0.0,
                "tasks_completed": 0,
                "tasks_rejected": 0,
                "status": "active",
                "battery_history": [(0, battery_level)],
            })
        
        return workers
    
    async def _simulate_with_energy(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]],
        scenario: EnergyScenarioConfig,
        energy_aware: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Simulate execution with energy constraints"""
        
        results = []
        energy_data = []
        
        task_idx = 0
        sim_time = 0
        
        for task in tasks:
            # Select worker
            if energy_aware:
                worker = self._select_energy_aware_worker(workers, scenario)
            else:
                worker = self._select_round_robin_worker(workers, task_idx)
            
            if worker is None:
                # No workers available - all low battery
                results.append({
                    "task_id": task["task_id"],
                    "worker_id": None,
                    "status": "rejected",
                    "reason": "no_available_workers",
                    "success": False,
                })
                continue
            
            # Check if worker can accept task
            if worker["battery_level"] < scenario.reject_tasks_below_battery:
                worker["tasks_rejected"] += 1
                worker["status"] = "low_battery"
                
                results.append({
                    "task_id": task["task_id"],
                    "worker_id": worker["worker_id"],
                    "status": "rejected",
                    "reason": "low_battery",
                    "battery_level": worker["battery_level"],
                    "success": False,
                })
                continue
            
            # Simulate task execution
            duration, success, error = self.simulate_task_execution(task, worker["config"])
            
            # Calculate energy consumption
            energy_consumed = random.gauss(
                scenario.energy_per_task_mean,
                scenario.energy_per_task_std
            )
            energy_consumed = max(0.1, energy_consumed)
            
            # Update battery level (if not charging)
            if not worker["is_charging"]:
                battery_drain = (energy_consumed / worker["battery_capacity_mwh"]) * 100
                worker["battery_level"] = max(0, worker["battery_level"] - battery_drain)
            
            worker["total_energy_consumed"] += energy_consumed
            
            if success:
                worker["tasks_completed"] += 1
            
            sim_time += duration
            worker["battery_history"].append((sim_time, worker["battery_level"]))
            
            # Record energy event
            energy_event = {
                "task_id": task["task_id"],
                "worker_id": worker["worker_id"],
                "energy_consumed": energy_consumed,
                "battery_before": worker["battery_level"] + (energy_consumed / worker["battery_capacity_mwh"]) * 100,
                "battery_after": worker["battery_level"],
                "is_charging": worker["is_charging"],
            }
            energy_data.append(energy_event)
            
            results.append({
                "task_id": task["task_id"],
                "worker_id": worker["worker_id"],
                "duration": duration,
                "success": success,
                "error": error,
                "energy_consumed": energy_consumed,
                "battery_level_after": worker["battery_level"],
            })
            
            task_idx += 1
            await asyncio.sleep(0.001)
        
        return results, energy_data
    
    def _select_round_robin_worker(
        self,
        workers: List[Dict[str, Any]],
        task_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Select worker using round-robin"""
        active_workers = [w for w in workers if w["status"] == "active"]
        if not active_workers:
            return None
        return active_workers[task_idx % len(active_workers)]
    
    def _select_energy_aware_worker(
        self,
        workers: List[Dict[str, Any]],
        scenario: EnergyScenarioConfig
    ) -> Optional[Dict[str, Any]]:
        """Select worker based on energy considerations"""
        active_workers = [w for w in workers if w["status"] == "active"]
        
        if not active_workers:
            return None
        
        # Priority: charging devices first, then by battery level
        def worker_score(w):
            if w["is_charging"]:
                return 1000  # High priority
            return w["battery_level"]  # Priority by battery level
        
        sorted_workers = sorted(active_workers, key=worker_score, reverse=True)
        return sorted_workers[0]
    
    def _analyze_energy_results(
        self,
        scenario_name: str,
        scenario: EnergyScenarioConfig,
        energy_aware: bool,
        workers: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        energy_data: List[Dict[str, Any]],
        duration: float,
        iteration: int
    ) -> Dict[str, Any]:
        """Analyze energy scenario results"""
        
        completed_tasks = sum(1 for r in results if r.get("success", False))
        rejected_tasks = sum(1 for r in results if r.get("status") == "rejected")
        failed_tasks = sum(1 for r in results if not r.get("success", True) and r.get("status") != "rejected")
        
        # Energy metrics
        total_energy = sum(r.get("energy_consumed", 0) for r in results)
        energy_per_task = total_energy / completed_tasks if completed_tasks > 0 else 0
        
        # Battery metrics
        final_batteries = [w["battery_level"] for w in workers]
        initial_batteries = [w["initial_battery"] for w in workers]
        battery_drain = [init - final for init, final in zip(initial_batteries, final_batteries)]
        
        workers_depleted = sum(1 for w in workers if w["battery_level"] < 10)
        workers_low_battery = sum(1 for w in workers if w["battery_level"] < 20)
        
        # Task distribution by battery state
        tasks_by_charging = sum(1 for e in energy_data if e["is_charging"])
        tasks_by_battery = sum(1 for e in energy_data if not e["is_charging"])
        
        return {
            "scenario_name": scenario_name,
            "scenario_config": scenario.to_dict(),
            "energy_aware_scheduling": energy_aware,
            "iteration": iteration,
            "duration": duration,
            
            # Task metrics
            "total_tasks": len(results),
            "completed_tasks": completed_tasks,
            "rejected_tasks": rejected_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": completed_tasks / len(results) * 100 if results else 0,
            "rejection_rate": rejected_tasks / len(results) * 100 if results else 0,
            "throughput": completed_tasks / duration if duration > 0 else 0,
            
            # Energy metrics
            "total_energy_consumed_mwh": total_energy,
            "avg_energy_per_task_mwh": energy_per_task,
            "energy_efficiency": completed_tasks / total_energy if total_energy > 0 else 0,
            
            # Battery metrics
            "avg_initial_battery": statistics.mean(initial_batteries),
            "avg_final_battery": statistics.mean(final_batteries),
            "avg_battery_drain": statistics.mean(battery_drain) if battery_drain else 0,
            "max_battery_drain": max(battery_drain) if battery_drain else 0,
            "workers_depleted": workers_depleted,
            "workers_low_battery": workers_low_battery,
            
            # Distribution metrics
            "tasks_on_charging_devices": tasks_by_charging,
            "tasks_on_battery_devices": tasks_by_battery,
            "charging_device_utilization": tasks_by_charging / len(results) * 100 if results else 0,
            
            # Per-worker metrics
            "worker_stats": [
                {
                    "worker_id": w["worker_id"],
                    "device_type": w["device_type"],
                    "is_charging": w["is_charging"],
                    "initial_battery": w["initial_battery"],
                    "final_battery": w["battery_level"],
                    "tasks_completed": w["tasks_completed"],
                    "tasks_rejected": w["tasks_rejected"],
                    "energy_consumed": w["total_energy_consumed"],
                }
                for w in workers
            ],
        }
    
    def _aggregate_iteration(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all scenarios"""
        if not results:
            return {}
        
        return {
            "num_scenarios": len(results),
            "total_tasks": sum(r["total_tasks"] for r in results),
            "completed_tasks": sum(r["completed_tasks"] for r in results),
            "rejected_tasks": sum(r["rejected_tasks"] for r in results),
            "total_energy_consumed": sum(r["total_energy_consumed_mwh"] for r in results),
            "avg_throughput": statistics.mean(r["throughput"] for r in results),
            "avg_success_rate": statistics.mean(r["success_rate"] for r in results),
            "scenario_results": results,
        }
    
    async def teardown(self) -> None:
        """Clean up"""
        print("EnergyConstraintExperiment: Teardown complete")
    
    def get_energy_analysis(self) -> Dict[str, Any]:
        """
        Analyze energy consumption patterns and constraints impact.
        
        Returns:
            Analysis of energy efficiency and constraint handling
        """
        analysis = {
            "scenario_comparison": {},
            "scheduling_comparison": {},
            "energy_efficiency": {},
            "recommendations": [],
        }
        
        # Analyze each scenario
        for scenario_name, results in self.scenario_results.items():
            if not results:
                continue
            
            avg_success_rate = statistics.mean(r["success_rate"] for r in results)
            avg_rejection_rate = statistics.mean(r["rejection_rate"] for r in results)
            avg_energy_per_task = statistics.mean(r["avg_energy_per_task_mwh"] for r in results)
            avg_battery_drain = statistics.mean(r["avg_battery_drain"] for r in results)
            
            analysis["scenario_comparison"][scenario_name] = {
                "success_rate": avg_success_rate,
                "rejection_rate": avg_rejection_rate,
                "avg_energy_per_task": avg_energy_per_task,
                "avg_battery_drain": avg_battery_drain,
            }
        
        # Compare energy-aware vs standard scheduling
        standard_results = []
        energy_aware_results = []
        
        for results in self.scenario_results.values():
            for r in results:
                if r.get("energy_aware_scheduling"):
                    energy_aware_results.append(r)
                else:
                    standard_results.append(r)
        
        if standard_results and energy_aware_results:
            std_rejection = statistics.mean(r["rejection_rate"] for r in standard_results)
            ea_rejection = statistics.mean(r["rejection_rate"] for r in energy_aware_results)
            
            std_success = statistics.mean(r["success_rate"] for r in standard_results)
            ea_success = statistics.mean(r["success_rate"] for r in energy_aware_results)
            
            analysis["scheduling_comparison"] = {
                "standard_rejection_rate": std_rejection,
                "energy_aware_rejection_rate": ea_rejection,
                "rejection_reduction": std_rejection - ea_rejection,
                "standard_success_rate": std_success,
                "energy_aware_success_rate": ea_success,
            }
            
            if ea_rejection < std_rejection:
                analysis["recommendations"].append(
                    f"Energy-aware scheduling reduces task rejections by "
                    f"{(std_rejection - ea_rejection):.1f}%"
                )
        
        # Energy efficiency recommendations
        if analysis["scenario_comparison"]:
            best_scenario = max(
                analysis["scenario_comparison"].items(),
                key=lambda x: x[1]["success_rate"]
            )
            worst_scenario = min(
                analysis["scenario_comparison"].items(),
                key=lambda x: x[1]["success_rate"]
            )
            
            analysis["recommendations"].append(
                f"Best performing: {best_scenario[0]} ({best_scenario[1]['success_rate']:.1f}% success)"
            )
            
            if worst_scenario[1]["rejection_rate"] > 10:
                analysis["recommendations"].append(
                    f"High rejection rate in {worst_scenario[0]}: "
                    f"Consider energy-aware scheduling or maintaining higher battery levels"
                )
        
        return analysis
