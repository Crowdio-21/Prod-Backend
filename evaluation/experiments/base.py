"""
Base Experiment Classes

Provides abstract base class and common configurations for all
experimental scenarios.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import random


class WorkloadType(Enum):
    """Types of computational workloads"""
    CPU_LIGHT = "cpu_light"  # Simple calculations
    CPU_HEAVY = "cpu_heavy"  # Complex calculations
    MEMORY_LIGHT = "memory_light"  # Low memory usage
    MEMORY_HEAVY = "memory_heavy"  # High memory usage
    IO_BOUND = "io_bound"  # I/O intensive
    MIXED = "mixed"  # Combination of above


class DeviceCategory(Enum):
    """Categories of devices for heterogeneity testing"""
    HIGH_END_PC = "high_end_pc"
    MID_RANGE_PC = "mid_range_pc"
    LOW_END_PC = "low_end_pc"
    HIGH_END_MOBILE = "high_end_mobile"
    MID_RANGE_MOBILE = "mid_range_mobile"
    LOW_END_MOBILE = "low_end_mobile"
    EDGE_DEVICE = "edge_device"  # Raspberry Pi, etc.


@dataclass
class WorkloadConfig:
    """Configuration for workload generation"""
    workload_type: WorkloadType = WorkloadType.CPU_LIGHT
    
    # Task configuration
    num_tasks: int = 100
    task_duration_mean: float = 1.0  # seconds
    task_duration_std: float = 0.3
    
    # Data size (bytes)
    input_size_mean: int = 1024
    input_size_std: int = 256
    output_size_mean: int = 512
    output_size_std: int = 128
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_interval: float = 5.0  # seconds
    checkpoint_size_mean: int = 2048
    
    # Dependencies
    has_dependencies: bool = False
    dependency_probability: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_type": self.workload_type.value,
            "num_tasks": self.num_tasks,
            "task_duration_mean": self.task_duration_mean,
            "task_duration_std": self.task_duration_std,
            "input_size_mean": self.input_size_mean,
            "output_size_mean": self.output_size_mean,
            "enable_checkpointing": self.enable_checkpointing,
        }


@dataclass
class WorkerSimulationConfig:
    """Configuration for simulated worker behavior"""
    device_category: DeviceCategory = DeviceCategory.MID_RANGE_PC
    
    # Performance characteristics
    processing_speed_factor: float = 1.0  # Relative to baseline
    memory_limit_mb: int = 4096
    
    # Reliability
    failure_probability: float = 0.0  # Per-task failure probability
    disconnect_probability: float = 0.0  # Per-minute disconnect probability
    reconnect_delay_mean: float = 5.0  # seconds
    
    # Energy (for mobile/battery devices)
    battery_capacity_mwh: float = 15000.0
    initial_battery_level: float = 100.0
    is_charging: bool = False
    energy_per_task: float = 5.0  # mWh
    
    # Network
    latency_mean_ms: float = 50.0
    latency_std_ms: float = 10.0
    bandwidth_mbps: float = 100.0
    
    @classmethod
    def from_category(cls, category: DeviceCategory) -> "WorkerSimulationConfig":
        """Create config based on device category"""
        configs = {
            DeviceCategory.HIGH_END_PC: cls(
                device_category=category,
                processing_speed_factor=2.0,
                memory_limit_mb=32768,
                failure_probability=0.001,
                latency_mean_ms=20.0,
                bandwidth_mbps=1000.0,
            ),
            DeviceCategory.MID_RANGE_PC: cls(
                device_category=category,
                processing_speed_factor=1.0,
                memory_limit_mb=8192,
                failure_probability=0.005,
                latency_mean_ms=50.0,
                bandwidth_mbps=100.0,
            ),
            DeviceCategory.LOW_END_PC: cls(
                device_category=category,
                processing_speed_factor=0.5,
                memory_limit_mb=4096,
                failure_probability=0.01,
                latency_mean_ms=100.0,
                bandwidth_mbps=50.0,
            ),
            DeviceCategory.HIGH_END_MOBILE: cls(
                device_category=category,
                processing_speed_factor=0.8,
                memory_limit_mb=8192,
                failure_probability=0.02,
                battery_capacity_mwh=20000.0,
                is_charging=False,
                latency_mean_ms=80.0,
                bandwidth_mbps=100.0,
            ),
            DeviceCategory.MID_RANGE_MOBILE: cls(
                device_category=category,
                processing_speed_factor=0.4,
                memory_limit_mb=4096,
                failure_probability=0.05,
                battery_capacity_mwh=15000.0,
                is_charging=False,
                latency_mean_ms=100.0,
                bandwidth_mbps=50.0,
            ),
            DeviceCategory.LOW_END_MOBILE: cls(
                device_category=category,
                processing_speed_factor=0.2,
                memory_limit_mb=2048,
                failure_probability=0.1,
                battery_capacity_mwh=10000.0,
                is_charging=False,
                latency_mean_ms=150.0,
                bandwidth_mbps=20.0,
            ),
            DeviceCategory.EDGE_DEVICE: cls(
                device_category=category,
                processing_speed_factor=0.3,
                memory_limit_mb=1024,
                failure_probability=0.02,
                battery_capacity_mwh=20000.0,
                is_charging=True,  # Usually plugged in
                latency_mean_ms=50.0,
                bandwidth_mbps=100.0,
            ),
        }
        return configs.get(category, cls())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_category": self.device_category.value,
            "processing_speed_factor": self.processing_speed_factor,
            "memory_limit_mb": self.memory_limit_mb,
            "failure_probability": self.failure_probability,
            "disconnect_probability": self.disconnect_probability,
            "battery_capacity_mwh": self.battery_capacity_mwh,
            "initial_battery_level": self.initial_battery_level,
            "is_charging": self.is_charging,
            "latency_mean_ms": self.latency_mean_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
        }


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str = ""
    
    # Timing
    duration_seconds: float = 300.0  # 5 minutes default
    warmup_seconds: float = 30.0
    cooldown_seconds: float = 10.0
    
    # Iterations
    num_iterations: int = 1
    
    # Workload
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    
    # Workers
    num_workers: int = 5
    worker_configs: List[WorkerSimulationConfig] = field(default_factory=list)
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None
    
    # Output
    output_dir: str = "./evaluation_results"
    save_detailed_logs: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "num_iterations": self.num_iterations,
            "workload": self.workload.to_dict(),
            "num_workers": self.num_workers,
            "worker_configs": [w.to_dict() for w in self.worker_configs],
            "random_seed": self.random_seed,
        }


@dataclass
class ExperimentResult:
    """Results from an experiment run"""
    config: ExperimentConfig
    
    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    
    # Task metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Performance metrics
    throughput: float = 0.0  # tasks/second
    avg_task_time: Optional[float] = None
    p95_task_time: Optional[float] = None
    
    # Load balancing
    jains_fairness_index: float = 0.0
    load_imbalance_ratio: float = 0.0
    
    # Failure/recovery
    total_failures: int = 0
    avg_recovery_time: Optional[float] = None
    recovery_success_rate: float = 0.0
    
    # Energy
    total_energy_consumed_mwh: float = 0.0
    avg_energy_per_task: float = 0.0
    
    # Communication
    total_bytes_transferred: int = 0
    avg_latency_ms: Optional[float] = None
    
    # Detailed results by iteration
    iteration_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Raw data for further analysis
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_duration(self) -> None:
        """Calculate experiment duration"""
        if self.completed_at and self.started_at:
            self.duration_seconds = self.completed_at - self.started_at
    
    def to_dict(self) -> Dict[str, Any]:
        self.calculate_duration()
        return {
            "config": self.config.to_dict(),
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "completed_at": datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "task_metrics": {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": self.completed_tasks / self.total_tasks * 100 if self.total_tasks > 0 else 0,
            },
            "performance_metrics": {
                "throughput_tasks_per_sec": self.throughput,
                "avg_task_time_sec": self.avg_task_time,
                "p95_task_time_sec": self.p95_task_time,
            },
            "load_balancing": {
                "jains_fairness_index": self.jains_fairness_index,
                "load_imbalance_ratio": self.load_imbalance_ratio,
            },
            "failure_recovery": {
                "total_failures": self.total_failures,
                "avg_recovery_time_sec": self.avg_recovery_time,
                "recovery_success_rate": self.recovery_success_rate,
            },
            "energy": {
                "total_energy_consumed_mwh": self.total_energy_consumed_mwh,
                "avg_energy_per_task_mwh": self.avg_energy_per_task,
            },
            "communication": {
                "total_bytes_transferred": self.total_bytes_transferred,
                "avg_latency_ms": self.avg_latency_ms,
            },
            "iteration_results": self.iteration_results,
        }
    
    def save(self, filepath: str) -> None:
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"ExperimentResult: Saved to {filepath}")


class Experiment(ABC):
    """
    Abstract base class for experiments.
    
    Subclasses implement specific experimental scenarios.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.result: Optional[ExperimentResult] = None
        
        # Set random seed if specified
        if config.random_seed is not None:
            random.seed(config.random_seed)
        
        # Metrics collectors (to be initialized)
        self.metrics_collector = None
        self.energy_tracker = None
        self.load_analyzer = None
        self.failure_tracker = None
        self.comm_tracker = None
    
    @abstractmethod
    async def setup(self) -> None:
        """
        Set up the experiment.
        
        Override to initialize workers, connections, etc.
        """
        pass
    
    @abstractmethod
    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Run a single iteration of the experiment.
        
        Args:
            iteration: Iteration number (0-indexed)
        
        Returns:
            Dict with iteration results
        """
        pass
    
    @abstractmethod
    async def teardown(self) -> None:
        """
        Clean up after experiment.
        
        Override to disconnect workers, save data, etc.
        """
        pass
    
    async def run(self) -> ExperimentResult:
        """
        Run the complete experiment.
        
        Returns:
            ExperimentResult with all metrics
        """
        self.result = ExperimentResult(config=self.config)
        
        try:
            print(f"Experiment '{self.config.name}': Starting setup...")
            await self.setup()
            
            # Warmup period
            if self.config.warmup_seconds > 0:
                print(f"Experiment '{self.config.name}': Warmup ({self.config.warmup_seconds}s)...")
                await asyncio.sleep(self.config.warmup_seconds)
            
            # Run iterations
            for i in range(self.config.num_iterations):
                print(f"Experiment '{self.config.name}': Running iteration {i + 1}/{self.config.num_iterations}...")
                iteration_result = await self.run_iteration(i)
                self.result.iteration_results.append(iteration_result)
                
                # Short pause between iterations
                if i < self.config.num_iterations - 1:
                    await asyncio.sleep(1.0)
            
            # Cooldown period
            if self.config.cooldown_seconds > 0:
                print(f"Experiment '{self.config.name}': Cooldown ({self.config.cooldown_seconds}s)...")
                await asyncio.sleep(self.config.cooldown_seconds)
            
            # Aggregate results
            await self._aggregate_results()
            
        except Exception as e:
            print(f"Experiment '{self.config.name}': Error - {e}")
            raise
        
        finally:
            print(f"Experiment '{self.config.name}': Teardown...")
            await self.teardown()
            self.result.completed_at = time.time()
        
        return self.result
    
    async def _aggregate_results(self) -> None:
        """Aggregate results from all iterations"""
        if not self.result or not self.result.iteration_results:
            return
        
        # Average metrics across iterations
        iterations = self.result.iteration_results
        n = len(iterations)
        
        def avg(key: str) -> Optional[float]:
            values = [it.get(key) for it in iterations if it.get(key) is not None]
            return sum(values) / len(values) if values else None
        
        self.result.throughput = avg("throughput") or 0.0
        self.result.avg_task_time = avg("avg_task_time")
        self.result.p95_task_time = avg("p95_task_time")
        self.result.jains_fairness_index = avg("jains_fairness_index") or 0.0
        self.result.load_imbalance_ratio = avg("load_imbalance_ratio") or 0.0
        self.result.avg_recovery_time = avg("avg_recovery_time")
        self.result.recovery_success_rate = avg("recovery_success_rate") or 0.0
        self.result.avg_energy_per_task = avg("avg_energy_per_task") or 0.0
        self.result.avg_latency_ms = avg("avg_latency_ms")
        
        # Sum metrics across iterations
        self.result.total_tasks = sum(it.get("total_tasks", 0) for it in iterations)
        self.result.completed_tasks = sum(it.get("completed_tasks", 0) for it in iterations)
        self.result.failed_tasks = sum(it.get("failed_tasks", 0) for it in iterations)
        self.result.total_failures = sum(it.get("total_failures", 0) for it in iterations)
        self.result.total_energy_consumed_mwh = sum(it.get("energy_consumed", 0) for it in iterations)
        self.result.total_bytes_transferred = sum(it.get("bytes_transferred", 0) for it in iterations)
    
    def generate_workload(self) -> List[Dict[str, Any]]:
        """
        Generate tasks based on workload configuration.
        
        Returns:
            List of task definitions
        """
        config = self.config.workload
        tasks = []
        
        for i in range(config.num_tasks):
            duration = max(0.1, random.gauss(
                config.task_duration_mean,
                config.task_duration_std
            ))
            
            input_size = max(100, int(random.gauss(
                config.input_size_mean,
                config.input_size_std
            )))
            
            output_size = max(50, int(random.gauss(
                config.output_size_mean,
                config.output_size_std
            )))
            
            task = {
                "task_id": f"task_{i}",
                "workload_type": config.workload_type.value,
                "expected_duration": duration,
                "input_size": input_size,
                "output_size": output_size,
                "enable_checkpointing": config.enable_checkpointing,
                "checkpoint_interval": config.checkpoint_interval,
            }
            
            # Add dependencies if configured
            if config.has_dependencies and i > 0:
                if random.random() < config.dependency_probability:
                    # Depend on a random previous task
                    dep_idx = random.randint(0, i - 1)
                    task["dependencies"] = [f"task_{dep_idx}"]
            
            tasks.append(task)
        
        return tasks
    
    def simulate_task_execution(
        self,
        task: Dict[str, Any],
        worker_config: WorkerSimulationConfig
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Simulate task execution on a worker.
        
        Args:
            task: Task definition
            worker_config: Worker simulation configuration
        
        Returns:
            Tuple of (actual_duration, success, error_message)
        """
        # Adjust duration based on worker speed
        base_duration = task["expected_duration"]
        actual_duration = base_duration / worker_config.processing_speed_factor
        
        # Add some randomness
        actual_duration *= random.uniform(0.8, 1.2)
        
        # Check for failure
        if random.random() < worker_config.failure_probability:
            return actual_duration * random.uniform(0.2, 0.8), False, "Simulated task failure"
        
        return actual_duration, True, None
