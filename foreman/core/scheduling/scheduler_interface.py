"""
Task scheduling interface and implementations
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set
from dataclasses import dataclass


@dataclass
class Task:
    """Task information for scheduling"""

    id: str
    job_id: str
    args: str
    priority: int = 0
    retry_count: int = 0
    stage_func_code: Optional[str] = None  # Per-stage function code for pipeline tasks


@dataclass
class Worker:
    """Worker information for scheduling - EXTENDED for MCDM"""

    # Basic info (existing)
    id: str
    status: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task_id: Optional[str] = None

    # Device specifications (NEW for MCDM)
    cpu_cores: int = 1
    cpu_threads: int = 1
    cpu_frequency_mhz: float = 1000.0
    cpu_usage_percent: float = 0.0
    cpu_model: Optional[str] = None
    ram_total_mb: float = 1024.0
    ram_available_mb: float = 512.0

    # Battery/Power (NEW for MCDM)
    battery_level: float = 100.0  # 0-100
    is_charging: bool = True

    # Network (NEW for MCDM)
    network_type: str = "WiFi"  # WiFi, Cellular, Ethernet
    network_speed_mbps: float = 10.0

    # Performance (NEW for MCDM)
    avg_task_duration_sec: float = 0.0
    success_rate: float = 1.0  # Moved from property to field for MCDM

    # GPU (NEW for MCDM)
    gpu_available: bool = False
    gpu_model: Optional[str] = None

    # Storage (NEW for MCDM)
    storage_available_gb: float = 0.0

    # Device info
    device_type: Optional[str] = None
    os_type: Optional[str] = None
    os_version: Optional[str] = None
    runtime: Optional[str] = None
    model_runtime: Optional[str] = None

    # Computed properties
    @property
    def ram_usage_percent(self) -> float:
        """Calculate RAM usage percentage"""
        if self.ram_total_mb > 0:
            used = self.ram_total_mb - self.ram_available_mb
            return (used / self.ram_total_mb) * 100.0
        return 0.0

    @property
    def is_low_battery(self) -> bool:
        """Check if device has low battery"""
        return self.battery_level < 20.0 and not self.is_charging

    @property
    def is_high_performance(self) -> bool:
        """Check if device is high-performance"""
        return (
            self.cpu_cores >= 4
            and self.ram_total_mb >= 4096
            and self.cpu_frequency_mhz >= 2000
        )


class TaskScheduler(ABC):
    """Abstract base class for task scheduling algorithms"""

    @abstractmethod
    async def select_worker(
        self, task: Task, available_workers: Set[str], all_workers: dict
    ) -> Optional[str]:
        """
        Select the best worker for a given task

        Args:
            task: Task to be assigned
            available_workers: Set of available worker IDs
            all_workers: Dictionary of all workers (id -> Worker)

        Returns:
            Selected worker ID or None if no suitable worker
        """
        pass

    @abstractmethod
    async def select_task(
        self, pending_tasks: List[Task], worker_id: str
    ) -> Optional[Task]:
        """
        Select the best task for a given worker

        Args:
            pending_tasks: List of pending tasks
            worker_id: Worker ID requesting a task

        Returns:
            Selected task or None if no suitable task
        """
        pass

    async def batch_select_workers(
        self, tasks: List[Task], available_workers: Set[str], all_workers: dict
    ) -> List[tuple]:
        """
        Optimized batch assignment: select workers for multiple tasks at once.

        This method ranks workers ONCE and assigns them to tasks, avoiding
        redundant MCDM computations. Subclasses can override for custom logic.

        Optimization strategy:
        - If tasks >= available_workers: Skip ranking, assign all workers (no choice needed)
        - If tasks < available_workers: Rank once, pick top N workers

        Args:
            tasks: List of tasks to assign
            available_workers: Set of available worker IDs
            all_workers: Dictionary of all workers (id -> Worker)

        Returns:
            List of (task, worker_id) tuples for successful assignments
        """
        # Default implementation: fall back to individual select_worker calls
        # Subclasses (like BaseMCDMScheduler) should override for optimization
        assignments = []
        remaining_workers = set(available_workers)

        for task in tasks:
            if not remaining_workers:
                break
            worker_id = await self.select_worker(task, remaining_workers, all_workers)
            if worker_id:
                assignments.append((task, worker_id))
                remaining_workers.discard(worker_id)

        return assignments
