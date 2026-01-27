"""
Load Balancing Analyzer Module

Analyzes and measures the efficiency of load distribution across workers
in the CROWDio distributed computing system.

Metrics:
- Load distribution uniformity (coefficient of variation)
- Worker utilization rates
- Task queue imbalance
- Makespan vs optimal makespan ratio
- Jain's fairness index
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
import math


class SchedulerType(Enum):
    """Types of scheduling algorithms"""
    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY = "priority"
    PERFORMANCE_BASED = "performance_based"
    ENERGY_AWARE = "energy_aware"


@dataclass
class WorkerLoad:
    """Load information for a single worker"""
    worker_id: str
    
    # Task counts
    tasks_assigned: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_in_progress: int = 0
    
    # Time-based metrics
    total_execution_time: float = 0.0  # Total time spent executing tasks
    total_idle_time: float = 0.0  # Total idle time
    avg_task_time: float = 0.0
    
    # Utilization
    utilization_rate: float = 0.0  # 0-100%
    
    # Queue metrics
    current_queue_size: int = 0
    max_queue_size: int = 0
    
    # Performance
    throughput: float = 0.0  # Tasks per second
    
    # Tracking
    first_task_time: Optional[float] = None
    last_task_time: Optional[float] = None
    
    def calculate_utilization(self, total_tracking_time: float) -> None:
        """Calculate utilization rate"""
        if total_tracking_time > 0:
            self.utilization_rate = (self.total_execution_time / total_tracking_time) * 100.0
            self.throughput = self.tasks_completed / total_tracking_time if total_tracking_time > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoadDistribution:
    """Snapshot of load distribution across all workers"""
    timestamp: float = field(default_factory=time.time)
    
    # Worker loads
    worker_loads: Dict[str, WorkerLoad] = field(default_factory=dict)
    
    # Distribution metrics
    total_workers: int = 0
    active_workers: int = 0  # Workers with tasks
    
    # Load balance metrics
    coefficient_of_variation: float = 0.0  # CV of task distribution (lower = better)
    jains_fairness_index: float = 0.0  # 0-1, 1 = perfectly fair
    load_imbalance_ratio: float = 0.0  # max/avg ratio
    
    # Makespan analysis
    actual_makespan: float = 0.0  # Time from first to last task completion
    optimal_makespan: float = 0.0  # Theoretical best makespan
    makespan_ratio: float = 0.0  # actual/optimal (1 = optimal)
    
    # Aggregate stats
    avg_tasks_per_worker: float = 0.0
    std_tasks_per_worker: float = 0.0
    min_tasks_per_worker: int = 0
    max_tasks_per_worker: int = 0
    
    def calculate_metrics(self) -> None:
        """Calculate all distribution metrics"""
        if not self.worker_loads:
            return
        
        task_counts = [w.tasks_completed for w in self.worker_loads.values()]
        
        if not task_counts:
            return
        
        self.total_workers = len(self.worker_loads)
        self.active_workers = sum(1 for c in task_counts if c > 0)
        
        # Basic stats
        self.avg_tasks_per_worker = statistics.mean(task_counts)
        self.min_tasks_per_worker = min(task_counts)
        self.max_tasks_per_worker = max(task_counts)
        
        if len(task_counts) > 1:
            self.std_tasks_per_worker = statistics.stdev(task_counts)
        
        # Coefficient of Variation (lower = more balanced)
        if self.avg_tasks_per_worker > 0:
            self.coefficient_of_variation = self.std_tasks_per_worker / self.avg_tasks_per_worker
        
        # Load imbalance ratio
        if self.avg_tasks_per_worker > 0:
            self.load_imbalance_ratio = self.max_tasks_per_worker / self.avg_tasks_per_worker
        
        # Jain's fairness index: (sum(x))^2 / (n * sum(x^2))
        # Value of 1 means perfect fairness
        if task_counts and sum(task_counts) > 0:
            sum_x = sum(task_counts)
            sum_x_squared = sum(x ** 2 for x in task_counts)
            n = len(task_counts)
            if sum_x_squared > 0:
                self.jains_fairness_index = (sum_x ** 2) / (n * sum_x_squared)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_workers": self.total_workers,
            "active_workers": self.active_workers,
            "coefficient_of_variation": self.coefficient_of_variation,
            "jains_fairness_index": self.jains_fairness_index,
            "load_imbalance_ratio": self.load_imbalance_ratio,
            "makespan_ratio": self.makespan_ratio,
            "avg_tasks_per_worker": self.avg_tasks_per_worker,
            "std_tasks_per_worker": self.std_tasks_per_worker,
            "min_tasks_per_worker": self.min_tasks_per_worker,
            "max_tasks_per_worker": self.max_tasks_per_worker,
            "worker_loads": {k: v.to_dict() for k, v in self.worker_loads.items()},
        }


class LoadBalancingAnalyzer:
    """
    Analyzes load balancing efficiency across the distributed system.
    
    Provides:
    - Real-time load distribution monitoring
    - Historical load balance analysis
    - Scheduler efficiency comparison
    - Recommendations for load rebalancing
    """
    
    def __init__(self, collection_interval: float = 1.0):
        """
        Initialize load balancing analyzer.
        
        Args:
            collection_interval: Interval for periodic snapshots (seconds)
        """
        self.collection_interval = collection_interval
        
        # Worker load tracking: worker_id -> WorkerLoad
        self._worker_loads: Dict[str, WorkerLoad] = {}
        
        # Historical snapshots
        self._distribution_history: List[LoadDistribution] = []
        
        # Tracking state
        self._tracking_start_time: Optional[float] = None
        self._lock = asyncio.Lock()
        
        # Collection task
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Task timing tracking: task_id -> (start_time, worker_id)
        self._task_timings: Dict[str, Tuple[float, str]] = {}
    
    # ==================== Worker Management ====================
    
    async def register_worker(self, worker_id: str) -> WorkerLoad:
        """Register a new worker for load tracking"""
        async with self._lock:
            if worker_id not in self._worker_loads:
                self._worker_loads[worker_id] = WorkerLoad(worker_id=worker_id)
            return self._worker_loads[worker_id]
    
    async def unregister_worker(self, worker_id: str) -> Optional[WorkerLoad]:
        """Unregister a worker"""
        async with self._lock:
            return self._worker_loads.pop(worker_id, None)
    
    # ==================== Event Recording ====================
    
    async def record_task_assigned(
        self,
        task_id: str,
        worker_id: str,
        queue_size: int = 0
    ) -> None:
        """Record a task assignment to a worker"""
        async with self._lock:
            if worker_id not in self._worker_loads:
                self._worker_loads[worker_id] = WorkerLoad(worker_id=worker_id)
            
            load = self._worker_loads[worker_id]
            load.tasks_assigned += 1
            load.tasks_in_progress += 1
            load.current_queue_size = queue_size
            load.max_queue_size = max(load.max_queue_size, queue_size)
            
            self._task_timings[task_id] = (time.time(), worker_id)
    
    async def record_task_started(
        self,
        task_id: str,
        worker_id: str
    ) -> None:
        """Record that a task started execution"""
        async with self._lock:
            now = time.time()
            
            if worker_id not in self._worker_loads:
                self._worker_loads[worker_id] = WorkerLoad(worker_id=worker_id)
            
            load = self._worker_loads[worker_id]
            
            if load.first_task_time is None:
                load.first_task_time = now
            
            # Update task timing to execution start
            if task_id in self._task_timings:
                _, _ = self._task_timings[task_id]
                self._task_timings[task_id] = (now, worker_id)
    
    async def record_task_completed(
        self,
        task_id: str,
        worker_id: str,
        execution_time: Optional[float] = None
    ) -> None:
        """Record task completion"""
        async with self._lock:
            now = time.time()
            
            if worker_id not in self._worker_loads:
                self._worker_loads[worker_id] = WorkerLoad(worker_id=worker_id)
            
            load = self._worker_loads[worker_id]
            load.tasks_completed += 1
            load.tasks_in_progress = max(0, load.tasks_in_progress - 1)
            load.last_task_time = now
            
            # Calculate execution time
            if execution_time is None and task_id in self._task_timings:
                start_time, _ = self._task_timings.pop(task_id)
                execution_time = now - start_time
            
            if execution_time:
                load.total_execution_time += execution_time
                # Update running average
                n = load.tasks_completed
                load.avg_task_time = (
                    (load.avg_task_time * (n - 1) + execution_time) / n
                )
    
    async def record_task_failed(
        self,
        task_id: str,
        worker_id: str
    ) -> None:
        """Record task failure"""
        async with self._lock:
            if worker_id not in self._worker_loads:
                self._worker_loads[worker_id] = WorkerLoad(worker_id=worker_id)
            
            load = self._worker_loads[worker_id]
            load.tasks_failed += 1
            load.tasks_in_progress = max(0, load.tasks_in_progress - 1)
            
            if task_id in self._task_timings:
                del self._task_timings[task_id]
    
    # ==================== Load Distribution Analysis ====================
    
    async def get_current_distribution(self) -> LoadDistribution:
        """Get current load distribution snapshot"""
        async with self._lock:
            tracking_time = time.time() - (self._tracking_start_time or time.time())
            
            # Update utilization for all workers
            for load in self._worker_loads.values():
                load.calculate_utilization(tracking_time)
            
            distribution = LoadDistribution(
                timestamp=time.time(),
                worker_loads=dict(self._worker_loads)
            )
            distribution.calculate_metrics()
            
            # Calculate makespan metrics
            if self._worker_loads:
                first_task = min(
                    (w.first_task_time for w in self._worker_loads.values() if w.first_task_time),
                    default=None
                )
                last_task = max(
                    (w.last_task_time for w in self._worker_loads.values() if w.last_task_time),
                    default=None
                )
                
                if first_task and last_task:
                    distribution.actual_makespan = last_task - first_task
                    
                    # Calculate optimal makespan (if all workers perfectly balanced)
                    total_work = sum(w.total_execution_time for w in self._worker_loads.values())
                    num_workers = len(self._worker_loads)
                    if num_workers > 0:
                        distribution.optimal_makespan = total_work / num_workers
                        if distribution.optimal_makespan > 0:
                            distribution.makespan_ratio = (
                                distribution.actual_makespan / distribution.optimal_makespan
                            )
            
            return distribution
    
    async def get_worker_load(self, worker_id: str) -> Optional[WorkerLoad]:
        """Get load information for a specific worker"""
        async with self._lock:
            return self._worker_loads.get(worker_id)
    
    async def get_least_loaded_workers(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get the n least loaded workers by task count"""
        async with self._lock:
            workers = [
                (w.worker_id, w.tasks_in_progress + w.current_queue_size)
                for w in self._worker_loads.values()
            ]
            workers.sort(key=lambda x: x[1])
            return workers[:n]
    
    async def get_most_loaded_workers(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get the n most loaded workers by task count"""
        async with self._lock:
            workers = [
                (w.worker_id, w.tasks_in_progress + w.current_queue_size)
                for w in self._worker_loads.values()
            ]
            workers.sort(key=lambda x: x[1], reverse=True)
            return workers[:n]
    
    # ==================== Periodic Collection ====================
    
    async def start_tracking(self) -> None:
        """Start load tracking and periodic collection"""
        if self._running:
            return
        
        self._running = True
        self._tracking_start_time = time.time()
        self._collection_task = asyncio.create_task(self._collection_loop())
        print("LoadBalancingAnalyzer: Started tracking")
    
    async def stop_tracking(self) -> None:
        """Stop load tracking"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
        print("LoadBalancingAnalyzer: Stopped tracking")
    
    async def _collection_loop(self) -> None:
        """Periodic collection loop"""
        while self._running:
            try:
                distribution = await self.get_current_distribution()
                async with self._lock:
                    self._distribution_history.append(distribution)
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"LoadBalancingAnalyzer: Error in collection loop: {e}")
    
    # ==================== Analysis and Reporting ====================
    
    async def analyze_scheduler_efficiency(self) -> Dict[str, Any]:
        """Analyze overall scheduler efficiency"""
        distribution = await self.get_current_distribution()
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_workers": distribution.total_workers,
            "active_workers": distribution.active_workers,
            "efficiency_metrics": {
                "jains_fairness_index": distribution.jains_fairness_index,
                "coefficient_of_variation": distribution.coefficient_of_variation,
                "load_imbalance_ratio": distribution.load_imbalance_ratio,
                "makespan_ratio": distribution.makespan_ratio,
            },
            "task_distribution": {
                "avg_tasks_per_worker": distribution.avg_tasks_per_worker,
                "std_tasks_per_worker": distribution.std_tasks_per_worker,
                "min_tasks_per_worker": distribution.min_tasks_per_worker,
                "max_tasks_per_worker": distribution.max_tasks_per_worker,
            },
            "interpretation": self._interpret_metrics(distribution),
        }
        
        return analysis
    
    def _interpret_metrics(self, distribution: LoadDistribution) -> Dict[str, str]:
        """Provide human-readable interpretation of metrics"""
        interpretations = {}
        
        # Jain's fairness index interpretation
        jfi = distribution.jains_fairness_index
        if jfi >= 0.95:
            interpretations["fairness"] = "Excellent - Nearly perfect load distribution"
        elif jfi >= 0.85:
            interpretations["fairness"] = "Good - Well balanced load distribution"
        elif jfi >= 0.70:
            interpretations["fairness"] = "Fair - Some imbalance exists"
        else:
            interpretations["fairness"] = "Poor - Significant load imbalance"
        
        # CV interpretation
        cv = distribution.coefficient_of_variation
        if cv <= 0.1:
            interpretations["variation"] = "Excellent - Very consistent load"
        elif cv <= 0.25:
            interpretations["variation"] = "Good - Low variation"
        elif cv <= 0.5:
            interpretations["variation"] = "Fair - Moderate variation"
        else:
            interpretations["variation"] = "Poor - High variation in load"
        
        # Makespan interpretation
        mr = distribution.makespan_ratio
        if mr <= 1.1:
            interpretations["makespan"] = "Optimal - Near-perfect scheduling"
        elif mr <= 1.5:
            interpretations["makespan"] = "Good - Reasonable overhead"
        elif mr <= 2.0:
            interpretations["makespan"] = "Fair - Room for improvement"
        else:
            interpretations["makespan"] = "Poor - Significant scheduling overhead"
        
        return interpretations
    
    async def get_rebalancing_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for load rebalancing"""
        distribution = await self.get_current_distribution()
        recommendations = []
        
        if distribution.total_workers < 2:
            return recommendations
        
        avg_load = distribution.avg_tasks_per_worker
        
        for worker_id, load in distribution.worker_loads.items():
            if load.tasks_in_progress > avg_load * 1.5:
                recommendations.append({
                    "type": "reduce_load",
                    "worker_id": worker_id,
                    "current_load": load.tasks_in_progress,
                    "target_load": int(avg_load),
                    "action": f"Consider migrating {load.tasks_in_progress - int(avg_load)} tasks"
                })
            elif load.tasks_in_progress < avg_load * 0.5 and avg_load > 0:
                recommendations.append({
                    "type": "increase_load",
                    "worker_id": worker_id,
                    "current_load": load.tasks_in_progress,
                    "target_load": int(avg_load),
                    "action": f"Can accept {int(avg_load) - load.tasks_in_progress} more tasks"
                })
        
        return recommendations
    
    async def compare_schedulers(
        self,
        scheduler_results: Dict[str, LoadDistribution]
    ) -> Dict[str, Any]:
        """
        Compare efficiency of different scheduling algorithms.
        
        Args:
            scheduler_results: Dict mapping scheduler name to its LoadDistribution
        
        Returns:
            Comparison analysis
        """
        comparison = {
            "schedulers": {},
            "ranking": {
                "fairness": [],
                "variation": [],
                "makespan": [],
            }
        }
        
        for name, dist in scheduler_results.items():
            comparison["schedulers"][name] = {
                "jains_fairness_index": dist.jains_fairness_index,
                "coefficient_of_variation": dist.coefficient_of_variation,
                "makespan_ratio": dist.makespan_ratio,
                "avg_tasks_per_worker": dist.avg_tasks_per_worker,
            }
        
        # Rank by each metric
        by_fairness = sorted(
            scheduler_results.items(),
            key=lambda x: x[1].jains_fairness_index,
            reverse=True
        )
        comparison["ranking"]["fairness"] = [name for name, _ in by_fairness]
        
        by_variation = sorted(
            scheduler_results.items(),
            key=lambda x: x[1].coefficient_of_variation
        )
        comparison["ranking"]["variation"] = [name for name, _ in by_variation]
        
        by_makespan = sorted(
            scheduler_results.items(),
            key=lambda x: x[1].makespan_ratio
        )
        comparison["ranking"]["makespan"] = [name for name, _ in by_makespan]
        
        return comparison
    
    # ==================== Export ====================
    
    async def export_analysis(self, filepath: str) -> None:
        """Export load balancing analysis to file"""
        analysis = await self.analyze_scheduler_efficiency()
        
        async with self._lock:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "current_analysis": analysis,
                "distribution_history": [d.to_dict() for d in self._distribution_history],
                "recommendations": await self.get_rebalancing_recommendations(),
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"LoadBalancingAnalyzer: Exported analysis to {filepath}")
    
    async def reset(self) -> None:
        """Reset all tracking data"""
        async with self._lock:
            self._worker_loads.clear()
            self._distribution_history.clear()
            self._task_timings.clear()
            self._tracking_start_time = None
        print("LoadBalancingAnalyzer: Reset all tracking data")
