"""
Real Metrics Collector for CROWDio Evaluation.

Collects actual metrics from the CROWDio database and running system.
Unlike simulated metrics, this module queries real job/task/worker data.
"""

from __future__ import annotations
import asyncio
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from foreman.db.base import AsyncSessionLocal
from foreman.db.models import JobModel, TaskModel, WorkerModel, WorkerFailureModel


@dataclass
class RealTaskMetrics:
    """Metrics for a single task from actual execution."""
    task_id: str
    job_id: str
    worker_id: Optional[str]
    status: str
    # Timing
    assigned_at: Optional[datetime]
    completed_at: Optional[datetime]
    execution_time_seconds: Optional[float] = None
    queue_time_seconds: Optional[float] = None
    # Checkpointing
    checkpoint_count: int = 0
    progress_percent: float = 0.0
    has_checkpoint: bool = False


@dataclass
class RealJobMetrics:
    """Metrics for a single job from actual execution."""
    job_id: str
    status: str
    total_tasks: int
    completed_tasks: int
    # Timing
    created_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_execution_time_seconds: Optional[float] = None
    # Performance
    throughput_tasks_per_second: float = 0.0
    completion_rate: float = 0.0
    # Task breakdown
    task_metrics: List[RealTaskMetrics] = field(default_factory=list)
    # Computed stats
    avg_task_time: float = 0.0
    min_task_time: float = 0.0
    max_task_time: float = 0.0
    std_task_time: float = 0.0


@dataclass  
class RealWorkerMetrics:
    """Metrics for a single worker from actual data."""
    worker_id: str
    status: str
    # Performance
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    failure_rate: float = 0.0
    # Device info
    device_type: Optional[str] = None
    cpu_cores: Optional[int] = None
    ram_total_mb: Optional[float] = None
    battery_level: Optional[float] = None
    is_charging: Optional[bool] = None
    network_type: Optional[str] = None
    # Timing
    first_connected_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    uptime_seconds: Optional[float] = None


@dataclass
class RealSystemMetrics:
    """System-wide metrics from actual execution."""
    timestamp: datetime
    # Jobs
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    running_jobs: int = 0
    # Tasks
    total_tasks: int = 0
    completed_tasks: int = 0
    pending_tasks: int = 0
    assigned_tasks: int = 0
    failed_tasks: int = 0
    # Workers
    total_workers: int = 0
    online_workers: int = 0
    busy_workers: int = 0
    offline_workers: int = 0
    # Performance
    overall_throughput: float = 0.0
    avg_job_completion_time: float = 0.0
    avg_task_execution_time: float = 0.0
    # Load balancing
    load_distribution: Dict[str, int] = field(default_factory=dict)
    jains_fairness_index: float = 0.0
    coefficient_of_variation: float = 0.0
    # Failures
    total_worker_failures: int = 0
    avg_recovery_time: float = 0.0
    checkpoint_recovery_rate: float = 0.0
    # Job details
    job_metrics: List[RealJobMetrics] = field(default_factory=list)
    worker_metrics: List[RealWorkerMetrics] = field(default_factory=list)


class RealMetricsCollector:
    """
    Collects real metrics from the CROWDio system.
    
    Queries the actual database to gather metrics from completed
    jobs, tasks, workers, and failures.
    """
    
    def __init__(self, output_dir: Path | str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def collect_all_metrics(
        self,
        since: Optional[datetime] = None,
        job_ids: Optional[List[str]] = None
    ) -> RealSystemMetrics:
        """
        Collect all system metrics.
        
        Args:
            since: Only include data since this timestamp
            job_ids: Only include these specific jobs (if provided)
        
        Returns:
            Complete system metrics
        """
        async with AsyncSessionLocal() as session:
            metrics = RealSystemMetrics(timestamp=datetime.now())
            
            # Collect job metrics
            job_metrics = await self._collect_job_metrics(session, since, job_ids)
            metrics.job_metrics = job_metrics
            
            # Collect worker metrics
            worker_metrics = await self._collect_worker_metrics(session)
            metrics.worker_metrics = worker_metrics
            
            # Aggregate job stats
            metrics.total_jobs = len(job_metrics)
            metrics.completed_jobs = sum(1 for j in job_metrics if j.status == "completed")
            metrics.failed_jobs = sum(1 for j in job_metrics if j.status == "failed")
            metrics.running_jobs = sum(1 for j in job_metrics if j.status == "running")
            
            # Aggregate task stats
            all_tasks = [t for j in job_metrics for t in j.task_metrics]
            metrics.total_tasks = len(all_tasks)
            metrics.completed_tasks = sum(1 for t in all_tasks if t.status == "completed")
            metrics.pending_tasks = sum(1 for t in all_tasks if t.status == "pending")
            metrics.assigned_tasks = sum(1 for t in all_tasks if t.status == "assigned")
            metrics.failed_tasks = sum(1 for t in all_tasks if t.status == "failed")
            
            # Aggregate worker stats
            metrics.total_workers = len(worker_metrics)
            metrics.online_workers = sum(1 for w in worker_metrics if w.status == "online")
            metrics.busy_workers = sum(1 for w in worker_metrics if w.status == "busy")
            metrics.offline_workers = sum(1 for w in worker_metrics if w.status == "offline")
            
            # Compute performance metrics
            completed_jobs = [j for j in job_metrics if j.status == "completed" and j.total_execution_time_seconds]
            if completed_jobs:
                metrics.avg_job_completion_time = statistics.mean(
                    j.total_execution_time_seconds for j in completed_jobs
                )
                metrics.overall_throughput = sum(j.throughput_tasks_per_second for j in completed_jobs) / len(completed_jobs)
            
            completed_tasks = [t for t in all_tasks if t.execution_time_seconds]
            if completed_tasks:
                metrics.avg_task_execution_time = statistics.mean(
                    t.execution_time_seconds for t in completed_tasks
                )
            
            # Compute load balancing metrics
            metrics.load_distribution = await self._compute_load_distribution(session, job_ids)
            if metrics.load_distribution:
                metrics.jains_fairness_index = self._compute_jains_fairness(
                    list(metrics.load_distribution.values())
                )
                metrics.coefficient_of_variation = self._compute_cv(
                    list(metrics.load_distribution.values())
                )
            
            # Collect failure metrics
            failure_stats = await self._collect_failure_metrics(session, since)
            metrics.total_worker_failures = failure_stats["total_failures"]
            metrics.avg_recovery_time = failure_stats["avg_recovery_time"]
            metrics.checkpoint_recovery_rate = failure_stats["checkpoint_recovery_rate"]
            
            return metrics
    
    async def _collect_job_metrics(
        self,
        session: AsyncSession,
        since: Optional[datetime],
        job_ids: Optional[List[str]]
    ) -> List[RealJobMetrics]:
        """Collect metrics for all jobs."""
        # Build query
        query = select(JobModel)
        if since:
            query = query.where(JobModel.created_at >= since)
        if job_ids:
            query = query.where(JobModel.id.in_(job_ids))
        query = query.order_by(JobModel.created_at.desc())
        
        result = await session.execute(query)
        jobs = result.scalars().all()
        
        job_metrics = []
        for job in jobs:
            jm = RealJobMetrics(
                job_id=job.id,
                status=job.status,
                total_tasks=job.total_tasks,
                completed_tasks=job.completed_tasks,
                created_at=job.created_at,
                completed_at=job.completed_at,
            )
            
            # Calculate execution time
            if job.created_at and job.completed_at:
                jm.total_execution_time_seconds = (
                    job.completed_at - job.created_at
                ).total_seconds()
                if jm.total_execution_time_seconds > 0:
                    jm.throughput_tasks_per_second = (
                        job.completed_tasks / jm.total_execution_time_seconds
                    )
            
            jm.completion_rate = job.completed_tasks / job.total_tasks if job.total_tasks > 0 else 0
            
            # Collect task metrics for this job
            jm.task_metrics = await self._collect_task_metrics(session, job.id)
            
            # Compute task timing stats
            task_times = [t.execution_time_seconds for t in jm.task_metrics if t.execution_time_seconds]
            if task_times:
                jm.avg_task_time = statistics.mean(task_times)
                jm.min_task_time = min(task_times)
                jm.max_task_time = max(task_times)
                if len(task_times) > 1:
                    jm.std_task_time = statistics.stdev(task_times)
            
            job_metrics.append(jm)
        
        return job_metrics
    
    async def _collect_task_metrics(
        self,
        session: AsyncSession,
        job_id: str
    ) -> List[RealTaskMetrics]:
        """Collect metrics for all tasks in a job."""
        result = await session.execute(
            select(TaskModel).where(TaskModel.job_id == job_id)
        )
        tasks = result.scalars().all()
        
        task_metrics = []
        for task in tasks:
            tm = RealTaskMetrics(
                task_id=task.id,
                job_id=task.job_id,
                worker_id=task.worker_id,
                status=task.status,
                assigned_at=task.assigned_at,
                completed_at=task.completed_at,
                checkpoint_count=task.checkpoint_count or 0,
                progress_percent=task.progress_percent or 0.0,
                has_checkpoint=bool(task.base_checkpoint_data or task.delta_checkpoints),
            )
            
            # Calculate execution time
            if task.assigned_at and task.completed_at:
                tm.execution_time_seconds = (
                    task.completed_at - task.assigned_at
                ).total_seconds()
            
            task_metrics.append(tm)
        
        return task_metrics
    
    async def _collect_worker_metrics(
        self,
        session: AsyncSession
    ) -> List[RealWorkerMetrics]:
        """Collect metrics for all workers."""
        result = await session.execute(select(WorkerModel))
        workers = result.scalars().all()
        
        worker_metrics = []
        for worker in workers:
            total_tasks = worker.total_tasks_completed + worker.total_tasks_failed
            failure_rate = worker.total_tasks_failed / total_tasks if total_tasks > 0 else 0
            
            wm = RealWorkerMetrics(
                worker_id=worker.id,
                status=worker.status,
                total_tasks_completed=worker.total_tasks_completed,
                total_tasks_failed=worker.total_tasks_failed,
                failure_rate=failure_rate,
                device_type=worker.device_type,
                cpu_cores=worker.cpu_cores,
                ram_total_mb=worker.ram_total_mb,
                battery_level=worker.battery_level,
                is_charging=worker.is_charging,
                network_type=worker.network_type,
                first_connected_at=worker.first_connected_at,
                last_seen=worker.last_seen,
            )
            
            # Calculate uptime
            if worker.first_connected_at and worker.last_seen:
                wm.uptime_seconds = (
                    worker.last_seen - worker.first_connected_at
                ).total_seconds()
            
            worker_metrics.append(wm)
        
        return worker_metrics
    
    async def _compute_load_distribution(
        self,
        session: AsyncSession,
        job_ids: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Compute task distribution across workers."""
        query = select(
            TaskModel.worker_id, 
            func.count(TaskModel.id).label("task_count")
        ).where(
            TaskModel.worker_id.isnot(None)
        ).group_by(TaskModel.worker_id)
        
        if job_ids:
            query = query.where(TaskModel.job_id.in_(job_ids))
        
        result = await session.execute(query)
        rows = result.all()
        
        return {row[0]: row[1] for row in rows}
    
    async def _collect_failure_metrics(
        self,
        session: AsyncSession,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Collect failure and recovery metrics."""
        query = select(WorkerFailureModel)
        if since:
            query = query.where(WorkerFailureModel.failed_at >= since)
        
        result = await session.execute(query)
        failures = result.scalars().all()
        
        total_failures = len(failures)
        checkpoint_available = sum(1 for f in failures if f.checkpoint_available)
        checkpoint_recovery_rate = checkpoint_available / total_failures if total_failures > 0 else 0
        
        # Estimate recovery time (time between failure and next successful task for that worker)
        # This is an approximation - actual recovery tracking would need more data
        avg_recovery_time = 0.0  # Would need more sophisticated tracking
        
        return {
            "total_failures": total_failures,
            "avg_recovery_time": avg_recovery_time,
            "checkpoint_recovery_rate": checkpoint_recovery_rate,
        }
    
    def _compute_jains_fairness(self, values: List[int]) -> float:
        """Compute Jain's fairness index."""
        if not values or sum(values) == 0:
            return 0.0
        n = len(values)
        sum_x = sum(values)
        sum_x_squared = sum(x * x for x in values)
        return (sum_x ** 2) / (n * sum_x_squared) if sum_x_squared > 0 else 0.0
    
    def _compute_cv(self, values: List[int]) -> float:
        """Compute coefficient of variation."""
        if not values or len(values) < 2:
            return 0.0
        mean = statistics.mean(values)
        if mean == 0:
            return 0.0
        return statistics.stdev(values) / mean
    
    def export_to_json(self, metrics: RealSystemMetrics, filename: str = "real_metrics.json") -> Path:
        """Export metrics to JSON file."""
        filepath = self.output_dir / filename
        
        # Convert to dict for JSON serialization
        data = self._metrics_to_dict(metrics)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def _metrics_to_dict(self, metrics: RealSystemMetrics) -> Dict[str, Any]:
        """Convert metrics dataclass to dict."""
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "summary": {
                "total_jobs": metrics.total_jobs,
                "completed_jobs": metrics.completed_jobs,
                "failed_jobs": metrics.failed_jobs,
                "running_jobs": metrics.running_jobs,
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "pending_tasks": metrics.pending_tasks,
                "assigned_tasks": metrics.assigned_tasks,
                "failed_tasks": metrics.failed_tasks,
                "total_workers": metrics.total_workers,
                "online_workers": metrics.online_workers,
                "busy_workers": metrics.busy_workers,
                "offline_workers": metrics.offline_workers,
            },
            "performance": {
                "overall_throughput_tasks_per_sec": metrics.overall_throughput,
                "avg_job_completion_time_sec": metrics.avg_job_completion_time,
                "avg_task_execution_time_sec": metrics.avg_task_execution_time,
            },
            "load_balancing": {
                "distribution": metrics.load_distribution,
                "jains_fairness_index": metrics.jains_fairness_index,
                "coefficient_of_variation": metrics.coefficient_of_variation,
            },
            "reliability": {
                "total_worker_failures": metrics.total_worker_failures,
                "avg_recovery_time_sec": metrics.avg_recovery_time,
                "checkpoint_recovery_rate": metrics.checkpoint_recovery_rate,
            },
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "total_tasks": j.total_tasks,
                    "completed_tasks": j.completed_tasks,
                    "completion_rate": j.completion_rate,
                    "total_execution_time_sec": j.total_execution_time_seconds,
                    "throughput_tasks_per_sec": j.throughput_tasks_per_second,
                    "avg_task_time_sec": j.avg_task_time,
                    "min_task_time_sec": j.min_task_time,
                    "max_task_time_sec": j.max_task_time,
                    "std_task_time_sec": j.std_task_time,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                }
                for j in metrics.job_metrics
            ],
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "status": w.status,
                    "device_type": w.device_type,
                    "cpu_cores": w.cpu_cores,
                    "ram_total_mb": w.ram_total_mb,
                    "battery_level": w.battery_level,
                    "is_charging": w.is_charging,
                    "network_type": w.network_type,
                    "total_tasks_completed": w.total_tasks_completed,
                    "total_tasks_failed": w.total_tasks_failed,
                    "failure_rate": w.failure_rate,
                    "uptime_seconds": w.uptime_seconds,
                }
                for w in metrics.worker_metrics
            ],
        }


async def collect_real_evaluation(
    output_dir: str = "evaluation_results",
    since_hours: Optional[int] = None,
    job_ids: Optional[List[str]] = None
) -> RealSystemMetrics:
    """
    Convenience function to collect real metrics.
    
    Args:
        output_dir: Directory to save results
        since_hours: Only include data from the last N hours
        job_ids: Only include specific jobs
    
    Returns:
        Complete system metrics
    """
    collector = RealMetricsCollector(output_dir)
    
    since = None
    if since_hours:
        since = datetime.now() - timedelta(hours=since_hours)
    
    metrics = await collector.collect_all_metrics(since=since, job_ids=job_ids)
    
    # Export to JSON
    filepath = collector.export_to_json(metrics)
    print(f"Metrics exported to: {filepath}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect real CROWDio metrics")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--since-hours", type=int, help="Only include data from last N hours")
    parser.add_argument("--job-ids", nargs="+", help="Specific job IDs to analyze")
    args = parser.parse_args()
    
    metrics = asyncio.run(collect_real_evaluation(
        output_dir=args.output,
        since_hours=args.since_hours,
        job_ids=args.job_ids
    ))
    
    print("\n" + "=" * 60)
    print("REAL METRICS COLLECTED")
    print("=" * 60)
    print(f"\nJobs: {metrics.total_jobs} total, {metrics.completed_jobs} completed")
    print(f"Tasks: {metrics.total_tasks} total, {metrics.completed_tasks} completed")
    print(f"Workers: {metrics.total_workers} total, {metrics.online_workers} online")
    print(f"\nPerformance:")
    print(f"  - Throughput: {metrics.overall_throughput:.2f} tasks/sec")
    print(f"  - Avg job time: {metrics.avg_job_completion_time:.2f}s")
    print(f"  - Avg task time: {metrics.avg_task_execution_time:.2f}s")
    print(f"\nLoad Balancing:")
    print(f"  - Jain's Fairness: {metrics.jains_fairness_index:.3f}")
    print(f"  - CoV: {metrics.coefficient_of_variation:.3f}")
    print(f"\nReliability:")
    print(f"  - Total failures: {metrics.total_worker_failures}")
    print(f"  - Checkpoint recovery: {metrics.checkpoint_recovery_rate:.1%}")
