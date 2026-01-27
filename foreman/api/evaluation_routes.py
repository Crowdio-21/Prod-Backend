"""
Evaluation API Routes for CROWDio Foreman.

Provides REST API endpoints for evaluation metrics that can be displayed
on the dashboard.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import statistics

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from foreman.db.base import get_db
from foreman.db.models import JobModel, TaskModel, WorkerModel, WorkerFailureModel


# ==================== Response Models ====================

class TaskMetricsResponse(BaseModel):
    """Metrics for a single task."""
    task_id: str
    job_id: str
    worker_id: Optional[str]
    status: str
    execution_time_seconds: Optional[float]
    has_checkpoint: bool


class JobMetricsResponse(BaseModel):
    """Metrics for a single job."""
    job_id: str
    status: str
    total_tasks: int
    completed_tasks: int
    completion_rate: float
    total_execution_time_seconds: Optional[float]
    throughput_tasks_per_sec: float
    avg_task_time_seconds: float
    min_task_time_seconds: float
    max_task_time_seconds: float
    created_at: Optional[datetime]
    completed_at: Optional[datetime]


class WorkerMetricsResponse(BaseModel):
    """Metrics for a single worker."""
    worker_id: str
    status: str
    device_type: Optional[str]
    cpu_cores: Optional[int]
    ram_total_mb: Optional[float]
    battery_level: Optional[float]
    is_charging: Optional[bool]
    total_tasks_completed: int
    total_tasks_failed: int
    failure_rate: float


class LoadBalancingMetricsResponse(BaseModel):
    """Load balancing metrics."""
    distribution: dict  # worker_id -> task_count
    jains_fairness_index: float
    coefficient_of_variation: float
    most_loaded_worker: Optional[str]
    least_loaded_worker: Optional[str]
    load_imbalance_ratio: float


class PerformanceMetricsResponse(BaseModel):
    """Overall performance metrics."""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_tasks: int
    completed_tasks: int
    overall_throughput: float
    avg_job_completion_time: float
    avg_task_execution_time: float


class EvaluationResponse(BaseModel):
    """Complete evaluation response."""
    timestamp: datetime
    performance: PerformanceMetricsResponse
    load_balancing: LoadBalancingMetricsResponse
    jobs: List[JobMetricsResponse]
    workers: List[WorkerMetricsResponse]
    # Charts data (for frontend rendering)
    charts: dict


# ==================== Router ====================

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/metrics", response_model=EvaluationResponse)
async def get_evaluation_metrics(
    db: AsyncSession = Depends(get_db),
    job_ids: Optional[str] = Query(None, description="Comma-separated job IDs to filter"),
    since_hours: Optional[int] = Query(None, description="Only include data from last N hours")
):
    """
    Get comprehensive evaluation metrics for the CROWDio system.
    
    Returns performance, load balancing, and per-job/worker metrics,
    along with chart data for frontend visualization.
    """
    # Parse job_ids filter
    job_id_list = job_ids.split(",") if job_ids else None
    
    # Calculate time filter
    since = None
    if since_hours:
        since = datetime.now() - timedelta(hours=since_hours)
    
    # Collect all metrics
    job_metrics = await _get_job_metrics(db, job_id_list, since)
    worker_metrics = await _get_worker_metrics(db)
    load_balancing = await _get_load_balancing_metrics(db, job_id_list)
    performance = await _get_performance_metrics(db, job_id_list, since)
    
    # Generate chart data
    charts = _generate_chart_data(job_metrics, worker_metrics, load_balancing)
    
    return EvaluationResponse(
        timestamp=datetime.now(),
        performance=performance,
        load_balancing=load_balancing,
        jobs=job_metrics,
        workers=worker_metrics,
        charts=charts
    )


@router.get("/job/{job_id}", response_model=JobMetricsResponse)
async def get_job_evaluation(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get evaluation metrics for a specific job."""
    metrics = await _get_job_metrics(db, [job_id], None)
    if not metrics:
        return JobMetricsResponse(
            job_id=job_id,
            status="not_found",
            total_tasks=0,
            completed_tasks=0,
            completion_rate=0,
            total_execution_time_seconds=None,
            throughput_tasks_per_sec=0,
            avg_task_time_seconds=0,
            min_task_time_seconds=0,
            max_task_time_seconds=0,
            created_at=None,
            completed_at=None
        )
    return metrics[0]


@router.get("/load-distribution")
async def get_load_distribution(
    db: AsyncSession = Depends(get_db),
    job_ids: Optional[str] = Query(None)
):
    """Get load distribution across workers for chart rendering."""
    job_id_list = job_ids.split(",") if job_ids else None
    metrics = await _get_load_balancing_metrics(db, job_id_list)
    
    return {
        "labels": list(metrics.distribution.keys()),
        "data": list(metrics.distribution.values()),
        "fairness_index": metrics.jains_fairness_index,
        "cv": metrics.coefficient_of_variation
    }


@router.get("/throughput-history")
async def get_throughput_history(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(10, description="Number of recent jobs to include")
):
    """Get throughput history for recent jobs."""
    result = await db.execute(
        select(JobModel)
        .where(JobModel.status == "completed")
        .order_by(JobModel.completed_at.desc())
        .limit(limit)
    )
    jobs = result.scalars().all()
    
    labels = []
    throughputs = []
    execution_times = []
    
    for job in reversed(jobs):  # Oldest first
        if job.created_at and job.completed_at:
            exec_time = (job.completed_at - job.created_at).total_seconds()
            throughput = job.completed_tasks / exec_time if exec_time > 0 else 0
            
            labels.append(job.id[:8])
            throughputs.append(round(throughput, 2))
            execution_times.append(round(exec_time, 2))
    
    return {
        "labels": labels,
        "throughputs": throughputs,
        "execution_times": execution_times
    }


@router.get("/worker-performance")
async def get_worker_performance(
    db: AsyncSession = Depends(get_db)
):
    """Get worker performance data for chart rendering."""
    result = await db.execute(select(WorkerModel))
    workers = result.scalars().all()
    
    labels = []
    completed = []
    failed = []
    
    for worker in workers:
        labels.append(worker.id[:8])
        completed.append(worker.total_tasks_completed)
        failed.append(worker.total_tasks_failed)
    
    return {
        "labels": labels,
        "completed": completed,
        "failed": failed
    }


# ==================== Helper Functions ====================

async def _get_job_metrics(
    db: AsyncSession,
    job_ids: Optional[List[str]],
    since: Optional[datetime]
) -> List[JobMetricsResponse]:
    """Get metrics for jobs."""
    query = select(JobModel)
    if job_ids:
        query = query.where(JobModel.id.in_(job_ids))
    if since:
        query = query.where(JobModel.created_at >= since)
    query = query.order_by(JobModel.created_at.desc())
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    metrics = []
    for job in jobs:
        # Get task execution times
        task_result = await db.execute(
            select(TaskModel).where(TaskModel.job_id == job.id)
        )
        tasks = task_result.scalars().all()
        
        task_times = []
        for task in tasks:
            if task.assigned_at and task.completed_at:
                task_times.append((task.completed_at - task.assigned_at).total_seconds())
        
        exec_time = None
        throughput = 0.0
        if job.created_at and job.completed_at:
            exec_time = (job.completed_at - job.created_at).total_seconds()
            throughput = job.completed_tasks / exec_time if exec_time > 0 else 0
        
        metrics.append(JobMetricsResponse(
            job_id=job.id,
            status=job.status,
            total_tasks=job.total_tasks,
            completed_tasks=job.completed_tasks,
            completion_rate=job.completed_tasks / job.total_tasks if job.total_tasks > 0 else 0,
            total_execution_time_seconds=exec_time,
            throughput_tasks_per_sec=throughput,
            avg_task_time_seconds=statistics.mean(task_times) if task_times else 0,
            min_task_time_seconds=min(task_times) if task_times else 0,
            max_task_time_seconds=max(task_times) if task_times else 0,
            created_at=job.created_at,
            completed_at=job.completed_at
        ))
    
    return metrics


async def _get_worker_metrics(db: AsyncSession) -> List[WorkerMetricsResponse]:
    """Get metrics for all workers."""
    result = await db.execute(select(WorkerModel))
    workers = result.scalars().all()
    
    metrics = []
    for worker in workers:
        total = worker.total_tasks_completed + worker.total_tasks_failed
        failure_rate = worker.total_tasks_failed / total if total > 0 else 0
        
        metrics.append(WorkerMetricsResponse(
            worker_id=worker.id,
            status=worker.status,
            device_type=worker.device_type,
            cpu_cores=worker.cpu_cores,
            ram_total_mb=worker.ram_total_mb,
            battery_level=worker.battery_level,
            is_charging=worker.is_charging,
            total_tasks_completed=worker.total_tasks_completed,
            total_tasks_failed=worker.total_tasks_failed,
            failure_rate=failure_rate
        ))
    
    return metrics


async def _get_load_balancing_metrics(
    db: AsyncSession,
    job_ids: Optional[List[str]]
) -> LoadBalancingMetricsResponse:
    """Get load balancing metrics."""
    query = select(
        TaskModel.worker_id,
        func.count(TaskModel.id).label("task_count")
    ).where(
        TaskModel.worker_id.isnot(None)
    ).group_by(TaskModel.worker_id)
    
    if job_ids:
        query = query.where(TaskModel.job_id.in_(job_ids))
    
    result = await db.execute(query)
    rows = result.all()
    
    distribution = {row[0]: row[1] for row in rows}
    values = list(distribution.values())
    
    # Calculate Jain's fairness index
    jfi = 0.0
    cv = 0.0
    if values and sum(values) > 0:
        n = len(values)
        sum_x = sum(values)
        sum_x_squared = sum(x * x for x in values)
        jfi = (sum_x ** 2) / (n * sum_x_squared) if sum_x_squared > 0 else 0
        
        if len(values) > 1:
            mean = statistics.mean(values)
            cv = statistics.stdev(values) / mean if mean > 0 else 0
    
    most_loaded = max(distribution, key=distribution.get) if distribution else None
    least_loaded = min(distribution, key=distribution.get) if distribution else None
    
    imbalance = 0.0
    if distribution and len(distribution) > 1:
        max_load = max(values)
        min_load = min(values)
        imbalance = (max_load - min_load) / max_load if max_load > 0 else 0
    
    return LoadBalancingMetricsResponse(
        distribution=distribution,
        jains_fairness_index=jfi,
        coefficient_of_variation=cv,
        most_loaded_worker=most_loaded,
        least_loaded_worker=least_loaded,
        load_imbalance_ratio=imbalance
    )


async def _get_performance_metrics(
    db: AsyncSession,
    job_ids: Optional[List[str]],
    since: Optional[datetime]
) -> PerformanceMetricsResponse:
    """Get overall performance metrics."""
    # Job counts
    job_query = select(JobModel)
    if job_ids:
        job_query = job_query.where(JobModel.id.in_(job_ids))
    if since:
        job_query = job_query.where(JobModel.created_at >= since)
    
    result = await db.execute(job_query)
    jobs = result.scalars().all()
    
    total_jobs = len(jobs)
    completed_jobs = sum(1 for j in jobs if j.status == "completed")
    failed_jobs = sum(1 for j in jobs if j.status == "failed")
    
    # Task counts
    task_query = select(TaskModel)
    if job_ids:
        task_query = task_query.where(TaskModel.job_id.in_(job_ids))
    
    result = await db.execute(task_query)
    tasks = result.scalars().all()
    
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.status == "completed")
    
    # Calculate averages
    job_times = []
    task_times = []
    throughputs = []
    
    for job in jobs:
        if job.created_at and job.completed_at:
            exec_time = (job.completed_at - job.created_at).total_seconds()
            job_times.append(exec_time)
            if exec_time > 0:
                throughputs.append(job.completed_tasks / exec_time)
    
    for task in tasks:
        if task.assigned_at and task.completed_at:
            task_times.append((task.completed_at - task.assigned_at).total_seconds())
    
    return PerformanceMetricsResponse(
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        overall_throughput=statistics.mean(throughputs) if throughputs else 0,
        avg_job_completion_time=statistics.mean(job_times) if job_times else 0,
        avg_task_execution_time=statistics.mean(task_times) if task_times else 0
    )


def _generate_chart_data(
    job_metrics: List[JobMetricsResponse],
    worker_metrics: List[WorkerMetricsResponse],
    load_balancing: LoadBalancingMetricsResponse
) -> dict:
    """Generate chart data for frontend visualization."""
    return {
        "load_distribution": {
            "labels": list(load_balancing.distribution.keys()),
            "data": list(load_balancing.distribution.values()),
            "type": "bar"
        },
        "job_throughput": {
            "labels": [j.job_id[:8] for j in job_metrics],
            "data": [j.throughput_tasks_per_sec for j in job_metrics],
            "type": "bar"
        },
        "job_execution_time": {
            "labels": [j.job_id[:8] for j in job_metrics],
            "data": [j.total_execution_time_seconds or 0 for j in job_metrics],
            "type": "bar"
        },
        "worker_tasks": {
            "labels": [w.worker_id[:8] for w in worker_metrics],
            "completed": [w.total_tasks_completed for w in worker_metrics],
            "failed": [w.total_tasks_failed for w in worker_metrics],
            "type": "stacked_bar"
        },
        "task_time_distribution": {
            "labels": [j.job_id[:8] for j in job_metrics],
            "avg": [j.avg_task_time_seconds for j in job_metrics],
            "min": [j.min_task_time_seconds for j in job_metrics],
            "max": [j.max_task_time_seconds for j in job_metrics],
            "type": "line"
        }
    }
