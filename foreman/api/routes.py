
from fastapi import  Depends, HTTPException, APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from foreman.db.base import get_db
from foreman.db.crud import (
    get_jobs, get_job, get_workers, get_job_stats,
    get_worker_failures, get_worker_failure_stats, delete_worker, clear_database
)
from foreman.db.models import JobModel, TaskModel, WorkerModel, WorkerFailureModel
from foreman.schema.schema import ( 
    JobResponse, WorkerResponse, JobStats, WorkerFailureResponse, WorkerFailureStats
)


# Create an APIRouter instance
router = APIRouter(
    prefix=""
)

@router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard page with evaluation charts"""
    import os
    # Use the new dashboard with charts
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(dashboard_path):
        # Fallback to temp dashboard if new one doesn't exist
        dashboard_path = os.path.join(os.path.dirname(__file__), "temp_dashboard.html")
    with open(dashboard_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


# REST API endpoints
@router.get("/api/stats", response_model=JobStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get job statistics"""
    return await get_job_stats(db)


@router.get("/api/jobs", response_model=list[JobResponse])
async def list_jobs(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """List all jobs"""
    jobs = await get_jobs(db, skip=skip, limit=limit)
    return [JobResponse.from_orm(job) for job in jobs]


@router.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job_by_id(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get job by ID"""
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.from_orm(job)


@router.get("/api/workers", response_model=list[WorkerResponse])
async def list_workers(db: AsyncSession = Depends(get_db)):
    """List all workers"""
    workers = await get_workers(db)
    return [WorkerResponse.from_orm(worker) for worker in workers]


@router.delete("/api/database/clear")
async def clear_all_data(db: AsyncSession = Depends(get_db)):
    """Clear all data from the database (workers, jobs, tasks, failures)"""
    result = await clear_database(db)
    return {
        "message": f"Database cleared successfully. Deleted: {result['workers_deleted']} workers, {result['jobs_deleted']} jobs, {result['tasks_deleted']} tasks, {result['worker_failures_deleted']} failures."
    }


@router.delete("/api/workers/{worker_id}")
async def remove_worker(worker_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a worker by ID"""
    deleted = await delete_worker(db, worker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {"message": f"Worker {worker_id} deleted successfully"}


@router.get("/api/workers/{worker_id}/failures", response_model=dict)
async def get_worker_failures_endpoint(worker_id: str, skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Get a worker's failure history and failure rate"""
    failures = await get_worker_failures(db, worker_id, skip=skip, limit=limit)
    failures_response = [WorkerFailureResponse.from_orm(f) for f in failures]
    stats: WorkerFailureStats = await get_worker_failure_stats(db, worker_id)
    return {
        "worker_id": worker_id,
        "failures": [f.dict() for f in failures_response],
        "stats": stats.dict()
    }


# ==================== Checkpointing Dashboard API ====================

@router.get("/api/checkpoints/job/{job_id}")
async def get_job_checkpoint_dashboard(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get comprehensive checkpoint data for dashboard display.
    
    Returns job-level summary and per-task checkpoint details with timeline events.
    """
    import json
    from datetime import datetime
    from sqlalchemy.orm import selectinload
    
    # Get job with tasks (eagerly loaded)
    result = await db.execute(
        select(JobModel)
        .options(selectinload(JobModel.tasks))
        .where(JobModel.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get all workers for status lookup
    workers_result = await db.execute(select(WorkerModel))
    workers = {w.id: w for w in workers_result.scalars().all()}
    
    # Get worker failures for this job
    failures_result = await db.execute(
        select(WorkerFailureModel).where(WorkerFailureModel.job_id == job_id)
    )
    failures = failures_result.scalars().all()
    failures_by_task = {}
    for f in failures:
        if f.task_id not in failures_by_task:
            failures_by_task[f.task_id] = []
        failures_by_task[f.task_id].append(f)
    
    # Process tasks
    tasks_data = []
    total_checkpoints = 0
    resumed_count = 0
    failed_count = 0
    
    for task in job.tasks:
        # Check if task was resumed (has failure with checkpoint_available)
        task_failures = failures_by_task.get(task.id, [])
        was_resumed = any(f.checkpoint_available for f in task_failures)
        if was_resumed:
            resumed_count += 1
        
        if task.status == 'failed':
            failed_count += 1
        
        # Get worker info
        worker = workers.get(task.worker_id) if task.worker_id else None
        worker_status = worker.status if worker else None
        
        # Build timeline events
        timeline = []
        
        # Add checkpoint events from delta_checkpoint_blobs
        if task.delta_checkpoint_blobs:
            try:
                blobs = json.loads(task.delta_checkpoint_blobs)
                checkpoint_ids = sorted([int(k) for k in blobs.keys()])
                
                for i, cp_id in enumerate(checkpoint_ids):
                    event_type = 'base_checkpoint' if cp_id == 1 else 'delta_checkpoint'
                    # Estimate timestamp based on task creation and checkpoint count
                    # In real implementation, store timestamps with checkpoints
                    timeline.append({
                        'type': event_type,
                        'description': f'Checkpoint #{cp_id}' + (' (Base)' if cp_id == 1 else ''),
                        'timestamp': task.last_checkpoint_at.isoformat() if task.last_checkpoint_at else datetime.now().isoformat()
                    })
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Add failure events
        for failure in task_failures:
            timeline.append({
                'type': 'failure',
                'description': f'Worker {failure.worker_id[:8]}... failed',
                'timestamp': failure.failed_at.isoformat() if failure.failed_at else datetime.now().isoformat()
            })
            
            if failure.checkpoint_available:
                timeline.append({
                    'type': 'resume',
                    'description': 'Task resumed from checkpoint',
                    'timestamp': failure.failed_at.isoformat() if failure.failed_at else datetime.now().isoformat()
                })
        
        # Add completion event
        if task.status == 'completed' and task.completed_at:
            timeline.append({
                'type': 'complete',
                'description': 'Task completed',
                'timestamp': task.completed_at.isoformat()
            })
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        total_checkpoints += task.checkpoint_count or 0
        
        tasks_data.append({
            'task_id': task.id,
            'worker_id': task.worker_id,
            'worker_status': worker_status,
            'status': task.status,
            'progress_percent': task.progress_percent or 0,
            'checkpoint_count': task.checkpoint_count or 0,
            'last_checkpoint_at': task.last_checkpoint_at.isoformat() if task.last_checkpoint_at else None,
            'resumed_from_checkpoint': was_resumed,
            'timeline': timeline
        })
    
    # Sort tasks: running first, then by progress
    tasks_data.sort(key=lambda t: (
        0 if t['status'] == 'assigned' else 1 if t['status'] == 'pending' else 2,
        -t['progress_percent']
    ))
    
    return {
        'job_id': job_id,
        'job_status': job.status,
        'total_tasks': job.total_tasks or 0,
        'completed_tasks': job.completed_tasks or 0,
        'failed_tasks': failed_count,
        'resumed_tasks': resumed_count,
        'total_checkpoints': total_checkpoints,
        'checkpointing_enabled': total_checkpoints > 0,
        'tasks': tasks_data
    }


@router.get("/api/checkpoints/recovery-events")
async def get_recovery_events(limit: int = 20, db: AsyncSession = Depends(get_db)):
    """
    Get recent recovery events (worker failures, task resumptions).
    
    Returns a list of events for the recovery events log.
    """
    from sqlalchemy import desc
    
    # Get recent worker failures
    result = await db.execute(
        select(WorkerFailureModel)
        .order_by(desc(WorkerFailureModel.failed_at))
        .limit(limit)
    )
    failures = result.scalars().all()
    
    events = []
    for failure in failures:
        # Add failure event
        events.append({
            'timestamp': failure.failed_at.isoformat() if failure.failed_at else None,
            'event_type': 'worker_failure',
            'task_id': failure.task_id,
            'worker_id': failure.worker_id,
            'details': failure.error_message[:100] if failure.error_message else 'Worker disconnected'
        })
        
        # Add resume event if checkpoint was available
        if failure.checkpoint_available:
            events.append({
                'timestamp': failure.failed_at.isoformat() if failure.failed_at else None,
                'event_type': 'task_resumed',
                'task_id': failure.task_id,
                'worker_id': failure.worker_id,
                'details': 'Task eligible for checkpoint recovery'
            })
    
    # Sort by timestamp descending
    events.sort(key=lambda x: x['timestamp'] or '', reverse=True)
    
    return events[:limit]

# ==================== Scheduling Visualization API ====================

@router.get("/api/scheduling-info")
async def get_scheduling_info(db: AsyncSession = Depends(get_db)):
    """
    Get comprehensive scheduling visualization data.
    
    Returns:
    - Active scheduler algorithm and config
    - Per-worker: device specs, score, assigned tasks, task history
    - Per-job: task distribution across workers (Gantt-style data)
    """
    import json
    from sqlalchemy.orm import selectinload

    # 1. Get active scheduler config
    from foreman.db.models import SchedulerConfigModel
    sched_result = await db.execute(
        select(SchedulerConfigModel).where(SchedulerConfigModel.is_active == True)
    )
    active_config = sched_result.scalar_one_or_none()

    scheduler_info = None
    if active_config:
        scheduler_info = {
            "algorithm": active_config.algorithm_name,
            "description": active_config.description,
            "criteria_weights": json.loads(active_config.criteria_weights) if active_config.criteria_weights else [],
            "criteria_names": json.loads(active_config.criteria_names) if active_config.criteria_names else [],
            "criteria_types": json.loads(active_config.criteria_types) if active_config.criteria_types else [],
        }
    else:
        # Check if simple scheduler is being used (default FIFO)
        scheduler_info = {
            "algorithm": "fifo",
            "description": "First-In-First-Out (default)",
            "criteria_weights": [],
            "criteria_names": [],
            "criteria_types": [],
        }

    # 2. Get all workers with full device specs
    workers = await get_workers(db)
    workers_data = []
    for w in workers:
        # Compute success rate
        total = (w.total_tasks_completed or 0) + (w.total_tasks_failed or 0)
        success_rate = (w.total_tasks_completed / total * 100) if total > 0 else 100.0

        workers_data.append({
            "id": w.id,
            "status": w.status,
            "device_type": w.device_type or "Unknown",
            "os_type": w.os_type,
            "cpu_model": w.cpu_model,
            "cpu_cores": w.cpu_cores,
            "cpu_frequency_mhz": w.cpu_frequency_mhz,
            "ram_total_mb": w.ram_total_mb,
            "ram_available_mb": w.ram_available_mb,
            "battery_level": w.battery_level,
            "is_charging": w.is_charging,
            "current_task_id": w.current_task_id,
            "total_tasks_completed": w.total_tasks_completed or 0,
            "total_tasks_failed": w.total_tasks_failed or 0,
            "success_rate": round(success_rate, 1),
            "last_seen": w.last_seen.isoformat() if w.last_seen else None,
        })

    # 3. Get all jobs with task assignments for Gantt-style view
    jobs_result = await db.execute(
        select(JobModel)
        .options(selectinload(JobModel.tasks))
        .order_by(JobModel.created_at.desc())
        .limit(10)
    )
    jobs = jobs_result.scalars().all()

    jobs_data = []
    for job in jobs:
        task_assignments = []
        for task in job.tasks:
            task_assignments.append({
                "task_id": task.id,
                "worker_id": task.worker_id,
                "status": task.status,
                "assigned_at": task.assigned_at.isoformat() if task.assigned_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            })
        
        # Sort tasks by assigned_at
        task_assignments.sort(key=lambda t: t["assigned_at"] or "")

        jobs_data.append({
            "job_id": job.id,
            "status": job.status,
            "total_tasks": job.total_tasks or 0,
            "completed_tasks": job.completed_tasks or 0,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "task_assignments": task_assignments,
        })

    return {
        "scheduler": scheduler_info,
        "workers": workers_data,
        "jobs": jobs_data,
    }