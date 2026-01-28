from datetime import datetime
from typing import Optional, List, Tuple
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from foreman.schema.schema import JobStats, WorkerFailureStats
from .models import *


async def create_job(session: AsyncSession, job_id: str, total_tasks: int) -> JobModel:
    """Create a new job"""
    job = JobModel(id=job_id, total_tasks=total_tasks)
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def create_task(session: AsyncSession, task_id: str, job_id: str) -> TaskModel:
    """Create a new task"""
    task = TaskModel(id=task_id, job_id=job_id)
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def create_worker(session: AsyncSession, worker_id: str, device_specs: dict = None) -> WorkerModel:
    """
    Create a new worker with optional device specifications
    
    Args:
        session: Database session
        worker_id: Worker identifier
        device_specs: Optional dictionary with device specifications
    """
    worker_data = {"id": worker_id}
    
    # Add device specs if provided
    if device_specs:
        worker_data.update({
            "device_type": device_specs.get("device_type"),
            "os_type": device_specs.get("os_type"),
            "os_version": device_specs.get("os_version"),
            "cpu_model": device_specs.get("cpu_model"),
            "cpu_cores": device_specs.get("cpu_cores"),
            "cpu_threads": device_specs.get("cpu_threads"),
            "cpu_frequency_mhz": device_specs.get("cpu_frequency_mhz"),
            "ram_total_mb": device_specs.get("ram_total_mb"),
            "ram_available_mb": device_specs.get("ram_available_mb"),
            "gpu_model": device_specs.get("gpu_model"),
            "battery_level": device_specs.get("battery_level"),
            "is_charging": device_specs.get("is_charging"),
            "network_type": device_specs.get("network_type"),
            "python_version": device_specs.get("python_version"),
        })
    
    worker = WorkerModel(**worker_data)
    session.add(worker)
    await session.commit()
    await session.refresh(worker)
    return worker


async def update_worker_device_specs(session: AsyncSession, worker_id: str, device_specs: dict):
    """
    Update worker device specifications
    
    Args:
        session: Database session
        worker_id: Worker identifier
        device_specs: Dictionary with device specifications
    """
    update_data = {
        "device_type": device_specs.get("device_type"),
        "os_type": device_specs.get("os_type"),
        "os_version": device_specs.get("os_version"),
        "cpu_model": device_specs.get("cpu_model"),
        "cpu_cores": device_specs.get("cpu_cores"),
        "cpu_threads": device_specs.get("cpu_threads"),
        "cpu_frequency_mhz": device_specs.get("cpu_frequency_mhz"),
        "ram_total_mb": device_specs.get("ram_total_mb"),
        "ram_available_mb": device_specs.get("ram_available_mb"),
        "gpu_model": device_specs.get("gpu_model"),
        "battery_level": device_specs.get("battery_level"),
        "is_charging": device_specs.get("is_charging"),
        "network_type": device_specs.get("network_type"),
        "python_version": device_specs.get("python_version"),
        "connected_at": datetime.now(),
    }
    
    stmt = update(WorkerModel).where(WorkerModel.id == worker_id).values(**update_data)
    await session.execute(stmt)
    await session.commit()


async def get_job(session: AsyncSession, job_id: str) -> Optional[JobModel]:
    """Get job by ID"""
    result = await session.execute(select(JobModel).where(JobModel.id == job_id))
    return result.scalar_one_or_none()


async def get_jobs(
    session: AsyncSession, skip: int = 0, limit: int = 100
) -> List[JobModel]:
    """Get all jobs with pagination"""
    result = await session.execute(
        select(JobModel).order_by(JobModel.created_at.desc()).offset(skip).limit(limit)
    )
    return result.scalars().all()


async def get_workers(session: AsyncSession) -> List[WorkerModel]:
    """Get all workers"""
    result = await session.execute(select(WorkerModel))
    return result.scalars().all()


async def delete_worker(session: AsyncSession, worker_id: str) -> bool:
    """Delete a worker by ID"""
    result = await session.execute(select(WorkerModel).filter_by(id=worker_id))
    worker = result.scalar_one_or_none()
    if worker:
        await session.delete(worker)
        await session.commit()
        return True
    return False


async def clear_database(session: AsyncSession) -> dict:
    """Clear all workers, jobs, tasks, and worker failures from the database"""
    from sqlalchemy import delete as sql_delete
    
    # Delete in order to respect foreign key constraints
    worker_failures_result = await session.execute(sql_delete(WorkerFailureModel))
    tasks_result = await session.execute(sql_delete(TaskModel))
    jobs_result = await session.execute(sql_delete(JobModel))
    workers_result = await session.execute(sql_delete(WorkerModel))
    
    await session.commit()
    
    return {
        "workers_deleted": workers_result.rowcount,
        "jobs_deleted": jobs_result.rowcount,
        "tasks_deleted": tasks_result.rowcount,
        "worker_failures_deleted": worker_failures_result.rowcount
    }


async def update_job_status(
    session: AsyncSession, job_id: str, status: str, completed_tasks: int = None
):
    """Update job status"""
    update_data = {"status": status}
    if completed_tasks is not None:
        update_data["completed_tasks"] = completed_tasks
    if status == "completed":
        update_data["completed_at"] = datetime.now()

    stmt = update(JobModel).where(JobModel.id == job_id).values(**update_data)
    await session.execute(stmt)
    await session.commit()


async def update_task_status(
    session: AsyncSession,
    task_id: str,
    status: str,
    worker_id: str = None,
    result: str = None,
    error: str = None,
):
    """Update task status"""
    update_data = {"status": status}
    if worker_id:
        update_data["worker_id"] = worker_id
    if result:
        update_data["result"] = result
    if error:
        update_data["error_message"] = error

    if status == "assigned":
        update_data["assigned_at"] = datetime.now()
    elif status in ["completed", "failed"]:
        update_data["completed_at"] = datetime.now()

    stmt = update(TaskModel).where(TaskModel.id == task_id).values(**update_data)
    await session.execute(stmt)
    await session.commit()


async def claim_task_for_worker(
    session: AsyncSession, task_id: str, worker_id: str
) -> Optional[str]:
    """Atomically claim a pending task for a worker. Returns job_id if claimed."""
    stmt = (
        update(TaskModel)
        .where(TaskModel.id == task_id, TaskModel.status == "pending")
        .values(
            status="assigned",
            worker_id=worker_id,
            assigned_at=datetime.now(),
        )
        .returning(TaskModel.job_id)
    )

    result = await session.execute(stmt)
    row = result.fetchone()
    if not row:
        await session.rollback()
        return None

    await session.commit()
    return row[0]


async def update_worker_status(
    session: AsyncSession, worker_id: str, status: str, current_task_id: str = None
):
    """Update worker status"""
    update_data = {"status": status, "last_seen": datetime.now()}
    if current_task_id:
        update_data["current_task_id"] = current_task_id

    stmt = update(WorkerModel).where(WorkerModel.id == worker_id).values(**update_data)
    await session.execute(stmt)
    await session.commit()


async def update_worker_task_stats(
    session: AsyncSession, worker_id: str, task_completed: bool = True
):
    """Update worker task completion statistics"""
    if task_completed:
        # Increment completed tasks
        stmt = (
            update(WorkerModel)
            .where(WorkerModel.id == worker_id)
            .values(total_tasks_completed=WorkerModel.total_tasks_completed + 1)
        )
    else:
        # Increment failed tasks
        stmt = (
            update(WorkerModel)
            .where(WorkerModel.id == worker_id)
            .values(total_tasks_failed=WorkerModel.total_tasks_failed + 1)
        )

    await session.execute(stmt)
    await session.commit()


async def increment_job_completed_tasks(session: AsyncSession, job_id: str):
    """Increment the completed_tasks count for a job"""
    stmt = (
        update(JobModel)
        .where(JobModel.id == job_id)
        .values(completed_tasks=JobModel.completed_tasks + 1)
    )
    await session.execute(stmt)
    await session.commit()


async def complete_task_if_assigned(
    session: AsyncSession,
    task_id: str,
    worker_id: str,
    result: str,
) -> Tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """
    Atomically mark a task as completed only if it is currently assigned to this worker.

    Returns (accepted, job_id, completed_tasks, total_tasks).
    """

    task = await session.get(TaskModel, task_id)
    if not task:
        await session.rollback()
        return False, None, None, None

    if task.status != "assigned":
        await session.rollback()
        return False, task.job_id, None, None

    if task.worker_id and task.worker_id != worker_id:
        await session.rollback()
        return False, task.job_id, None, None

    task.status = "completed"
    task.result = result
    task.completed_at = datetime.now()
    if not task.worker_id:
        task.worker_id = worker_id

    job = await session.get(JobModel, task.job_id)
    if not job:
        await session.rollback()
        return False, task.job_id, None, None

    prev_completed = job.completed_tasks or 0
    total_tasks = job.total_tasks or 0
    if prev_completed < total_tasks:
        job.completed_tasks = prev_completed + 1

    await session.commit()

    return True, job.id, job.completed_tasks, job.total_tasks


async def record_worker_failure(
    session: AsyncSession,
    worker_id: str,
    task_id: str,
    error_message: str,
    job_id: Optional[str] = None,
) -> None:
    """Insert a worker failure record"""
    failure = WorkerFailureModel(
        worker_id=worker_id,
        task_id=task_id,
        job_id=job_id or "",
        error_message=error_message,
        failed_at=datetime.now(),
    )
    session.add(failure)
    await session.commit()


async def get_worker_failures(
    session: AsyncSession, worker_id: str, skip: int = 0, limit: int = 100
) -> List[WorkerFailureModel]:
    """Fetch failure history for a worker"""
    result = await session.execute(
        select(WorkerFailureModel)
        .where(WorkerFailureModel.worker_id == worker_id)
        .order_by(WorkerFailureModel.failed_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def get_worker_failure_stats(
    session: AsyncSession, worker_id: str
) -> WorkerFailureStats:
    """Compute failure stats and rate for a worker"""
    total_failures_result = await session.execute(
        select(func.count(WorkerFailureModel.id)).where(
            WorkerFailureModel.worker_id == worker_id
        )
    )
    total_failures = total_failures_result.scalar() or 0

    # Total tasks for worker = completed + failed from Workers table
    worker_result = await session.execute(
        select(WorkerModel).where(WorkerModel.id == worker_id)
    )
    worker: Optional[WorkerModel] = worker_result.scalar_one_or_none()
    if worker:
        total_tasks = (worker.total_tasks_completed or 0) + (
            worker.total_tasks_failed or 0
        )
    else:
        total_tasks = 0

    failure_rate = (
        float(total_failures) / float(total_tasks) if total_tasks > 0 else 0.0
    )

    return WorkerFailureStats(
        worker_id=worker_id,
        total_failures=total_failures,
        total_tasks=total_tasks,
        failure_rate=round(failure_rate, 4),
    )


async def get_online_workers_count(session: AsyncSession) -> int:
    """Get count of online workers"""
    result = await session.execute(
        select(func.count(WorkerModel.id)).where(WorkerModel.status == "online")
    )
    return result.scalar() or 0


async def get_job_stats(session: AsyncSession) -> JobStats:
    """Get job statistics"""
    # Get all jobs and count by status
    jobs = await get_jobs(session, skip=0, limit=10000)  # Get all jobs

    total_jobs = len(jobs)
    pending_jobs = len([j for j in jobs if j.status == "pending"])
    running_jobs = len([j for j in jobs if j.status == "running"])
    completed_jobs = len([j for j in jobs if j.status == "completed"])
    failed_jobs = len([j for j in jobs if j.status == "failed"])

    # Get all tasks and count by status
    result = await session.execute(select(TaskModel))
    tasks = result.scalars().all()

    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t.status == "completed"])
    failed_tasks = len([t for t in tasks if t.status == "failed"])

    # Get online workers count
    online_workers = await get_online_workers_count(session)

    return JobStats(
        total_jobs=total_jobs,
        pending_jobs=pending_jobs,
        running_jobs=running_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
        online_workers=online_workers,
    )


async def get_job_tasks(session: AsyncSession, job_id: str) -> List[TaskModel]:
    """Get all tasks for a job"""
    result = await session.execute(select(TaskModel).where(TaskModel.job_id == job_id))
    return result.scalars().all()


async def get_pending_tasks(
    session: AsyncSession, job_id: str = None
) -> List[TaskModel]:
    """Get pending tasks, optionally filtered by job_id"""
    query = select(TaskModel).where(TaskModel.status == "pending")
    if job_id:
        query = query.where(TaskModel.job_id == job_id)
    
    # Order by task ID to ensure sequential processing
    query = query.order_by(TaskModel.id)

    result = await session.execute(query)
    return result.scalars().all()


async def get_assigned_tasks(
    session: AsyncSession, job_id: str = None
) -> List[TaskModel]:
    """Get tasks with 'assigned' status, optionally filtered by job_id"""
    query = select(TaskModel).where(TaskModel.status == "assigned")
    if job_id:
        query = query.where(TaskModel.job_id == job_id)
    
    query = query.order_by(TaskModel.id)
    result = await session.execute(query)
    return result.scalars().all()


async def get_job_by_id(session: AsyncSession, job_id: str) -> Optional[JobModel]:
    """Get job by ID (alias for get_job)"""
    return await get_job(session, job_id)
