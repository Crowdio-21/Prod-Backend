import json
import base64
import gzip
import pickle
from typing import Optional
from ...db.base import db_session
from ...db.crud import (
    get_pending_tasks,
    get_assigned_tasks,
    get_job_tasks,
    get_job_by_id,
    create_job,
    create_worker,
    get_workers,
    update_worker_device_specs,
    update_worker_status,
    update_job_status,
    update_task_status,
    claim_task_for_worker,
    update_worker_task_stats,
    increment_job_completed_tasks,
    complete_task_if_assigned,
    record_worker_failure,
    get_latest_task_failure_with_checkpoint,
)
from ...db.models import TaskModel


def _decode_checkpoint_blob(blob_base64: str) -> Optional[dict]:
    """
    Decode a checkpoint blob from delta_checkpoint_blobs
    
    The blob encoding chain is:
    - PC Worker: pickle(delta_dict) → gzip compress → gzip compress → base64 encode
    - Android Worker: JSON(delta_dict) → gzip compress → gzip compress → base64 encode
    
    So to decode: base64 decode → gzip decompress → gzip decompress → (pickle or JSON)
    
    Args:
        blob_base64: Base64 encoded double-compressed blob
        
    Returns:
        Decoded dictionary or None if decoding fails
    """
    try:
        # Step 1: base64 decode
        compressed_data = base64.b64decode(blob_base64)
        
        # Step 2: First gzip decompress (storage handler compression)
        first_decompress = gzip.decompress(compressed_data)
        
        # Step 3: Second gzip decompress (worker compression)
        second_decompress = gzip.decompress(first_decompress)
        
        # Step 4: Try to decode as JSON first (Android workers), then pickle (PC workers)
        # JSON is more common for cross-platform, so try it first
        try:
            # Try JSON decode (Android workers send JSON)
            checkpoint_state = json.loads(second_decompress.decode('utf-8'))
            print(f"_decode_checkpoint_blob: Decoded as JSON")
            return checkpoint_state
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle (PC workers send pickle)
            checkpoint_state = pickle.loads(second_decompress)
            print(f"_decode_checkpoint_blob: Decoded as pickle")
            return checkpoint_state
            
    except Exception as e:
        print(f"_decode_checkpoint_blob: Error decoding checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def _make_json_serializable(obj):
    """Convert non-JSON-serializable objects to serializable format"""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, bytes):
        return f"<bytes: {len(obj)} bytes>"
    elif hasattr(obj, '__dict__'):
        return f"<{type(obj).__name__}: {str(obj)[:100]}>"
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


async def _record_worker_failure(
    worker_id, 
    task_id, 
    error, 
    job_id, 
    checkpoint_available: bool = False,
    latest_checkpoint_data: Optional[str] = None
):
    """
    Record a worker failure with optional checkpoint data
    
    Args:
        worker_id: Worker identifier
        task_id: Task identifier
        error: Error message
        job_id: Job identifier
        checkpoint_available: Whether checkpoint exists for recovery
        latest_checkpoint_data: JSON string of decoded latest checkpoint delta
    """
    async with db_session() as session:
        await record_worker_failure(
            session, 
            worker_id, 
            task_id, 
            error_message=str(error), 
            job_id=job_id,
            checkpoint_available=checkpoint_available,
            latest_checkpoint_data=latest_checkpoint_data,
        )


async def _get_latest_task_failure_with_checkpoint(task_id: str) -> Optional[dict]:
    """
    Get the latest failure record with checkpoint data for a task.
    
    Used when reassigning orphaned tasks to check if we can resume
    from a checkpoint instead of starting from scratch.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Dictionary with checkpoint data including 'state' field, or None if not found
    """
    async with db_session() as session:
        failure = await get_latest_task_failure_with_checkpoint(session, task_id)
        if failure and failure.latest_checkpoint_data:
            try:
                return json.loads(failure.latest_checkpoint_data)
            except json.JSONDecodeError:
                print(f"_get_latest_task_failure_with_checkpoint: Invalid JSON for task {task_id}")
                return None
        return None


async def _get_job_tasks(job_id):
    # Get all tasks for this job
    async with db_session() as session:
        tasks = await get_job_tasks(session, job_id)
    return tasks


async def _get_pending_tasks(job_id=None):
    # Get pending tasks for this job
    async with db_session() as session:
        pending_tasks = await get_pending_tasks(session, job_id)
    return pending_tasks


async def _get_assigned_tasks(job_id=None):
    """Get tasks with 'assigned' status, optionally filtered by job_id"""
    async with db_session() as session:
        assigned_tasks = await get_assigned_tasks(session, job_id)
    return assigned_tasks


async def _get_tasks_needing_recovery():
    """Get tasks that need recovery - either assigned or pending with a worker_id"""
    from ...db.crud import get_tasks_needing_recovery
    async with db_session() as session:
        return await get_tasks_needing_recovery(session)


async def _get_job_by_id(job_id: str):
    """Get job by ID from database"""

    async with db_session() as session:
        return await get_job_by_id(session, job_id)


async def _create_job_in_database(job_id: str, total_tasks: int, supports_checkpointing: bool = False):
    """
    Create job in database
    
    Args:
        job_id: Job identifier
        total_tasks: Total number of tasks
        supports_checkpointing: Whether job tasks support checkpointing
    """

    # Create a new session for this operation
    async with db_session() as session:
        await create_job(session, job_id, total_tasks, supports_checkpointing=supports_checkpointing)
    print(f"Created job {job_id} in database (checkpointing={'enabled' if supports_checkpointing else 'disabled'})")


async def _create_worker_in_database(worker_id: str, device_specs: dict = None):
    """
    Create worker in database if it doesn't exist, or update with device specs

    Args:
        worker_id: Worker identifier
        device_specs: Dictionary containing device specifications
    """

    # Create a new session for this operation
    async with db_session() as session:
        # Check if worker already exists
        workers = await get_workers(session)
        existing_worker_ids = [w.id for w in workers]

        if worker_id not in existing_worker_ids:
            # Create new worker with device specs
            await create_worker(session, worker_id, device_specs=device_specs)
            print(f"Created worker {worker_id} in database")
        else:
            # Worker exists, update device specs and mark as online
            if device_specs:
                await update_worker_device_specs(session, worker_id, device_specs)
                print(f"Updated device specs for worker {worker_id}")
            await update_worker_status(session, worker_id, "online")
            print(f"Worker {worker_id} already exists in database, marked as online")


async def _create_tasks_for_job(job_id: str, args_list: list):
    """Create tasks in database for a job"""

    # Create a new session for this operation
    async with db_session() as session:
        tasks = []
        for i, args in enumerate(args_list):
            task_id = f"{job_id}_task_{i}"
            task = TaskModel(
                id=task_id,
                job_id=job_id,
                args=json.dumps(args),  # Serialize args to JSON string
                status="pending",
            )
            tasks.append(task)

        # Add all tasks to database
        session.add_all(tasks)
        await session.commit()

    print(f"Created {len(tasks)} tasks for job {job_id}")


async def _update_job_status(job_id: str, status: str, completed_tasks: int = None):
    """Update job status with new session"""

    async with db_session() as session:
        await update_job_status(session, job_id, status, completed_tasks)


async def _update_task_status(
    task_id: str,
    status: str,
    worker_id: str = None,
    result: str = None,
    error: str = None,
    clear_worker: bool = False,
):
    """Update task status with new session
    
    Args:
        task_id: Task identifier
        status: New status
        worker_id: Worker ID to set (if provided)
        result: Task result (if provided)
        error: Error message (if provided)
        clear_worker: If True, explicitly clear worker_id (set to None)
    """

    async with db_session() as session:
        await update_task_status(session, task_id, status, worker_id, result, error, clear_worker)


async def _claim_task_for_worker(task_id: str, worker_id: str):
    """Claim a pending task for a worker atomically"""

    async with db_session() as session:
        return await claim_task_for_worker(session, task_id, worker_id)


async def _update_worker_status(
    worker_id: str, status: str, current_task_id: str = None
):
    """Update worker status with new session"""

    async with db_session() as session:
        await update_worker_status(session, worker_id, status, current_task_id)


async def _update_worker_task_stats(worker_id: str, task_completed: bool = True):
    """Update worker task statistics with new session"""

    async with db_session() as session:
        await update_worker_task_stats(session, worker_id, task_completed)


async def _increment_job_completed_tasks(job_id: str):
    """Increment job completed tasks count with new session"""

    async with db_session() as session:
        await increment_job_completed_tasks(session, job_id)


async def _complete_task_if_assigned(task_id: str, worker_id: str, result: str):
    """Complete a task only if it is currently assigned to this worker"""

    async with db_session() as session:
        return await complete_task_if_assigned(session, task_id, worker_id, result)


async def _get_worker_stats(worker_id: str):
    """Get worker statistics from database (basic version - for backward compatibility)"""
    from ...db.models import WorkerModel
    from sqlalchemy import select

    async with db_session() as session:
        result = await session.execute(
            select(WorkerModel).where(WorkerModel.id == worker_id)
        )
        worker = result.scalar_one_or_none()

        if worker:
            return {
                "status": worker.status,
                "tasks_completed": worker.total_tasks_completed or 0,
                "tasks_failed": worker.total_tasks_failed or 0,
                "current_task_id": worker.current_task_id,
            }
        return None


async def _get_worker_stats_extended(worker_id: str):
    """
    Get EXTENDED worker statistics including device specs for MCDM

    Args:
        worker_id: Worker identifier

    Returns:
        Dictionary with complete worker data including:
        - Basic: status, tasks_completed, tasks_failed
        - Device: cpu_cores, cpu_frequency, ram_total, ram_available
        - Performance: cpu_usage, avg_task_duration, success_rate
        - Battery: battery_level, is_charging
        - Network: network_type, network_speed
        - GPU: gpu_available, gpu_model
        - Storage: storage_available
    """
    from ...db.models import WorkerModel
    from sqlalchemy import select

    async with db_session() as session:
        result = await session.execute(select(WorkerModel).filter_by(id=worker_id))
        worker = result.scalar_one_or_none()

        if not worker:
            return {}

        return {
            # Basic stats
            "status": worker.status,
            "tasks_completed": worker.total_tasks_completed or 0,
            "tasks_failed": worker.total_tasks_failed or 0,
            "current_task_id": worker.current_task_id,
            # CPU
            "cpu_cores": worker.cpu_cores or 1,
            "cpu_threads": worker.cpu_threads or 1,
            "cpu_frequency_mhz": worker.cpu_frequency_mhz or 1000.0,
            "cpu_usage_percent": worker.cpu_usage_percent or 0.0,
            "cpu_model": worker.cpu_model,
            # RAM
            "ram_total_mb": worker.ram_total_mb or 1024.0,
            "ram_available_mb": worker.ram_available_mb or 512.0,
            # Battery
            "battery_level": worker.battery_level or 100.0,
            "is_charging": (
                worker.is_charging if worker.is_charging is not None else True
            ),
            # Network
            "network_type": worker.network_type or "WiFi",
            "network_speed_mbps": worker.network_speed_mbps or 10.0,
            # GPU
            "gpu_available": worker.gpu_available or False,
            "gpu_model": worker.gpu_model,
            # Storage
            "storage_available_gb": worker.storage_available_gb or 0.0,
            # Performance
            "avg_task_duration_sec": worker.avg_task_duration_sec or 0.0,
            # Device info
            "device_type": worker.device_type,
            "os_type": worker.os_type,
            "os_version": worker.os_version,
        }
