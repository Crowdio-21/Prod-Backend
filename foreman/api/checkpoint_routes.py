"""
FastAPI routes for checkpoint monitoring and management

Provides REST endpoints for:
- Viewing job/task checkpoint progress
- Triggering manual checkpoints (debugging)
- Monitoring checkpoint storage
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from foreman.db.base import get_db_session
from foreman.db.models import TaskModel, JobModel

# Module-level router for compatibility
router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])


def create_checkpoint_routes(checkpoint_manager, checkpoint_recovery_manager):
    """
    Create checkpoint API routes

    Args:
        checkpoint_manager: CheckpointManager instance
        checkpoint_recovery_manager: CheckpointRecoveryManager instance

    Returns:
        FastAPI APIRouter with checkpoint endpoints
    """
    checkpoint_router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])

    @checkpoint_router.get("/jobs/{job_id}")
    async def get_job_checkpoint_progress(
        job_id: str, session: AsyncSession = Depends(get_db_session)
    ):
        """
        Get checkpoint progress for all tasks in a job
        
        Returns overall progress, per-task checkpoint info, and declarative
        checkpointing configuration if enabled.
        """
        try:
            job = await session.get(JobModel, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            tasks_with_checkpoints = []
            total_progress = 0
            checkpoint_count = 0
            resumed_count = 0
            failed_count = 0
            completed_count = 0

            for task in job.tasks:
                task_info = {
                    "task_id": task.id,
                    "status": task.status,
                    "worker_id": task.worker_id,
                    "progress_percent": task.progress_percent or 0,
                    "checkpoint_count": task.checkpoint_count or 0,
                    "last_checkpoint_at": (
                        task.last_checkpoint_at.isoformat()
                        if task.last_checkpoint_at
                        else None
                    ),
                    "base_checkpoint_size": task.base_checkpoint_size or 0,
                    "has_checkpoint": bool(task.base_checkpoint_data),
                }
                
                # Parse delta checkpoints for declarative info
                import json
                if task.delta_checkpoints:
                    try:
                        deltas = json.loads(task.delta_checkpoints)
                        task_info["delta_count"] = len(deltas)
                        # Extract state_vars from latest delta if present
                        if deltas and "state_vars" in deltas[-1]:
                            task_info["checkpoint_state_vars"] = deltas[-1]["state_vars"]
                    except json.JSONDecodeError:
                        task_info["delta_count"] = 0
                else:
                    task_info["delta_count"] = 0
                
                tasks_with_checkpoints.append(task_info)
                total_progress += task.progress_percent or 0
                checkpoint_count += task.checkpoint_count or 0
                
                # Track status counts
                if task.status == "completed":
                    completed_count += 1
                elif task.status == "failed":
                    failed_count += 1

            avg_progress = total_progress / len(job.tasks) if job.tasks else 0

            return {
                "job_id": job_id,
                "job_status": job.status,
                "total_tasks": job.total_tasks,
                "completed_tasks": completed_count,
                "failed_tasks": failed_count,
                "resumed_tasks": resumed_count,  # TODO: Track resumed tasks
                "tasks_with_checkpoints": sum(1 for t in tasks_with_checkpoints if t["has_checkpoint"]),
                "average_progress_percent": avg_progress,
                "total_checkpoints": checkpoint_count,
                "checkpointing_enabled": job.supports_checkpointing,
                "tasks": tasks_with_checkpoints,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @checkpoint_router.get("/tasks/{task_id}")
    async def get_task_checkpoint_info(
        task_id: str, session: AsyncSession = Depends(get_db_session)
    ):
        """
        Get detailed checkpoint information for a specific task
        """
        try:
            task = await session.get(TaskModel, task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            if not task.base_checkpoint_data:
                return {
                    "task_id": task_id,
                    "has_checkpoint": False,
                    "message": "No checkpoint available for this task",
                }

            import json

            deltas = json.loads(task.delta_checkpoints or "[]")

            return {
                "task_id": task_id,
                "job_id": task.job_id,
                "status": task.status,
                "has_checkpoint": True,
                "progress_percent": task.progress_percent,
                "checkpoint_count": task.checkpoint_count,
                "base_checkpoint_size_bytes": task.base_checkpoint_size,
                "delta_count": len(deltas),
                "last_checkpoint_at": (
                    task.last_checkpoint_at.isoformat()
                    if task.last_checkpoint_at
                    else None
                ),
                "storage_path": task.checkpoint_storage_path,
                "deltas": deltas[:5],  # Show last 5 deltas
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @checkpoint_router.post("/tasks/{task_id}/force-checkpoint")
    async def force_checkpoint_cleanup(
        task_id: str, session: AsyncSession = Depends(get_db_session)
    ):
        """
        Force cleanup of checkpoint data for a task (debugging)

        Use with caution - will delete checkpoint data for recovery.
        Only use on completed or failed tasks.
        """
        try:
            task = await session.get(TaskModel, task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            if task.status not in ["completed", "failed"]:
                raise HTTPException(
                    status_code=400,
                    detail="Can only cleanup checkpoints for completed/failed tasks",
                )

            success = await checkpoint_manager.cleanup_checkpoint(session, task_id)

            return {
                "task_id": task_id,
                "action": "checkpoint_cleanup",
                "success": success,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    @checkpoint_router.get("/storage-info/{task_id}")
    async def get_checkpoint_storage_info(task_id: str):
        """
        Get storage location and size info for a task's checkpoints
        """
        try:
            info = checkpoint_manager.storage_handler.get_checkpoint_info(task_id)
            return info
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return checkpoint_router
