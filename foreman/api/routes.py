
from fastapi import  Depends, HTTPException, APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from foreman.db.base import get_db
from foreman.db.crud import (
    get_jobs, get_job, get_workers, get_job_stats,
    get_worker_failures, get_worker_failure_stats
)
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


