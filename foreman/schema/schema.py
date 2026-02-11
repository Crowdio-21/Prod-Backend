from pydantic import BaseModel, computed_field
from datetime import datetime
from typing import Optional, List

# Pydantic Models for API
class JobBase(BaseModel):
    total_tasks: int


class JobCreate(JobBase):
    pass


class JobResponse(JobBase):
    id: str
    status: str
    completed_tasks: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class TaskResponse(BaseModel):
    id: str
    job_id: str
    worker_id: Optional[str] = None
    status: str
    result: Optional[str] = None
    error_message: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class WorkerResponse(BaseModel):
    id: str
    status: str
    last_seen: datetime
    current_task_id: Optional[str] = None
    total_tasks_completed: int
    total_tasks_failed: int
    
    # Device specifications
    device_type: Optional[str] = None
    os_type: Optional[str] = None
    cpu_model: Optional[str] = None
    cpu_cores: Optional[int] = None
    cpu_threads: Optional[int] = None
    cpu_frequency_mhz: Optional[float] = None
    ram_total_mb: Optional[float] = None
    ram_available_mb: Optional[float] = None
    gpu_model: Optional[str] = None
    battery_level: Optional[float] = None
    is_charging: Optional[bool] = None
    
    # Computed device score for scheduling (0-100) based on WRR algorithm weights
    @computed_field
    @property
    def device_score(self) -> float:
        """
        Calculate device score based on MCDM WRR scheduling algorithm.
        
        Weights (from config_manager.py WRR config):
        - success_rate: 35% (computed from tasks completed/failed)
        - cpu_cores: 25% (normalized to max 16 cores)
        - ram_available_mb: 20% (normalized to max 16GB)
        - battery_level: 20% (0-100%)
        
        Returns score from 0-100
        """
        score = 0.0
        
        # Success rate: 35 points (based on task completion ratio)
        total_tasks = self.total_tasks_completed + self.total_tasks_failed
        if total_tasks > 0:
            success_rate = self.total_tasks_completed / total_tasks
            score += success_rate * 35
        else:
            # New worker with no tasks gets full points for success_rate
            score += 35
        
        # CPU cores: 25 points (normalized, max 16 cores = full points)
        if self.cpu_cores:
            score += min(self.cpu_cores / 16.0, 1.0) * 25
        
        # RAM available: 20 points (normalized, max 16GB = full points)
        if self.ram_available_mb:
            score += min(self.ram_available_mb / 16384.0, 1.0) * 20
        
        # Battery level: 20 points (direct percentage, charging = 100%)
        if self.is_charging:
            score += 20
        elif self.battery_level is not None:
            score += (self.battery_level / 100.0) * 20
        else:
            # Desktop with no battery info gets full points
            score += 20
        
        return round(score, 1)
    
    class Config:
        from_attributes = True


class WorkerFailureResponse(BaseModel):
    id: int
    worker_id: str
    task_id: str
    job_id: Optional[str] = None
    error_message: str
    failed_at: datetime
    checkpoint_available: bool = False
    latest_checkpoint_data: Optional[str] = None
    
    class Config:
        from_attributes = True


class WorkerFailureStats(BaseModel):
    worker_id: str
    total_failures: int
    total_tasks: int
    failure_rate: float


class WorkerFailureSummary(BaseModel):
    worker_id: str
    failures: List["WorkerFailureResponse"]
    stats: WorkerFailureStats


class JobStats(BaseModel):
    total_jobs: int
    pending_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    online_workers: int
