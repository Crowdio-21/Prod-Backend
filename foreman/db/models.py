from datetime import datetime 
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship

from .base import Base

# SQLAlchemy Models
class JobModel(Base):
    """Job table model"""
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    total_tasks = Column(Integer)
    completed_tasks = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    supports_checkpointing = Column(Boolean, default=False)
    #serialized code could be added here if needed
    #arguments for the job could be added here if needed
    
    # Relationships
    tasks = relationship("TaskModel", back_populates="job")


class TaskModel(Base):
    """Task table model"""
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    worker_id = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, assigned, completed, failed
    args = Column(Text, nullable=True)  # Serialized task arguments
    result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    assigned_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Checkpoint fields (for incremental checkpointing)
    base_checkpoint_data = Column(Text, nullable=True)  # Hex-encoded serialized base state
    base_checkpoint_size = Column(Integer, default=0)  # Bytes of base checkpoint
    delta_checkpoints = Column(Text, nullable=True)  # JSON array of delta checkpoints
    last_checkpoint_at = Column(DateTime, nullable=True)
    progress_percent = Column(Float, default=0.0)  # Task progress 0-100
    checkpoint_count = Column(Integer, default=0)  # Number of checkpoints taken
    checkpoint_storage_path = Column(String, nullable=True)  # Path if stored externally
    
    # Relationships
    job = relationship("JobModel", back_populates="tasks")


class WorkerModel(Base):
    """Worker table model"""
    __tablename__ = "workers"
    
    id = Column(String, primary_key=True)
    status = Column(String, default="online")  # online, offline, busy
    last_seen = Column(DateTime, default=datetime.now)
    current_task_id = Column(String, nullable=True)
    total_tasks_completed = Column(Integer, default=0)
    total_tasks_failed = Column(Integer, default=0)
    
    # Device specifications
    device_type = Column(String, nullable=True)  # PC or Android
    os_type = Column(String, nullable=True)  # Windows, Linux, macOS, Android
    os_version = Column(String, nullable=True)
    cpu_model = Column(String, nullable=True)
    cpu_cores = Column(Integer, nullable=True)
    cpu_threads = Column(Integer, nullable=True)
    cpu_frequency_mhz = Column(Float, nullable=True)
    ram_total_mb = Column(Float, nullable=True)
    ram_available_mb = Column(Float, nullable=True)
    gpu_model = Column(String, nullable=True)
    battery_level = Column(Float, nullable=True)  # For Android/laptops
    is_charging = Column(Boolean, nullable=True)
    network_type = Column(String, nullable=True)  # WiFi, Cellular, Ethernet
    python_version = Column(String, nullable=True)
    
    # Connection tracking
    first_connected_at = Column(DateTime, default=datetime.now)
    connected_at = Column(DateTime, default=datetime.now)
    disconnected_at = Column(DateTime, nullable=True)
    


class WorkerFailureModel(Base):
    """Historical record of worker task failures"""
    __tablename__ = "worker_failures"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String, ForeignKey("workers.id"))
    task_id = Column(String)
    job_id = Column(String)
    error_message = Column(Text)
    failed_at = Column(DateTime, default=datetime.now)
    checkpoint_available = Column(Boolean, default=False)  # Whether checkpoint exists for recovery

