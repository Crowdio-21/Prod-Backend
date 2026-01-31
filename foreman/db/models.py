from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Boolean,
    Float,
    LargeBinary,
)
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
    # serialized code could be added here if needed
    # arguments for the job could be added here if needed

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
    base_checkpoint_data = Column(
        Text, nullable=True
    )  # Storage reference (fs_path or db_id)
    base_checkpoint_size = Column(Integer, default=0)  # Bytes of base checkpoint
    base_checkpoint_blob = Column(LargeBinary, nullable=True)  # Small base checkpoint data (<1MB)
    delta_checkpoints = Column(Text, nullable=True)  # JSON array of delta checkpoints
    delta_checkpoint_blobs = Column(Text, nullable=True)  # JSON map of delta_id -> base64 encoded blob
    last_checkpoint_at = Column(DateTime, nullable=True)
    progress_percent = Column(Float, default=0.0)  # Task progress 0-100
    checkpoint_count = Column(Integer, default=0)  # Number of checkpoints taken
    checkpoint_storage_path = Column(String, nullable=True)  # Path if stored externally or 'db' if in blob

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

    # MCDM Performance Metrics (added for intelligent task allocation)
    cpu_usage_percent = Column(Float, nullable=True)  # Current CPU usage (0-100)
    gpu_available = Column(Boolean, default=False)  # Whether GPU is available
    storage_available_gb = Column(Float, nullable=True)  # Available disk space in GB
    network_speed_mbps = Column(Float, nullable=True)  # Network bandwidth in Mbps
    avg_task_duration_sec = Column(
        Float, nullable=True
    )  # Rolling average of task execution time
    last_performance_update = Column(
        DateTime, nullable=True
    )  # When metrics were last updated

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
    checkpoint_available = Column(
        Boolean, default=False
    )  # Whether checkpoint exists for recovery
    latest_checkpoint_data = Column(
        Text, nullable=True
    )  # Decoded JSON from latest delta_checkpoint_blob entry


class SchedulerConfigModel(Base):
    """Configuration for MCDM schedulers"""

    __tablename__ = "scheduler_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_name = Column(
        String, nullable=False, unique=True
    )  # "aras", "edas", "mabac", "wrr"
    is_active = Column(Boolean, default=False)  # Only one can be active at a time

    # Criteria configuration stored as JSON
    criteria_weights = Column(Text, nullable=True)  # JSON: [0.3, 0.2, 0.15, ...]
    criteria_names = Column(
        Text, nullable=True
    )  # JSON: ["cpu_cores", "ram_available", ...]
    criteria_types = Column(
        Text, nullable=True
    )  # JSON: [1, 1, -1, ...] (1=benefit, -1=cost)

    # Metadata
    description = Column(Text, nullable=True)  # Human-readable description
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    created_by = Column(String, nullable=True)  # User/admin who created config


class WorkerPerformanceHistoryModel(Base):
    """Historical performance snapshots for analytics"""

    __tablename__ = "worker_performance_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String, ForeignKey("workers.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)

    # Performance snapshot
    cpu_usage = Column(Float, nullable=True)
    ram_usage_mb = Column(Float, nullable=True)
    ram_available_mb = Column(Float, nullable=True)
    battery_level = Column(Float, nullable=True)
    is_charging = Column(Boolean, nullable=True)
    network_type = Column(String, nullable=True)

    # Task statistics at this point in time
    tasks_completed_snapshot = Column(Integer, default=0)
    tasks_failed_snapshot = Column(Integer, default=0)
    avg_task_duration = Column(Float, nullable=True)
