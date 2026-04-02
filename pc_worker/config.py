"""
Configuration models for the PC worker service.
"""

from pydantic import BaseModel


class WorkerConfig(BaseModel):
    """Runtime configuration for a worker instance."""

    worker_id: str
    foreman_url: str = "ws://localhost:9000"
    max_concurrent_tasks: int = 1
    auto_restart: bool = True
    heartbeat_interval: int = 30
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    model_cache_dir: str = ".worker_model_cache"
