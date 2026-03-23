import logging
from pathlib import Path

from .scheduler_interface import *

# Configure FIFO-specific logger
logger = logging.getLogger("fifo_scheduler")
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# File handler for FIFO decisions
file_handler = logging.FileHandler(log_dir / "fifo_decisions.log")
file_handler.setLevel(logging.DEBUG)

# Console handler for important info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Detailed formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers if not already added
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class FIFOScheduler(TaskScheduler):
    """First-In-First-Out scheduler - assigns tasks in order received"""
    
    def __init__(self):
        """Initialize FIFO scheduler"""
        logger.info(f"\n{'='*80}")
        logger.info(f"FIFO Scheduler Initialized")
        logger.info(f"{'='*80}")
        logger.info(f"Strategy: First-In-First-Out (simple queue-based)")
        logger.info(f"{'='*80}\n")
    
    async def select_worker(
        self, 
        task: Task, 
        available_workers: Set[str],
        all_workers: dict
    ) -> Optional[str]:
        """Select first available worker"""
        logger.info(f"\n{'='*80}")
        logger.info(f"WORKER SELECTION - Task ID: {task.id} (Job: {task.job_id})")
        logger.info(f"{'='*80}")
        logger.info(f"Available workers: {sorted(list(available_workers))}")
        logger.info(f"Total workers in system: {len(all_workers)}")
        
        if not available_workers:
            logger.warning("No available workers")
            logger.info(f"{'='*80}\n")
            return None
        
        selected_worker = next(iter(available_workers))
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"FIFO SELECTION (First Available):")
        logger.info(f"{'─'*80}")
        for i, worker_id in enumerate(sorted(available_workers), 1):
            marker = "→ SELECTED" if worker_id == selected_worker else "  available"
            logger.info(f"  [{i}] {worker_id} {marker}")
        logger.info(f"{'─'*80}")
        
        logger.info(f"\n✓ SELECTED: {selected_worker}")
        logger.info(f"{'='*80}\n")
        
        return selected_worker
    
    async def select_task(
        self,
        pending_tasks: List[Task],
        worker_id: str
    ) -> Optional[Task]:
        """Select first pending task"""
        if not pending_tasks:
            logger.debug(f"No pending tasks for worker {worker_id}")
            return None
        
        selected_task = pending_tasks[0]
        logger.debug(f"Selected task {selected_task.id} for worker {worker_id} (FIFO - first in queue)")
        
        return selected_task

