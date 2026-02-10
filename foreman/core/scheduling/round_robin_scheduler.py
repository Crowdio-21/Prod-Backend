from .scheduler_interface import *

class RoundRobinScheduler(TaskScheduler):
    """Round-robin scheduler - distributes tasks evenly across workers"""
    
    def __init__(self):
        self.last_worker_idx = -1
        self.worker_order = []
    
    async def select_worker(
        self, 
        task: Task, 
        available_workers: Set[str],
        all_workers: dict,
        ordered_available_workers: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select next worker in round-robin order"""
        if not available_workers:
            return None
        
        # Update worker order if changed
        available_list = (
            list(ordered_available_workers)
            if ordered_available_workers is not None
            else sorted(available_workers)
        )
        if available_list != self.worker_order:
            self.worker_order = available_list
            self.last_worker_idx = -1
        
        # Select next worker
        self.last_worker_idx = (self.last_worker_idx + 1) % len(self.worker_order)
        return self.worker_order[self.last_worker_idx]
    
    async def select_task(
        self,
        pending_tasks: List[Task],
        worker_id: str
    ) -> Optional[Task]:
        """Select first pending task"""
        return pending_tasks[0] if pending_tasks else None
