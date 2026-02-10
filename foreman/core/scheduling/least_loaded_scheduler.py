from .scheduler_interface import *

class LeastLoadedScheduler(TaskScheduler):
    """Scheduler that assigns tasks to least loaded workers"""
    
    async def select_worker(
        self, 
        task: Task, 
        available_workers: Set[str],
        all_workers: dict,
        ordered_available_workers: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select worker with fewest total tasks completed"""
        if not available_workers:
            return None
        
        # Get worker objects
        workers = [
            all_workers.get(wid) 
            for wid in available_workers 
            if wid in all_workers
        ]
        
        if not workers:
            if ordered_available_workers:
                return ordered_available_workers[0]
            return next(iter(available_workers))
        
        # Select worker with minimum total tasks
        least_loaded = min(
            workers,
            key=lambda w: w.tasks_completed + w.tasks_failed
        )
        
        return least_loaded.id
    
    async def select_task(
        self,
        pending_tasks: List[Task],
        worker_id: str
    ) -> Optional[Task]:
        """Select first pending task"""
        return pending_tasks[0] if pending_tasks else None
