from .scheduler_interface import *

class PerformanceBasedScheduler(TaskScheduler):
    """Scheduler that prioritizes workers with better performance"""
    
    async def select_worker(
        self, 
        task: Task, 
        available_workers: Set[str],
        all_workers: dict,
        ordered_available_workers: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select worker with best success rate"""
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
        
        # Sort by success rate (descending) and total tasks (descending)
        best_worker = max(
            workers,
            key=lambda w: (w.success_rate, w.tasks_completed)
        )
        
        return best_worker.id
    
    async def select_task(
        self,
        pending_tasks: List[Task],
        worker_id: str
    ) -> Optional[Task]:
        """Select task with highest priority, then by retry count"""
        if not pending_tasks:
            return None
        
        return max(
            pending_tasks,
            key=lambda t: (t.priority, -t.retry_count)
        )

