from .scheduler_interface import *

class PriorityScheduler(TaskScheduler):
    """Priority-based scheduler - assigns high priority tasks first"""
    
    async def select_worker(
        self, 
        task: Task, 
        available_workers: Set[str],
        all_workers: dict,
        ordered_available_workers: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select worker with best success rate for high priority tasks"""
        if not available_workers:
            return None
        
        # For high priority tasks, use performance-based selection
        if task.priority > 0:
            workers = [
                all_workers.get(wid) 
                for wid in available_workers 
                if wid in all_workers
            ]
            
            if workers:
                best_worker = max(workers, key=lambda w: w.success_rate)
                return best_worker.id
        
        # For normal priority, use FIFO
        if ordered_available_workers:
            return ordered_available_workers[0]
        return next(iter(available_workers))
    
    async def select_task(
        self,
        pending_tasks: List[Task],
        worker_id: str
    ) -> Optional[Task]:
        """Select task with highest priority"""
        if not pending_tasks:
            return None
        
        return max(pending_tasks, key=lambda t: (t.priority, -t.retry_count))
