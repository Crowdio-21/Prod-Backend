"""
Task wrapper for checkpoint-aware execution on workers

Provides automatic state management, checkpoint variable tracking,
and transparent resume support for declarative checkpointing.

The wrapper:
1. Injects a state dictionary into task execution
2. Tracks declared checkpoint variables
3. Provides state to the CheckpointHandler
4. Handles resume from checkpoint transparently
"""

import asyncio
import builtins
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class TaskExecutionContext:
    """
    Context for a single task execution
    
    Maintains state and metadata during task execution for
    checkpoint handling and recovery.
    """
    task_id: str
    job_id: str
    
    # Checkpoint configuration
    checkpoint_enabled: bool = False
    checkpoint_interval: float = 10.0
    checkpoint_state_vars: List[str] = field(default_factory=list)
    
    # State management
    state: Dict[str, Any] = field(default_factory=dict)
    initial_state: Dict[str, Any] = field(default_factory=dict)
    
    # Resume information
    is_resumed: bool = False
    resume_progress: float = 0.0
    resume_checkpoint_count: int = 0
    
    # Execution tracking
    started_at: Optional[datetime] = None
    progress_percent: float = 0.0
    status: str = "pending"
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_state(self, key: str, value: Any) -> None:
        """Thread-safe state update"""
        with self._lock:
            self.state[key] = value
    
    def get_state(self) -> Dict[str, Any]:
        """Thread-safe state retrieval"""
        with self._lock:
            return dict(self.state)
    
    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get state filtered to only declared checkpoint variables
        
        Returns:
            Dictionary with only the declared checkpoint variables
        """
        with self._lock:
            if not self.checkpoint_state_vars:
                # No specific variables declared - checkpoint all state
                return dict(self.state)
            
            return {
                key: self.state[key]
                for key in self.checkpoint_state_vars
                if key in self.state
            }
    
    def set_progress(self, percent: float) -> None:
        """Update progress percentage"""
        with self._lock:
            self.progress_percent = min(100.0, max(0.0, percent))
            self.state["progress_percent"] = self.progress_percent
    
    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """
        Create checkpoint dictionary with state and metadata
        
        Returns:
            Dictionary suitable for checkpointing
        """
        with self._lock:
            checkpoint = self.get_checkpoint_state()
            checkpoint["_meta"] = {
                "task_id": self.task_id,
                "job_id": self.job_id,
                "progress_percent": self.progress_percent,
                "is_resumed": self.is_resumed,
                "checkpoint_time": datetime.now().isoformat()
            }
            return checkpoint


class TaskWrapper:
    """
    Wraps task execution with checkpoint support
    
    Handles:
    - State injection into task functions
    - Resume from checkpoint
    - Progress tracking
    - State validation
    """
    
    def __init__(self):
        # Track active execution contexts by task_id
        self._contexts: Dict[str, TaskExecutionContext] = {}
        self._lock = threading.Lock()
    
    def create_context(
        self,
        task_id: str,
        job_id: str,
        task_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_state: Optional[Dict[str, Any]] = None
    ) -> TaskExecutionContext:
        """
        Create execution context for a new task
        
        Args:
            task_id: Task identifier
            job_id: Job identifier
            task_metadata: Checkpoint configuration from @task decorator
            checkpoint_state: Pre-loaded state if resuming
            
        Returns:
            TaskExecutionContext for this execution
        """
        metadata = task_metadata or {}
        
        # Determine if resuming
        is_resumed = checkpoint_state is not None and len(checkpoint_state) > 0
        
        # Extract resume progress if available
        resume_progress = 0.0
        resume_checkpoint_count = 0
        if is_resumed:
            resume_progress = checkpoint_state.get("progress_percent", 0.0)
            meta = checkpoint_state.get("_meta", {})
            resume_progress = meta.get("progress_percent", resume_progress)
        
        context = TaskExecutionContext(
            task_id=task_id,
            job_id=job_id,
            checkpoint_enabled=metadata.get("checkpoint_enabled", False),
            checkpoint_interval=metadata.get("checkpoint_interval", 10.0),
            checkpoint_state_vars=metadata.get("checkpoint_state", []),
            state=dict(checkpoint_state) if checkpoint_state else {},
            initial_state=dict(checkpoint_state) if checkpoint_state else {},
            is_resumed=is_resumed,
            resume_progress=resume_progress,
            resume_checkpoint_count=resume_checkpoint_count,
            started_at=datetime.now(),
            progress_percent=resume_progress,
            status="running"
        )
        
        # Add resume flag to state for task function access
        if is_resumed:
            context.state["_is_resumed"] = True
        
        # Store context
        with self._lock:
            self._contexts[task_id] = context
        
        return context
    
    def get_context(self, task_id: str) -> Optional[TaskExecutionContext]:
        """Get execution context for a task"""
        with self._lock:
            return self._contexts.get(task_id)
    
    def remove_context(self, task_id: str) -> None:
        """Remove execution context when task completes"""
        with self._lock:
            if task_id in self._contexts:
                del self._contexts[task_id]
    
    def setup_global_state(self, context: TaskExecutionContext) -> None:
        """
        Set up global state access for the task function
        
        Makes state accessible via builtins._checkpoint_state so
        deserialized functions can access and update it.
        """
        builtins._checkpoint_state = context.state
        builtins._checkpoint_context = context
    
    def cleanup_global_state(self) -> None:
        """Clean up global state after task execution"""
        if hasattr(builtins, '_checkpoint_state'):
            delattr(builtins, '_checkpoint_state')
        if hasattr(builtins, '_checkpoint_context'):
            delattr(builtins, '_checkpoint_context')
    
    def get_state_callback(self, task_id: str) -> Callable[[], Dict[str, Any]]:
        """
        Create state getter callback for CheckpointHandler
        
        Returns:
            Callable that returns current checkpoint state
        """
        def get_state() -> Dict[str, Any]:
            context = self.get_context(task_id)
            if context:
                return context.to_checkpoint_dict()
            
            # Fallback to builtins
            if hasattr(builtins, '_checkpoint_state'):
                state = builtins._checkpoint_state
                if isinstance(state, dict):
                    return dict(state)
            
            return {"progress_percent": 0.0}
        
        return get_state
    
    def validate_checkpoint_variables(
        self,
        context: TaskExecutionContext,
        state: Dict[str, Any]
    ) -> bool:
        """
        Validate that declared checkpoint variables exist in state
        
        Args:
            context: Execution context
            state: Current state dictionary
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError if validation fails
        """
        if not context.checkpoint_enabled:
            return True
        
        if not context.checkpoint_state_vars:
            # No specific variables declared - all state is valid
            return True
        
        missing = set(context.checkpoint_state_vars) - set(state.keys())
        if missing:
            raise ValueError(
                f"Checkpoint validation failed for task {context.task_id}: "
                f"Declared variables not found in state: {missing}"
            )
        
        return True


# Global task wrapper instance
_task_wrapper = TaskWrapper()


def get_task_wrapper() -> TaskWrapper:
    """Get the global TaskWrapper instance"""
    return _task_wrapper


def create_wrapped_executor(
    func: Callable,
    task_id: str,
    job_id: str,
    task_args: List[Any],
    task_metadata: Optional[Dict[str, Any]] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create a wrapped executor function for checkpoint-aware execution
    
    The wrapped function:
    1. Sets up execution context
    2. Injects state into function if it accepts 'state' parameter
    3. Tracks progress and state changes
    4. Cleans up after execution
    
    Args:
        func: Original task function
        task_id: Task identifier
        job_id: Job identifier
        task_args: Arguments for the task
        task_metadata: Checkpoint configuration
        checkpoint_state: Pre-loaded state if resuming
        
    Returns:
        Wrapped callable ready for execution
    """
    wrapper = get_task_wrapper()
    
    def wrapped_executor() -> Any:
        """Execute task with checkpoint support"""
        # Create execution context
        context = wrapper.create_context(
            task_id=task_id,
            job_id=job_id,
            task_metadata=task_metadata,
            checkpoint_state=checkpoint_state
        )
        
        # Set up global state access
        wrapper.setup_global_state(context)
        
        try:
            # Check if function signature includes 'state' parameter
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            if 'state' in params:
                # Inject state as first argument
                if isinstance(task_args, list) and len(task_args) == 1:
                    result = func(context.state, task_args[0])
                elif isinstance(task_args, list):
                    result = func(context.state, *task_args)
                else:
                    result = func(context.state, task_args)
            else:
                # Call function normally, it will access state via builtins
                if isinstance(task_args, list) and len(task_args) == 1:
                    result = func(task_args[0])
                elif isinstance(task_args, list):
                    result = func(*task_args)
                else:
                    result = func(task_args)
            
            # Mark completed
            context.status = "completed"
            context.set_progress(100.0)
            
            return result
            
        except Exception as e:
            context.status = "failed"
            raise
        finally:
            # Clean up
            wrapper.cleanup_global_state()
            wrapper.remove_context(task_id)
    
    return wrapped_executor


def create_resumed_executor(
    func: Callable,
    task_id: str,
    job_id: str,
    checkpoint_state: Dict[str, Any],
    task_metadata: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create an executor for a resumed task
    
    Similar to create_wrapped_executor but specifically for resumed tasks,
    with the checkpoint state pre-loaded and resume flag set.
    
    Args:
        func: Original task function
        task_id: Task identifier
        job_id: Job identifier
        checkpoint_state: Reconstructed state from checkpoints
        task_metadata: Checkpoint configuration
        
    Returns:
        Wrapped callable ready for resumed execution
    """
    wrapper = get_task_wrapper()
    
    def resumed_executor() -> Any:
        """Execute resumed task from checkpoint state"""
        # Add resume flag
        state_with_flag = dict(checkpoint_state)
        state_with_flag["_is_resumed"] = True
        
        # Create context with checkpoint state
        context = wrapper.create_context(
            task_id=task_id,
            job_id=job_id,
            task_metadata=task_metadata,
            checkpoint_state=state_with_flag
        )
        
        # Log resume info
        progress = checkpoint_state.get("progress_percent", 0)
        print(f"[TaskWrapper] Resuming task {task_id} from {progress:.1f}% progress")
        
        # Set up global state access
        wrapper.setup_global_state(context)
        
        try:
            # Check function signature
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # For resumed tasks, we typically call with single arg (the original input)
            # The function should check _is_resumed and use checkpoint state
            if 'state' in params:
                # State-aware function
                result = func(context.state)
            else:
                # Function accesses state via builtins
                result = func()
            
            context.status = "completed"
            context.set_progress(100.0)
            
            return result
            
        except Exception as e:
            context.status = "failed"
            raise
        finally:
            wrapper.cleanup_global_state()
            wrapper.remove_context(task_id)
    
    return resumed_executor
