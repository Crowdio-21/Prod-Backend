from .client import CrowdComputeClient
from .decorators import (
    task,
    TaskMetadata,
    TaskConfig,
    get_task_metadata,
    get_task_config,
    is_checkpoint_task,
    create_state_dict,
    crowdio,
)
from .constants import Constant
from typing import Any, Callable, List, Optional, Dict

# Global client instance
_client = CrowdComputeClient()

# Re-export decorator for convenient import
__all__ = [
    "connect",
    "disconnect",
    "map",
    "run",
    "get",
    "submit",
    "pipeline",
    "task",
    "TaskMetadata",
    "TaskConfig",
    "get_task_metadata",
    "get_task_config",
    "is_checkpoint_task",
    "create_state_dict",
    "crowdio",
    "Constant",
]


# Public API functions
async def connect(host: str, port: int = 9000):
    """Connect to foreman server"""
    await _client.connect(host, port)


async def disconnect():
    """Disconnect from foreman server"""
    await _client.disconnect()


async def map(func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
    """
    Map function over iterable using distributed workers

    If function is decorated with @task, checkpoint metadata is automatically
    extracted and sent to workers for checkpoint-aware execution.

    Args:
        func: Function to execute (optionally decorated with @task)
        iterable: List of arguments to map over
        **kwargs: Additional options:
            - checkpoint: Override checkpoint setting (bool)
            - checkpoint_interval: Override checkpoint interval (float)

    Returns:
        List of results from all workers
    """
    return await _client.map(func, iterable, **kwargs)


async def run(func: Callable, *args, **kwargs) -> Any:
    """
    Run a single function with arguments on a worker

    If function is decorated with @task, checkpoint metadata is automatically
    used for checkpoint-aware execution.

    Args:
        func: Function to execute (optionally decorated with @task)
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the worker
    """
    return await _client.run(func, *args, **kwargs)


async def submit(func: Callable, iterable: List[Any], **kwargs) -> str:
    """
    Submit a job asynchronously without waiting for results

    Args:
        func: Function to execute (optionally decorated with @task)
        iterable: List of arguments for tasks
        **kwargs: Additional options

    Returns:
        job_id: Identifier to retrieve results later with get()
    """
    return await _client.submit(func, iterable, **kwargs)


async def get(job_id: str, timeout: Optional[float] = None) -> Any:
    """
    Get results for a specific job

    Args:
        job_id: Job identifier from submit()
        timeout: Maximum seconds to wait (None = wait forever)

    Returns:
        List of results or raises TimeoutError
    """
    return await _client.get_results(job_id, timeout)


async def pipeline(
    stages: List[Dict],
    dependency_map: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> List[Any]:
    """
    Execute a pipeline of dependent stages on distributed workers.

    Each stage defines a function and a list of arguments.  Tasks in later
    stages are automatically blocked until their upstream dependencies
    complete (tracked via a dependency counter).  When all upstream tasks
    finish, the downstream task's counter reaches zero and it becomes
    eligible for scheduling — just like how checkpointing is transparent
    to the user.

    Args:
        stages: Ordered list of stage dicts, each with:
            - func: Callable (optionally decorated with @task)
            - args_list: List of arguments (one per task in this stage)
            - pass_upstream_results: bool (default False) – if True,
              upstream results are injected into downstream args
            - name: Optional human-readable stage label
        dependency_map: Optional explicit task→[deps] mapping for
            arbitrary DAG topologies.  If omitted, stages form
            sequential barriers.
        **kwargs: Extra options (checkpoint, checkpoint_interval, …)

    Returns:
        List of results from the final stage.

    Example::

        results = await crowdio.pipeline([
            {"func": preprocess,  "args_list": raw_data},
            {"func": compute,     "args_list": [None]*len(raw_data),
             "pass_upstream_results": True},
            {"func": aggregate,   "args_list": [None],
             "pass_upstream_results": True},
        ])
    """
    return await _client.pipeline(stages, dependency_map=dependency_map, **kwargs)
