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
from .model_artifacts import build_partition_artifact, build_partition_artifacts
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
    "dnn_pipeline",
    "send_intermediate_feature",
    "decode_intermediate_feature_payload",
    "build_partition_artifact",
    "build_partition_artifacts",
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


async def dnn_pipeline(
    stages: List[Dict],
    inference_graph_id: str,
    topology_nodes: List[Dict],
    topology_edges: List[Dict],
    model_version_id: Optional[str] = None,
    model_artifacts: Optional[List[Dict]] = None,
    aggregation_strategy: str = "average",
    dependency_map: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> List[Any]:
    """
    Execute a topology-aware DNN pipeline on distributed workers.

    This is the DNN-oriented variant of pipeline() and includes explicit
    graph/topology metadata consumed by the foreman.
    """
    return await _client.dnn_pipeline(
        stages=stages,
        inference_graph_id=inference_graph_id,
        topology_nodes=topology_nodes,
        topology_edges=topology_edges,
        model_version_id=model_version_id,
        model_artifacts=model_artifacts,
        aggregation_strategy=aggregation_strategy,
        dependency_map=dependency_map,
        **kwargs,
    )


async def send_intermediate_feature(
    job_id: str,
    task_id: str,
    target_task_id: str,
    payload: Any,
    source_worker_id: str = "sdk-client",
) -> None:
    """Send an intermediate feature payload to foreman for DNN routing."""
    await _client.send_intermediate_feature(
        job_id=job_id,
        task_id=task_id,
        target_task_id=target_task_id,
        payload=payload,
        source_worker_id=source_worker_id,
    )


def decode_intermediate_feature_payload(payload: Any) -> Any:
    """Decode SDK tensor-aware payloads into original objects/tensors."""
    return _client.decode_intermediate_feature_payload(payload)
