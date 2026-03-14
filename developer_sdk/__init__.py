"""
CrowdCompute - Distributed Python Computing SDK

Provides declarative checkpointing support for distributed task execution.

Usage:
    import developer_sdk as crowdio

    @crowdio.task(
        checkpoint=True,
        checkpoint_interval=5,
        checkpoint_state=["i", "result", "progress_percent"]
    )
    def my_task(state, data):
        # Task automatically gets state injection and checkpointing
        for i in range(state.get("i", 0), len(data)):
            state["i"] = i
            state["result"] = process(data[i])
            state["progress_percent"] = (i + 1) / len(data) * 100
        return state["result"]

    async def main():
        await crowdio.connect("localhost", 9000)
        results = await crowdio.map(my_task, data_list)
        await crowdio.disconnect()
"""

from .api import (
    connect,
    map,
    run,
    get,
    disconnect,
    submit,
    pipeline,
    task,
    TaskMetadata,
    TaskConfig,
    get_task_metadata,
    get_task_config,
    is_checkpoint_task,
    create_state_dict,
    Constant,
    crowdio,
)

__version__ = "0.2.0"
__all__ = [
    # Connection API
    "connect",
    "disconnect",
    # Execution API
    "map",
    "run",
    "get",
    "submit",
    "pipeline",
    # Declarative Checkpointing API
    "task",
    "TaskMetadata",
    "TaskConfig",
    "get_task_metadata",
    "get_task_config",
    "is_checkpoint_task",
    "create_state_dict",
    "Constant",
    "crowdio",
]
