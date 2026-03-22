# developer_sdk

Python SDK for submitting distributed jobs to CrowdIO foreman/workers.

This package provides:
- async connection and job APIs
- function shipping and remote execution
- declarative task checkpointing via decorator metadata
- multi-stage pipeline and DNN-topology pipeline helpers
- tensor payload transport for intermediate DNN features
- mobile-safe path constants for runtime path injection

## Quick start

```python
import asyncio
import developer_sdk as crowdio


@crowdio.task(checkpoint=True, checkpoint_interval=5, checkpoint_state=["i", "acc"])
def square_task(x):
    # Worker receives plain function source; metadata is sent separately.
    return x * x


async def main():
    await crowdio.connect("localhost", 9000)
    try:
        results = await crowdio.map(square_task, [1, 2, 3, 4])
        print(results)
    finally:
        await crowdio.disconnect()


asyncio.run(main())
```

## Public API

Core async calls:
- connect(host, port=9000)
- disconnect()
- map(func, iterable, **kwargs)
- run(func, *args, **kwargs)
- submit(func, iterable, **kwargs)
- get(job_id, timeout=None)

Pipeline calls:
- pipeline(stages, dependency_map=None, **kwargs)
- dnn_pipeline(stages, inference_graph_id, topology_nodes, topology_edges, ...)

Intermediate feature routing:
- send_intermediate_feature(job_id, task_id, target_task_id, payload, source_worker_id="sdk-client")
- decode_intermediate_feature_payload(payload)

Declarative task API:
- task(...)
- TaskMetadata
- TaskConfig
- get_task_metadata(func)
- get_task_config(func)
- is_checkpoint_task(func)
- create_state_dict(checkpoint_state)
- crowdio namespace (decorator convenience)

Mobile path constants:
- Constant.FILE_DIR
- Constant.CACHE_DIR
- Constant.OUTPUT_DIR

Model/DNN helpers:
- build_partition_artifact(...)
- build_partition_artifacts(...)
- validate_topology(...)
- TopologyValidationError
- serialize_tensor(...)
- deserialize_tensor(...)

## Job patterns

### 1) Distributed map

```python
results = await crowdio.map(my_func, input_items)
```

### 2) Fire-and-fetch

```python
job_id = await crowdio.submit(my_func, input_items)
results = await crowdio.get(job_id, timeout=60)
```

### 3) Multi-stage pipeline

```python
stages = [
    {"func": stage1, "args_list": raw_items, "name": "preprocess"},
    {
        "func": stage2,
        "args_list": [None] * len(raw_items),
        "pass_upstream_results": True,
        "name": "compute",
    },
]
results = await crowdio.pipeline(stages)
```

## Checkpointing model

Use the task decorator to attach metadata. The SDK extracts that metadata and sends it with job submission.

```python
@crowdio.task(
    checkpoint=True,
    checkpoint_interval=10,
    checkpoint_state=["i", "partial_sum", "progress_percent"],
    retry_on_failure=True,
    max_retries=3,
)
def heavy_task(data):
    return sum(data)
```

Notes:
- checkpoint_state controls which variables are persisted (or all state when omitted).
- kwargs passed to map/submit can override decorator defaults (for example checkpoint_interval).

## Mobile path abstraction

Use Constant values in task configs instead of hardcoded device paths. Mobile runtimes can resolve these symbols to platform-specific paths.

```python
from developer_sdk import Constant

output_target = Constant.OUTPUT_DIR
cache_target = Constant.CACHE_DIR
```

## DNN feature transport

When payload includes numpy arrays, SDK can encode/decode tensors for transport:

```python
from developer_sdk import serialize_tensor, deserialize_tensor
```

For runtime message routing in DNN flows:

```python
await crowdio.send_intermediate_feature(
    job_id=job_id,
    task_id="stage-a:0",
    target_task_id="stage-b:0",
    payload=feature_tensor,
)
```

## Image utilities

The image_utils subpackage includes reusable helpers for distributed image workflows:
- split_image_into_tiles / split_image_into_grid / split_image_into_strips
- reassemble_tiles / reassemble_strips / merge_results
- apply_filter
- encode_image / decode_image
- load_image / save_image / get_image_info

## Notes and limitations

- APIs are async and require an event loop.
- Task functions are source-serialized; keep them import-safe and deterministic.
- dnn_pipeline validates topology and raises TopologyValidationError for invalid graphs.
