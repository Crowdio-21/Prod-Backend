# developer_sdk

Python SDK for submitting distributed jobs to CrowdIO foreman/workers.

This package provides:
- async connection and job APIs
- function shipping and remote execution
- declarative task checkpointing via decorator metadata
- multi-stage pipeline and DNN-topology pipeline helpers
- tensor payload transport for intermediate DNN features
- mobile-safe path constants for runtime path injection

## Start here: connect -> map -> disconnect

If you only need the core flow, start with these three calls:
- `connect(host, port)`
- `map(func, iterable)`
- `disconnect()`

Example (adapted from `tests/example_client.py`):

```python
import asyncio
from developer_sdk import connect, map, disconnect


def process_data(data):
    return sum(data) * 2


async def main():
    await connect("localhost", 9000)
    try:
        data_arrays = [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [100, 200, 300, 400, 500],
            [1000, 2000, 3000, 4000, 5000],
        ]
        processed_results = await map(process_data, data_arrays)
        print(processed_results)
    finally:
        await disconnect()


asyncio.run(main())
```

## Checkpointing with Monte Carlo example

After you understand connect/map/disconnect, add checkpointing with the task decorator.
This example is adapted from tests/montecarlo/monte_carlo_euler_client.py.

```python
import asyncio
from developer_sdk import connect, map as distributed_map, disconnect, crowdio


@crowdio.task(
    checkpoint=True,
    checkpoint_interval=2.0,
    checkpoint_state=["trials_completed", "total_count", "estimated_e", "progress_percent"],
)
def monte_carlo_euler_worker(num_trials):
    import random

    trials_completed = 0
    total_count = 0
    estimated_e = 0.0
    progress_percent = 0.0

    for i in range(num_trials):
        random_sum = 0.0
        count = 0
        while random_sum < 1.0:
            random_sum += random.random()
            count += 1

        total_count += count
        trials_completed = i + 1
        progress_percent = (trials_completed / num_trials) * 100
        estimated_e = total_count / trials_completed

    return {
        "num_trials": num_trials,
        "total_count": total_count,
        "estimated_e": round(estimated_e, 6),
        "status": "success",
    }


async def main():
    await connect("localhost", 9000)
    try:
        # Split total trials across task batches
        task_inputs = [250000, 250000, 250000, 250000]
        results = await distributed_map(monte_carlo_euler_worker, task_inputs)
        print(results)
    finally:
        await disconnect()


asyncio.run(main())
```

Notes:
- checkpoint_state lists the variables the runtime persists and restores.
- Keep worker logic pure; the framework handles resume from saved checkpoint state.
- You can still pass map/submit kwargs to override decorator defaults when needed.

## Pipeline job with image processing example

After checkpointing, the next pattern is a multi-stage pipeline where each stage depends on the previous one.
This example is adapted from tests/image_processing/pipeline_image_processing.py.

```python
import asyncio
import base64
from developer_sdk import connect, disconnect, crowdio, pipeline


@crowdio.task()
def preprocess_image(image_input):
    # Stage 0: decode input image and emit tile payloads
    from PIL import Image
    import io

    image_bytes = base64.b64decode(image_input["image_b64"])
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    tile_size = image_input.get("tile_size", 200)

    tiles = []
    tile_id = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = img.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
            buf = io.BytesIO()
            tile.save(buf, format="PNG")
            tiles.append({
                "tile_id": tile_id,
                "position": [x, y],
                "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
            })
            tile_id += 1

    return {
        "image_id": image_input["image_id"],
        "original_size": [width, height],
        "filter_type": image_input["filter_type"],
        "tiles": tiles,
    }


@crowdio.task()
def process_tiles(task_input):
    # Stage 1: receives all Stage-0 outputs via pass_upstream_results
    from PIL import Image, ImageFilter
    import io

    upstream = task_input.get("upstream_results", {})
    if not upstream:
        return {"status": "error", "error": "No upstream preprocess results"}

    merged_tiles = []
    image_id = None
    original_size = None

    for preprocess_result in upstream.values():
        image_id = preprocess_result["image_id"]
        original_size = preprocess_result["original_size"]
        filter_type = preprocess_result.get("filter_type", "sharpen")

        for tile in preprocess_result["tiles"]:
            tile_img = Image.open(io.BytesIO(base64.b64decode(tile["image"])))
            filtered = tile_img.filter(ImageFilter.SHARPEN if filter_type == "sharpen" else ImageFilter.SMOOTH)

            buf = io.BytesIO()
            filtered.save(buf, format="PNG")
            merged_tiles.append({
                "tile_id": tile["tile_id"],
                "position": tile["position"],
                "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
            })

    return {
        "image_id": image_id,
        "original_size": original_size,
        "filter_type": filter_type,
        "processed_tiles": merged_tiles,
    }


@crowdio.task()
def postprocess_image(task_input):
    # Stage 2: stitch processed tiles into one final image
    from PIL import Image
    import io

    upstream = task_input.get("upstream_results", {})
    if not upstream:
        return {"status": "error", "error": "No upstream process results"}

    first = next(iter(upstream.values()))
    width, height = first["original_size"]
    canvas = Image.new("RGB", (width, height))

    total_tiles = 0
    for process_result in upstream.values():
        for tile in process_result.get("processed_tiles", []):
            tile_img = Image.open(io.BytesIO(base64.b64decode(tile["image"])))
            canvas.paste(tile_img, tuple(tile["position"]))
            total_tiles += 1

    out = io.BytesIO()
    canvas.save(out, format="PNG")

    return {
        "status": "success",
        "image_id": first["image_id"],
        "filter_applied": first["filter_type"],
        "total_tiles": total_tiles,
        "result_image_b64": base64.b64encode(out.getvalue()).decode("utf-8"),
    }


async def main():
    await connect("localhost", 9000)
    try:
        # In practice, load this from a file and base64-encode it.
        stage_0_input = {
            "image_id": 0,
            "filter_type": "sharpen",
            "tile_size": 200,
            "image_b64": "<base64-encoded-input-image>",
        }

        results = await pipeline([
            {
                "func": preprocess_image,
                "args_list": [stage_0_input],
                "name": "preprocess",
            },
            {
                "func": process_tiles,
                "args_list": [None],
                "pass_upstream_results": True,
                "name": "process",
            },
            {
                "func": postprocess_image,
                "args_list": [None],
                "pass_upstream_results": True,
                "name": "postprocess",
            },
        ])

        final = results[0]
        print(f"Pipeline done: {final['status']} | tiles={final['total_tiles']}")
    finally:
        await disconnect()


asyncio.run(main())
```

Why pipeline here:
    - Stage 1 receives Stage 0 outputs through upstream_results, so no manual wiring is needed.
    - Stage 2 gets all processed tiles and returns one final artifact.
    - The caller receives final-stage outputs only, which keeps orchestration clean for end-to-end jobs.

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


Declarative task API:
- task(...)
- TaskMetadata
- TaskConfig
- get_task_metadata(func)
- get_task_config(func)
- is_checkpoint_task(func)
- create_state_dict(checkpoint_state)
- crowdio namespace (decorator convenience, defined in namespace.py)

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

## Mobile path abstraction

Use Constant values in task configs instead of hardcoded device paths. Mobile runtimes can resolve these symbols to platform-specific paths.

```python
import asyncio
from developer_sdk import connect, disconnect, crowdio, map as distributed_map


@crowdio.task()
def process_images_on_device(config):
    import builtins
    import os

    # Key helper: convert @CROWDIO:* aliases into real runtime paths.
    # Example: @CROWDIO:FILE_DIR -> /storage/emulated/0/MyPickedFolder
    def resolve_path_alias(value):
        # Non-alias values pass through unchanged.
        if not isinstance(value, str) or not value.startswith("@CROWDIO:"):
            return value
        alias_map = getattr(builtins, "_crowdio_path_aliases", {})
        return alias_map.get(value, value)

    image_dir = resolve_path_alias(config.get("image_dir"))
    if isinstance(image_dir, str) and image_dir.startswith("@CROWDIO:"):
        return {
            "processed": 0,
            "errors": [
                "Path alias was not resolved on worker. "
                "Ensure mobile runtime injects builtins._crowdio_path_aliases."
            ],
        }

    if not image_dir or not os.path.isdir(image_dir):
        return {
            "processed": 0,
            "errors": [f"Image directory not found: {image_dir}"],
        }

    # Process files from the resolved real path...
    return {"processed": 1, "errors": [], "image_dir": image_dir}


async def main():
    await connect("localhost", 9000)
    try:
        task_config = {
            # Readable developer config: no hardcoded Android/iOS paths.
            "image_dir": crowdio.Constant.FILE_DIR,
            "filter": "sharpen",
            "max_images": 10,
        }
        result = await distributed_map(process_images_on_device, [task_config])
        print(result)
    finally:
        await disconnect()


asyncio.run(main())
```

Expected runtime behavior on mobile workers:
- Runtime injects alias mapping into `builtins._crowdio_path_aliases`, for example:
  - `{"@CROWDIO:FILE_DIR": "/storage/emulated/0/MyPickedFolder"}`
- Worker resolves `@CROWDIO:*` values before file I/O.
- Developer code remains portable and path-safe across devices.

## Image utilities

The image_utils subpackage includes reusable helpers for distributed image workflows:
- split_image_into_tiles / split_image_into_grid / split_image_into_strips
- reassemble_tiles / reassemble_strips / merge_results
- apply_filter
- encode_image / decode_image
- load_image / save_image / get_image_info

Example: split -> process -> reassemble

Important:
- For remote/mobile workers, put task-specific imports inside the task function.
- CrowdIO sends function source for execution, so local module scope imports may not exist on the worker runtime.

```python
import asyncio
from developer_sdk import connect, disconnect, crowdio
from developer_sdk.image_utils import (
    load_image,
    save_image,
    get_image_info,
    split_image_into_tiles,
    reassemble_tiles,
)


@crowdio.task(checkpoint=True, checkpoint_interval=3.0, checkpoint_state=["progress"])
def process_tile(tile_data):
    # tile_data["image"] is base64 PNG from split_image_into_tiles
    from developer_sdk.image_utils import apply_filter
    progress = 100.0
    filtered = apply_filter(tile_data["image"], filter_type=tile_data.get("filter_type", "sharpen"))
    return {
        "tile_id": tile_data["tile_id"],
        "image": filtered,
        "position": tile_data["position"],
        "size": tile_data["size"],
    }


async def main():
    await connect("localhost", 9000)
    try:
        image = load_image("image.png")
        print(get_image_info(image))

        tiles = split_image_into_tiles(image, tile_size=200)
        tile_inputs = [{**t, "filter_type": "sharpen"} for t in tiles]

        processed_tiles = await process_tile.map(tile_inputs)
        result_image = reassemble_tiles(processed_tiles, image.size)
        save_image(result_image, "output/processed_sharpen.png")
    finally:
        await disconnect()


asyncio.run(main())
```

## Notes and limitations

- APIs are async and require an event loop.
- Task functions are source-serialized; keep them import-safe and deterministic.
- If a task depends on optional/runtime modules (for example image helpers), import them inside the task body.
- dnn_pipeline validates topology and raises TopologyValidationError for invalid graphs.
