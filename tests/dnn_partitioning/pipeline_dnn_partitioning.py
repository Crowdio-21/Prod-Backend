#!/usr/bin/env python3
"""
Distributed DNN Partitioning – Dependency Counter Pipeline
==========================================================

This test exercises the **dependency counter / pipeline execution** system
using a distributed DNN inference workload.  The model graph (DAG) is
partitioned across multiple crowd-workers so that:

  * Linear chains execute sequentially (layer N blocks on layer N-1).
  * **Skip connections** (ResNet-style) are handled by giving the join
    layer a dependency_count of 2 — one for the main branch, one for
    the skip branch.
  * **Parallel branches** (Inception-style) fan-out to different workers
    simultaneously; the concatenation layer waits for ALL branches via
    its dependency counter.

Architecture (Kotlin ↔ Python bridge for Chaquopy):
  ● Kotlin side  – orchestrates the DAG, manages AtomicInteger counters,
                   and communicates tensors between devices via Ktor sockets.
  ● Python side  – this file.  Each stage/layer slice runs a sub-model
                   (TFLite or Keras) to produce the activation tensor that
                   the next layer consumes.

Stages in this pipeline:
    Stage 0  (partition_model)   – Load the full model, build the partition
                                   plan (DAG → device map), and serialise
                                   sub-model weights for each partition.
    Stage 1  (run_layer_slice)   – Execute a partition slice on a worker.
                                   Multiple tasks may run in parallel for
                                   independent branches.
    Stage 2  (fuse_branches)     – Join / concatenate outputs from parallel
                                   branches or skip connections.  This stage
                                   is **blocked** until ALL upstream slices
                                   complete (dependency_count = branch count).
    Stage 3  (classify)          – Run the final classification head
                                   (softmax / dense) on the fused feature map.

Usage:
    # Start foreman + at least one worker first, then:
    uv run python tests/dnn_partitioning/pipeline_dnn_partitioning.py

    # With options:
    uv run python tests/dnn_partitioning/pipeline_dnn_partitioning.py \\
        --model resnet --input-size 224 --num-partitions 3 --output ./dnn_output

    # Inception-style parallel branches:
    uv run python tests/dnn_partitioning/pipeline_dnn_partitioning.py \\
        --model inception --input-size 299 --num-partitions 4

    # Local single-device mode (no foreman/workers needed):
    uv run python tests/dnn_partitioning/pipeline_dnn_partitioning.py \\
        --model resnet --local
"""

import sys
import os
import asyncio
import time
import argparse
import json
import math

# ---------------------------------------------------------------------------
# Project root on sys.path so developer_sdk resolves
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import connect, disconnect, crowdio, pipeline


# =====================================================================
# Helpers – lightweight tensor / model utilities
# =====================================================================

def _numpy_available():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _create_random_input(shape, dtype="float32"):
    """Create a random input tensor (numpy array or plain list)."""
    if _numpy_available():
        import numpy as np
        return np.random.randn(*shape).astype(dtype)
    # Fallback: nested lists
    import random
    random.seed(42)
    total = 1
    for s in shape:
        total *= s
    flat = [random.gauss(0, 1) for _ in range(total)]
    return flat  # consumer must reshape


def _serialize_tensor(tensor):
    """Serialize a tensor to a portable dict (base64-encoded bytes)."""
    import base64
    import io
    if _numpy_available():
        import numpy as np
        buf = io.BytesIO()
        np.save(buf, np.asarray(tensor))
        return {
            "format": "npy",
            "dtype": str(np.asarray(tensor).dtype),
            "shape": list(np.asarray(tensor).shape),
            "data_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        }
    # Fallback: JSON
    return {"format": "json", "data": tensor}


def _deserialize_tensor(blob):
    """Reconstruct a tensor from the dict produced by _serialize_tensor."""
    import base64
    import io
    fmt = blob.get("format", "json")
    if fmt == "npy" and _numpy_available():
        import numpy as np
        raw = base64.b64decode(blob["data_b64"])
        buf = io.BytesIO(raw)
        return np.load(buf)
    return blob.get("data", blob)


# =====================================================================
# DAG builder – describes how layers relate to one another
# =====================================================================

def build_resnet_dag(num_partitions=3, input_size=224):
    """
    Build a simplified ResNet-like DAG.

    The graph has a main chain with one skip connection:

        [input] → partition_0  → partition_1  →  fuse (skip join) → classify
                  ╰──────────── skip branch ──────╯

    The *fuse* node has dependency_count = 2 (main-branch + skip).

    Returns:
        layers  – list of layer descriptors
        edges   – list of (src_idx, dst_idx) edges
    """
    layers = []
    edges = []

    stem_spatial = input_size // 2  # stride-2 conv

    # Partition 0: conv block (stem)
    layers.append({
        "layer_id": "partition_0",
        "op": "conv_block",
        "params": {"filters": 64, "kernel": 7, "stride": 2},
        "output_spatial": stem_spatial,
        "partition_idx": 0,
    })

    # Partition 1: residual block (main branch continues from stem output)
    layers.append({
        "layer_id": "partition_1",
        "op": "residual_block",
        "params": {"filters": 64, "blocks": 2},
        "output_spatial": stem_spatial,
        "partition_idx": 1,
    })

    # Skip branch -- identity / 1x1 conv to match dimensions
    layers.append({
        "layer_id": "skip_branch",
        "op": "skip_connection",
        "params": {"project": True, "filters": 64},
        "output_spatial": stem_spatial,
        "partition_idx": 0,
    })

    # Fuse (join) -- element-wise add of main + skip
    layers.append({
        "layer_id": "fuse",
        "op": "add",
        "params": {},
        "output_spatial": stem_spatial,
        "dependency_count": 2,
        "partition_idx": min(num_partitions - 1, 2),
    })

    # Classify -- global average pool + dense + softmax
    layers.append({
        "layer_id": "classify",
        "op": "classify",
        "params": {"num_classes": 1000},
        "output_spatial": 1,
        "partition_idx": min(num_partitions - 1, 2),
    })

    # Edges: partition_0 → partition_1 → fuse → classify
    #         partition_0 → skip_branch  → fuse
    edges = [
        ("partition_0", "partition_1"),
        ("partition_0", "skip_branch"),
        ("partition_1", "fuse"),
        ("skip_branch", "fuse"),
        ("fuse", "classify"),
    ]

    return layers, edges


def build_inception_dag(num_partitions=4, input_size=224):
    """
    Build a simplified Inception-like DAG with parallel branches.

        [input] -> stem -> +-- branch_1x1 --+
                           +-- branch_3x3 --+
                           +-- branch_5x5 --+
                           '-- branch_pool --+
                                             '-> concat -> classify

    The *concat* node has dependency_count = 4.
    """
    stem_spatial = input_size // 2  # stride-2 stem

    layers = [
        {"layer_id": "stem", "op": "conv_block",
         "params": {"filters": 64, "kernel": 7, "stride": 2},
         "output_spatial": stem_spatial, "partition_idx": 0},
        {"layer_id": "branch_1x1", "op": "conv_block",
         "params": {"filters": 64, "kernel": 1, "stride": 1},
         "output_spatial": stem_spatial, "partition_idx": 0},
        {"layer_id": "branch_3x3", "op": "conv_block",
         "params": {"filters": 128, "kernel": 3, "stride": 1},
         "output_spatial": stem_spatial, "partition_idx": 1},
        {"layer_id": "branch_5x5", "op": "conv_block",
         "params": {"filters": 32, "kernel": 5, "stride": 1},
         "output_spatial": stem_spatial, "partition_idx": 2},
        {"layer_id": "branch_pool", "op": "maxpool_conv",
         "params": {"pool": 3, "filters": 32, "kernel": 1},
         "output_spatial": stem_spatial, "partition_idx": 3 % num_partitions},
        {"layer_id": "concat", "op": "concat",
         "params": {}, "output_spatial": stem_spatial,
         "dependency_count": 4, "partition_idx": 0},
        {"layer_id": "classify", "op": "classify",
         "params": {"num_classes": 1000},
         "output_spatial": 1, "partition_idx": 0},
    ]

    edges = [
        ("stem", "branch_1x1"),
        ("stem", "branch_3x3"),
        ("stem", "branch_5x5"),
        ("stem", "branch_pool"),
        ("branch_1x1", "concat"),
        ("branch_3x3", "concat"),
        ("branch_5x5", "concat"),
        ("branch_pool", "concat"),
        ("concat", "classify"),
    ]

    return layers, edges


# =====================================================================
# Stage 0 – Partition the model and emit the execution plan
# =====================================================================

@crowdio.task()
def partition_model(config):
    """
    Analyse the model graph, build a DAG of layer slices, and produce a
    partition plan mapping each slice to a worker partition index.

    This stage runs once.  Its output is consumed by Stage 1 tasks.

    Args:
        config (dict):
            model       – "resnet" | "inception"
            input_size  – spatial dimension (e.g. 224)
            num_partitions – how many devices / partitions
            dtype       – "float32" (default)

    Returns:
        dict with:
            model, input_size, num_partitions, dtype,
            layers     – list of layer descriptors
            edges      – adjacency list
            input_tensor – serialised random input for the first layer
            plan       – per-layer partition assignment
    """
    import time as _time
    import random

    start = _time.time()

    model_name = config.get("model", "resnet")
    input_size = config.get("input_size", 224)
    num_partitions = config.get("num_partitions", 3)
    dtype = config.get("dtype", "float32")

    # Build the DAG
    if model_name == "inception":
        layers, edges = build_inception_dag(num_partitions, input_size)
    else:
        layers, edges = build_resnet_dag(num_partitions, input_size)

    # Create a synthetic input tensor (channels-last: 1×H×W×3)
    random.seed(42)
    shape = (1, input_size, input_size, 3)

    try:
        import numpy as np
        input_tensor = np.random.randn(*shape).astype(dtype)
    except ImportError:
        total = 1
        for s in shape:
            total *= s
        input_tensor = [random.gauss(0, 1) for _ in range(total)]

    # Serialise the input
    serialised_input = _serialize_tensor(input_tensor)

    # Build per-layer partition plan
    plan = {l["layer_id"]: l["partition_idx"] for l in layers}

    elapsed = _time.time() - start
    print(
        f"[partition_model] model={model_name} layers={len(layers)} "
        f"edges={len(edges)} partitions={num_partitions} time={elapsed:.3f}s"
    )

    return {
        "model": model_name,
        "input_size": input_size,
        "num_partitions": num_partitions,
        "dtype": dtype,
        "layers": layers,
        "edges": edges,
        "input_tensor": serialised_input,
        "plan": plan,
        "partition_time": elapsed,
    }


# =====================================================================
# Stage 1 – Execute a single layer slice (simulated inference)
# =====================================================================

@crowdio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["layer_id", "progress_percent"],
)
def run_layer_slice(task_input):
    """
    Run inference on one layer slice.

    Receives upstream partition plan + input tensor.  Simulates
    compute proportional to the filter count and spatial size to mimic
    realistic latency differences between layers.

    Returns:
        dict with layer_id, output_tensor (serialised), exec_time
    """
    import time as _time
    import random
    import hashlib

    start = _time.time()

    layer_id = "unknown"
    progress_percent = 0.0

    # ── Unpack upstream results ─────────────────────────────────────
    upstream = task_input.get("upstream_results", {})
    original_args = task_input.get("original_args", {})

    if not upstream:
        return {"error": "No upstream results", "keys": list(task_input.keys())}

    # Merge all upstream partition plans (there is normally one from stage-0)
    partition_plan = {}
    layers = []
    input_tensor_blob = None
    model_name = "resnet"
    input_size = 224
    dtype = "float32"

    for _tid, result in upstream.items():
        if isinstance(result, str):
            import json as _json
            try:
                result = _json.loads(result)
            except Exception:
                import ast
                result = ast.literal_eval(result)

        partition_plan = result.get("plan", partition_plan)
        layers = result.get("layers", layers)
        if result.get("input_tensor"):
            input_tensor_blob = result["input_tensor"]
        model_name = result.get("model", model_name)
        input_size = result.get("input_size", input_size)
        dtype = result.get("dtype", dtype)

    # Determine which layer(s) this task should execute.
    # The task index inside stage-1 maps to the layer list order.
    task_index = original_args if isinstance(original_args, int) else 0
    if isinstance(original_args, dict):
        task_index = original_args.get("layer_index", 0)

    if task_index >= len(layers):
        return {"error": f"layer_index {task_index} out of range ({len(layers)})"}

    layer = layers[task_index]
    layer_id = layer["layer_id"]
    op = layer.get("op", "conv_block")
    params = layer.get("params", {})

    # ── Simulate inference (CPU-bound work) ─────────────────────────
    filters = params.get("filters", 64)
    kernel = params.get("kernel", 3)

    # Use the pre-computed output spatial size from the DAG, or fall back
    spatial = layer.get("output_spatial", input_size // max(params.get("stride", 1), 1))
    print(f"[DEBUG] layer={layer_id} output_spatial_in_layer={layer.get('output_spatial', 'MISSING')} "
          f"params_stride={params.get('stride', 'MISSING')} fallback={input_size // max(params.get('stride', 1), 1)} "
          f"ACTUAL_spatial={spatial} layer_keys={list(layer.keys())}")
    flops_proxy = filters * (kernel ** 2) * (spatial ** 2)
    # Map to a synthetic sleep so heavier layers take proportionally longer
    # Cap to keep tests fast: 0.05s – 2.0s
    sim_time = min(2.0, max(0.05, flops_proxy / 5e6))

    # Actually burn some CPU to produce a deterministic "activation" hash
    progress_percent = 0.0
    steps = max(1, int(sim_time * 100))
    accum = 0.0
    random.seed(hash(layer_id) & 0xFFFFFFFF)
    for step in range(steps):
        accum += random.gauss(0, 1)
        progress_percent = ((step + 1) / steps) * 100.0
        _time.sleep(sim_time / steps)

    # Produce a synthetic output tensor of shape (1, spatial, spatial, filters)
    out_shape = (1, spatial, spatial, filters)
    try:
        import numpy as np
        output = np.full(out_shape, accum / max(steps, 1), dtype=dtype)
    except ImportError:
        total_els = 1
        for s in out_shape:
            total_els *= s
        output = [accum / max(steps, 1)] * total_els

    output_blob = _serialize_tensor(output)

    elapsed = _time.time() - start
    print(
        f"[run_layer_slice] layer={layer_id} op={op} filters={filters} "
        f"spatial={spatial} sim_flops={flops_proxy:.0f} time={elapsed:.3f}s"
    )

    return {
        "layer_id": layer_id,
        "op": op,
        "output_tensor": output_blob,
        "output_shape": list(out_shape),
        "exec_time": elapsed,
        "sim_flops": flops_proxy,
    }


# =====================================================================
# Stage 2 – Fuse / join branches (skip-connection add, or concat)
# =====================================================================

@crowdio.task()
def fuse_branches(task_input):
    """
    Join / concatenate / add activation tensors from multiple upstream
    branches.  This stage is only unblocked when ALL branches have
    completed — the dependency counter ensures this.

    For ResNet (add):      element-wise addition of tensors.
    For Inception (concat): channel-axis concatenation.

    Returns:
        dict with fused_tensor, fused_shape, op, fuse_time
    """
    import time as _time

    start = _time.time()

    upstream = task_input.get("upstream_results", {})
    if not upstream:
        return {"error": "No upstream results for fusion"}

    # Collect all branch outputs
    branch_outputs = []
    ops_seen = set()

    for _tid, result in upstream.items():
        if isinstance(result, str):
            import json as _json
            try:
                result = _json.loads(result)
            except Exception:
                import ast
                result = ast.literal_eval(result)

        if "output_tensor" in result:
            tensor = _deserialize_tensor(result["output_tensor"])
            branch_outputs.append({
                "layer_id": result.get("layer_id", "?"),
                "tensor": tensor,
                "shape": result.get("output_shape", []),
            })
            ops_seen.add(result.get("op", ""))
        elif "fused_tensor" in result:
            # Upstream is itself a fuse stage (chained fusions)
            tensor = _deserialize_tensor(result["fused_tensor"])
            branch_outputs.append({
                "layer_id": result.get("layer_id", "fused"),
                "tensor": tensor,
                "shape": result.get("fused_shape", []),
            })
        # Also handle the partition_plan passthrough (from stage-0 → stage-2 skip)
        elif "plan" in result:
            # This is the partition-model output; skip it for fusion
            continue

    if not branch_outputs:
        return {"error": "No branch tensors found", "upstream_keys": list(upstream.keys())}

    print(
        f"[fuse_branches] merging {len(branch_outputs)} branches: "
        + ", ".join(b["layer_id"] for b in branch_outputs)
    )

    # ── Fuse ────────────────────────────────────────────────────────
    try:
        import numpy as np

        tensors = [np.asarray(b["tensor"]) for b in branch_outputs]

        # Determine fusion op: if shapes match → add; otherwise → concat
        shapes_match = all(t.shape == tensors[0].shape for t in tensors)
        if shapes_match and len(tensors) == 2:
            fused = tensors[0] + tensors[1]  # ResNet skip-add
            fuse_op = "add"
        else:
            # Concat along channel axis (last dim for channels-last)
            fused = np.concatenate(tensors, axis=-1)
            fuse_op = "concat"

        fused_shape = list(fused.shape)
        fused_blob = _serialize_tensor(fused)

    except ImportError:
        # Fallback: just concatenate flat lists
        flat = []
        for b in branch_outputs:
            t = b["tensor"]
            if isinstance(t, list):
                flat.extend(t)
            else:
                flat.append(t)
        fused_blob = _serialize_tensor(flat)
        fused_shape = [len(flat)]
        fuse_op = "concat_flat"

    elapsed = _time.time() - start
    print(f"[fuse_branches] op={fuse_op} output_shape={fused_shape} time={elapsed:.3f}s")

    return {
        "layer_id": "fused",
        "fuse_op": fuse_op,
        "fused_tensor": fused_blob,
        "fused_shape": fused_shape,
        "branch_count": len(branch_outputs),
        "fuse_time": elapsed,
    }


# =====================================================================
# Stage 3 – Final classification head
# =====================================================================

@crowdio.task()
def classify(task_input):
    """
    Run the classification head (Global Average Pooling → Dense → Softmax)
    on the fused feature map.

    Returns:
        dict with predicted_class, confidence, top5, classify_time
    """
    import time as _time
    import random
    import math as _math

    start = _time.time()

    upstream = task_input.get("upstream_results", {})
    if not upstream:
        return {"error": "No upstream results for classification"}

    # Get fused tensor
    fused_tensor = None
    fused_shape = None
    num_classes = 1000

    for _tid, result in upstream.items():
        if isinstance(result, str):
            import json as _json
            try:
                result = _json.loads(result)
            except Exception:
                import ast
                result = ast.literal_eval(result)

        if "fused_tensor" in result:
            fused_tensor = _deserialize_tensor(result["fused_tensor"])
            fused_shape = result.get("fused_shape", [])
        elif "output_tensor" in result:
            fused_tensor = _deserialize_tensor(result["output_tensor"])
            fused_shape = result.get("output_shape", [])

    if fused_tensor is None:
        return {"error": "No feature map found for classification"}

    # ── Simulate classification ─────────────────────────────────────
    # Global average pooling → dense → softmax
    try:
        import numpy as np
        feature = np.asarray(fused_tensor)
        # GAP: average over spatial dims (axes 1,2 for NHWC)
        if feature.ndim == 4:
            gap = feature.mean(axis=(1, 2)).flatten()
        else:
            gap = feature.flatten()

        # Simulated dense layer: random projection
        rng = np.random.RandomState(0)
        W = rng.randn(len(gap), num_classes).astype(np.float32) * 0.01
        logits = gap @ W

        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        top5_idx = probs.argsort()[-5:][::-1].tolist()
        top5 = [{"class": int(i), "prob": float(probs[i])} for i in top5_idx]
        pred_class = int(top5_idx[0])
        confidence = float(probs[pred_class])

    except ImportError:
        # Fallback: random prediction
        random.seed(42)
        pred_class = random.randint(0, num_classes - 1)
        confidence = random.random()
        top5 = [{"class": pred_class, "prob": confidence}]

    elapsed = _time.time() - start
    print(
        f"[classify] predicted_class={pred_class} confidence={confidence:.4f} "
        f"num_classes={num_classes} time={elapsed:.3f}s"
    )

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top5": top5,
        "fused_shape": fused_shape,
        "classify_time": elapsed,
    }


# =====================================================================
# Pipeline orchestration
# =====================================================================

async def run_dnn_pipeline(model_name, input_size, num_partitions, dtype="float32"):
    """
    Submit the 4-stage DNN pipeline through CROWDio.

    Stage 0  → partition_model   (1 task)
    Stage 1  → run_layer_slice   (N tasks, one per layer — independent branches
                                  run in parallel on different workers)
    Stage 2  → fuse_branches     (1 task — blocked until all Stage-1 complete)
    Stage 3  → classify          (1 task — blocked until Stage-2 completes)

    The dependency counter on the fuse task equals the number of incoming
    branches — exactly the mechanism the Kotlin AtomicInteger mirrors.
    """

    # Build the DAG to determine how many layer-slice tasks we need
    if model_name == "inception":
        layers, edges = build_inception_dag(num_partitions, input_size)
    else:
        layers, edges = build_resnet_dag(num_partitions, input_size)

    # Stage-1 needs one task per layer (excluding fuse & classify nodes)
    # The slice tasks are the layers that actually compute feature maps.
    slice_layers = [l for l in layers if l["op"] not in ("add", "concat", "classify")]
    num_slices = len(slice_layers)

    print(f"\n{'='*65}")
    print(f"DNN Pipeline: {model_name.upper()}")
    print(f"  Input        : {input_size}x{input_size}x3")
    print(f"  Partitions   : {num_partitions}")
    print(f"  Slice tasks  : {num_slices} (parallel branches)")
    print(f"  Total layers : {len(layers)}")
    print(f"  Stages       : partition -> slice x{num_slices} -> fuse -> classify")
    print(f"{'='*65}\n")

    config = {
        "model": model_name,
        "input_size": input_size,
        "num_partitions": num_partitions,
        "dtype": dtype,
    }

    start = time.time()

    # Build args_list for stage-1: one element per slice layer
    slice_args = [{"layer_index": i} for i in range(num_slices)]

    results = await pipeline([
        # Stage 0 – partition (single task)
        {
            "func": partition_model,
            "args_list": [config],
            "name": "partition_model",
        },
        # Stage 1 – layer slices (N tasks, can run in parallel on crowd)
        {
            "func": run_layer_slice,
            "args_list": slice_args,
            "pass_upstream_results": True,
            "name": "run_layer_slices",
        },
        # Stage 2 – fuse branches (1 task, blocked until ALL slices done)
        {
            "func": fuse_branches,
            "args_list": [None],
            "pass_upstream_results": True,
            "name": "fuse_branches",
        },
        # Stage 3 – classification head (1 task, blocked until fuse done)
        {
            "func": classify,
            "args_list": [None],
            "pass_upstream_results": True,
            "name": "classify",
        },
    ])

    elapsed = time.time() - start

    # ── Parse results ───────────────────────────────────────────────
    print(f"\nPipeline completed in {elapsed:.2f}s")
    if results:
        final = results[0]
        if isinstance(final, str):
            try:
                final = json.loads(final)
            except Exception:
                import ast
                final = ast.literal_eval(final)

        if "error" in final:
            print(f"  ERROR: {final['error']}")
        else:
            print(f"  Predicted class : {final.get('predicted_class')}")
            print(f"  Confidence      : {final.get('confidence', 0):.4f}")
            print(f"  Top-5           : {final.get('top5')}")
            print(f"  Fused shape     : {final.get('fused_shape')}")
            print(f"  Classify time   : {final.get('classify_time', 0):.3f}s")
            print(f"  Wall time       : {elapsed:.2f}s")

        return final, elapsed

    print("  No results returned!")
    return None, elapsed


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Distributed DNN Partitioning – dependency counter pipeline test"
    )
    p.add_argument("--model", default="resnet", choices=["resnet", "inception"],
                   help="Model architecture (default: resnet)")
    p.add_argument("--input-size", type=int, default=224,
                   help="Spatial input dimension (default: 224)")
    p.add_argument("--num-partitions", type=int, default=3,
                   help="Number of crowd-device partitions (default: 3)")
    p.add_argument("--dtype", default="float32",
                   help="Data type for tensors (default: float32)")
    p.add_argument("--output-dir",
                   default=os.path.join(os.path.dirname(__file__), "dnn_output"),
                   help="Output directory for results JSON")
    p.add_argument("--local", action="store_true",
                   help="Run all stages locally on this device (no foreman/workers)")
    p.add_argument("--host", default="localhost", help="Foreman host")
    p.add_argument("--port", type=int, default=9000, help="Foreman WebSocket port")
    return p.parse_args()


async def run_dnn_pipeline_local(model_name, input_size, num_partitions, dtype="float32"):
    """
    Run the full DNN pipeline locally on a single device.

    Executes all 4 stages sequentially in-process, simulating the same
    dependency-counter flow without requiring a foreman or workers.
    This is useful for:
      - Testing / debugging the pipeline logic
      - Single-device baselines for comparison with distributed runs
      - Environments where foreman/workers are not available
    """

    # Build the DAG
    if model_name == "inception":
        layers, edges = build_inception_dag(num_partitions, input_size)
    else:
        layers, edges = build_resnet_dag(num_partitions, input_size)

    slice_layers = [l for l in layers if l["op"] not in ("add", "concat", "classify")]
    num_slices = len(slice_layers)

    print(f"\n{'='*65}")
    print(f"DNN Pipeline (LOCAL): {model_name.upper()}")
    print(f"  Input        : {input_size}x{input_size}x3")
    print(f"  Partitions   : {num_partitions} (all on this device)")
    print(f"  Slice tasks  : {num_slices}")
    print(f"  Total layers : {len(layers)}")
    print(f"  Stages       : partition -> slice x{num_slices} -> fuse -> classify")
    print(f"{'='*65}\n")

    config = {
        "model": model_name,
        "input_size": input_size,
        "num_partitions": num_partitions,
        "dtype": dtype,
    }

    start = time.time()

    # ── Stage 0: partition ──────────────────────────────────────────
    print("[local] Stage 0: partition_model")
    stage0_result = partition_model.__crowdio_original__(config) if hasattr(partition_model, "__crowdio_original__") else partition_model(config)

    # ── Stage 1: run each layer slice sequentially ─────────────────
    slice_results = {}
    for i in range(num_slices):
        print(f"[local] Stage 1: run_layer_slice [{i+1}/{num_slices}]")
        task_input = {
            "original_args": {"layer_index": i},
            "upstream_results": {f"local_stage0_task0": stage0_result},
        }
        fn = run_layer_slice.__crowdio_original__ if hasattr(run_layer_slice, "__crowdio_original__") else run_layer_slice
        result = fn(task_input)
        slice_results[f"local_stage1_task{i}"] = result

    # ── Stage 2: fuse branches ─────────────────────────────────────
    print(f"[local] Stage 2: fuse_branches ({len(slice_results)} inputs)")
    fuse_input = {
        "original_args": None,
        "upstream_results": slice_results,
    }
    fn = fuse_branches.__crowdio_original__ if hasattr(fuse_branches, "__crowdio_original__") else fuse_branches
    fuse_result = fn(fuse_input)

    # ── Stage 3: classify ──────────────────────────────────────────
    print("[local] Stage 3: classify")
    classify_input = {
        "original_args": None,
        "upstream_results": {"local_stage2_task0": fuse_result},
    }
    fn = classify.__crowdio_original__ if hasattr(classify, "__crowdio_original__") else classify
    final = fn(classify_input)

    elapsed = time.time() - start

    # ── Parse results ───────────────────────────────────────────────
    print(f"\nPipeline (local) completed in {elapsed:.2f}s")
    if final:
        if "error" in final:
            print(f"  ERROR: {final['error']}")
        else:
            print(f"  Predicted class : {final.get('predicted_class')}")
            print(f"  Confidence      : {final.get('confidence', 0):.4f}")
            print(f"  Top-5           : {final.get('top5')}")
            print(f"  Fused shape     : {final.get('fused_shape')}")
            print(f"  Classify time   : {final.get('classify_time', 0):.3f}s")
            print(f"  Wall time       : {elapsed:.2f}s")
        return final, elapsed

    print("  No results returned!")
    return None, elapsed


async def main():
    args = parse_args()

    if args.local:
        result, elapsed = await run_dnn_pipeline_local(
            model_name=args.model,
            input_size=args.input_size,
            num_partitions=args.num_partitions,
            dtype=args.dtype,
        )
    else:
        await connect(args.host, args.port)

    try:
        if not args.local:
            result, elapsed = await run_dnn_pipeline(
                model_name=args.model,
                input_size=args.input_size,
                num_partitions=args.num_partitions,
                dtype=args.dtype,
            )

        # Save result
        if result:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(
                args.output_dir,
                f"dnn_{args.model}_{args.num_partitions}p_result.json",
            )
            # Remove large tensor blobs before saving summary
            summary = {k: v for k, v in result.items()
                       if k not in ("fused_tensor", "output_tensor")}
            summary["wall_time"] = elapsed
            summary["model"] = args.model
            summary["num_partitions"] = args.num_partitions
            summary["input_size"] = args.input_size

            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n  Result saved -> {out_path}")

        # ── Summary ─────────────────────────────────────────────────
        print(f"\n{'='*65}")
        print("DNN PARTITIONING TEST SUMMARY")
        print(f"{'='*65}")
        print(f"  Model          : {args.model}")
        print(f"  Input size     : {args.input_size}x{args.input_size}x3")
        print(f"  Partitions     : {args.num_partitions}")
        print(f"  Wall time      : {elapsed:.2f}s")
        if result and "predicted_class" in result:
            print(f"  Prediction     : class {result['predicted_class']} "
                  f"({result.get('confidence', 0):.4f})")
        print(f"{'='*65}")

    finally:
        if not args.local:
            await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
