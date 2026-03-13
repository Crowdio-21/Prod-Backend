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
                                   plan (DAG -> device map), and serialise
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
# TFLite helpers  (used by partition_model on the backend / PC worker)
# =====================================================================

def _tensorflow_available():
    """Return True when TensorFlow is importable on this machine."""
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _build_tflite_sub_models(num_partitions, input_size, num_classes=1000):
    """
    Build real TFLite sub-models for each partition using TensorFlow.

    Creates a sequential chain where partition *i* takes the output tensor
    of partition *i-1* as its input.  The final partition appends a
    GlobalAveragePooling2D -> Dense(num_classes, softmax) classification head.

    Each sub-model is converted to a TFLite flatbuffer so it can be shipped
    to Android workers and executed via ``tflite_runtime`` without requiring
    TensorFlow on the device.

    Returns:
        dict  partition_index (int) ->
              {"tflite_b64": str, "input_shape": list, "output_shape": list}
    """
    import tensorflow as tf
    import base64

    partitions = {}
    h, w, c = input_size, input_size, 3

    for i in range(num_partitions):
        filters = min(64 * (2 ** i), 512)
        stride = 2 if i < num_partitions - 1 else 1
        is_last = (i == num_partitions - 1)
        inp_shape = (h, w, c)

        model_inp = tf.keras.Input(shape=inp_shape, batch_size=1)
        x = tf.keras.layers.Conv2D(
            filters, 3, strides=stride, padding="same", use_bias=False
        )(model_inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if is_last:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs=model_inp, outputs=x)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_bytes = converter.convert()
        out_shape = list(model.output_shape[1:])   # drop batch dim

        partitions[i] = {
            "tflite_b64": base64.b64encode(tflite_bytes).decode("utf-8"),
            "input_shape": list(inp_shape),
            "output_shape": out_shape,
        }

        # Compute next partition's input spatial size
        h = (h + stride - 1) // stride
        w = (w + stride - 1) // stride
        c = filters

    return partitions


# =====================================================================
# DAG builder – describes how layers relate to one another
# =====================================================================

def build_resnet_dag(num_partitions=3, input_size=224):
    """
    Build a simplified ResNet-like DAG.

    The graph has a main chain with one skip connection:

        [input] -> partition_0  -> partition_1  ->  fuse (skip join) -> classify
                  ╰------------ skip branch ------╯

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

    # Edges: partition_0 -> partition_1 -> fuse -> classify
    #         partition_0 -> skip_branch  -> fuse
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
    Build the partition plan and convert each partition to a TFLite sub-model.

    Stage 0 runs once on a device that has TensorFlow installed (e.g. a PC
    worker).  Its output is forwarded to every Stage-1 Android device so that
    workers can run actual TFLite inference without needing TensorFlow locally.

    When TensorFlow is available the function creates real TFLite flatbuffers
    (one per partition) embedded as base64 strings under ``tflite_models``.
    Workers decode these bytes and run them via ``tflite_runtime``.

    Falls back to synthetic shape metadata only when TensorFlow is absent,
    allowing the pipeline to be exercised with simulated compute.

    Args:
        config (dict):
            model          – architecture tag ("resnet" | "inception")
            input_size     – spatial dimension of the input image (default 224)
            num_partitions – number of TFLite sub-models / devices (default 3)
            dtype          – tensor data type, "float32" (default)
            num_classes    – classifier output size (default 1000)

    Returns:
        dict with:
            model, input_size, num_partitions, dtype,
            layers         – one descriptor per partition (linear chain)
            edges          – adjacency list (linear: i -> i+1)
            input_tensor   – serialised 1xHxWx3 float32 image for partition 0
            plan           – {layer_id: partition_idx}
            tflite_models  – {str(partition_idx): {tflite_b64, input_shape,
                               output_shape}}  (present when TF is available)
    """
    import time as _time

    start = _time.time()

    model_name = config.get("model", "resnet")
    input_size = config.get("input_size", 224)
    num_partitions = config.get("num_partitions", 3)
    dtype = config.get("dtype", "float32")
    num_classes = config.get("num_classes", 1000)

    # -- Build TFLite sub-models on the machine that has TensorFlow -------
    tflite_models = {}
    if _tensorflow_available():
        try:
            print(
                f"[partition_model] Building {num_partitions} TFLite "
                f"sub-models (input {input_size}x{input_size}x3)..."
            )
            tflite_models = _build_tflite_sub_models(
                num_partitions, input_size, num_classes
            )
            size_summary = ", ".join(
                f"p{k}->{v['output_shape']} "
                f"({len(v['tflite_b64']) * 3 // 4 // 1024} KB)"
                for k, v in tflite_models.items()
            )
            print(f"[partition_model] TFLite models ready: {size_summary}")
        except Exception as _exc:
            print(
                f"[partition_model] TFLite build failed ({_exc}); "
                f"workers will use simulated inference"
            )

    # -- Build a linear partition chain as the DAG ------------------------
    layers = []
    edges = []
    for i in range(num_partitions):
        if tflite_models:
            info = tflite_models[i]
            out_s = info["output_shape"]
            spatial = out_s[0] if len(out_s) >= 2 else 1
            filters = out_s[-1]
        else:
            spatial = max(1, input_size // (2 ** i))
            filters = min(64 * (2 ** i), 512)

        layers.append({
            "layer_id": f"partition_{i}",
            "op": "tflite_partition",
            "partition_idx": i,
            "output_spatial": spatial,
            "params": {"filters": filters},
        })
        if i > 0:
            edges.append((f"partition_{i - 1}", f"partition_{i}"))

    # -- Create the first-stage input image -------------------------------
    shape = (1, input_size, input_size, 3)
    try:
        import numpy as np
        rng = np.random.RandomState(42)
        input_array = rng.randn(*shape).astype(dtype)
    except ImportError:
        import random as _r
        _r.seed(42)
        total = 1
        for _s in shape:
            total *= _s
        input_array = [_r.gauss(0, 0.5) for _ in range(total)]

    serialised_input = _serialize_tensor(input_array)
    plan = {l["layer_id"]: l["partition_idx"] for l in layers}

    elapsed = _time.time() - start
    print(
        f"[partition_model] model={model_name} partitions={num_partitions} "
        f"tflite_built={bool(tflite_models)} time={elapsed:.3f}s"
    )

    result = {
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
    if tflite_models:
        # JSON requires string keys; convert int -> str
        result["tflite_models"] = {str(k): v for k, v in tflite_models.items()}
    return result


# =====================================================================
# Stage 1 – Execute a TFLite partition slice on a worker device
# =====================================================================

@crowdio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["partition_idx", "progress_percent"],
)
def run_tflite_partition(task_input):
    """
    Run actual TFLite inference for one partition on a worker device.

    Designed to execute on Android workers via Chaquopy (tflite_runtime)
    or on PC workers with TensorFlow installed.

    IMPORTANT – SELF-CONTAINED DESIGN
    ----------------------------------
    All helper logic (tensor encode/decode, TFLite interpreter invocation)
    is defined as nested functions.  This is mandatory: the SDK serialises
    this function via ``inspect.getsource()`` and the Android executor
    ``exec()``s it in an isolated namespace.  Module-level helpers such as
    ``_serialize_tensor`` are NOT available in that namespace.

    Flow
    ----
    1. Extract the TFLite flatbuffer (base64) for this partition from
       upstream results.
    2. Reconstruct the input tensor:
         – partition 0  ->  original 1xHxWx3 image from ``partition_model``
         – partition N  ->  ``output_tensor`` from partition N-1
    3. Run ``tflite_runtime.interpreter.Interpreter`` (Android / Chaquopy)
       or ``tensorflow.lite.Interpreter`` (PC fallback).
    4. Return the output tensor and a pass-through of ``tflite_models`` so
       the next sequential stage can retrieve its own sub-model bytes.

    Falls back to a random (simulated) output when no TFLite backend is
    found, keeping the pipeline runnable for profiling / testing.

    Android build.gradle requirements
    ----------------------------------
    In your Chaquopy configuration add::

        python {
            pip {
                install "tflite-runtime"   // TFLite interpreter
                install "numpy"            // tensor math
            }
        }

    Args:
        task_input (dict):
            original_args    – {"partition_idx": int}
            upstream_results – dict of previous stage task results

    Returns:
        dict with partition_idx, output_tensor, output_shape, exec_time,
        backend_used, tflite_models (pass-through for next stage), dtype
    """
    import time as _time
    import base64
    import io
    import numpy as np

    # -- Inline helpers ----------------------------------------------------
    # Defined inside the function so the serialised source is self-contained
    # when exec'd on Android workers via Chaquopy.

    def _decode_tensor(blob):
        """Reconstruct a numpy array from a serialised tensor dict."""
        if isinstance(blob, str):
            import json as _j
            try:
                blob = _j.loads(blob)
            except Exception:
                import ast as _a
                blob = _a.literal_eval(blob)
        fmt = blob.get("format", "json")
        if fmt == "npy":
            raw = base64.b64decode(blob["data_b64"])
            return np.load(io.BytesIO(raw))
        data = blob.get("data", [])
        shape = blob.get("shape")
        arr = np.array(data, dtype=blob.get("dtype", "float32"))
        if shape:
            arr = arr.reshape(shape)
        return arr

    def _encode_tensor(arr):
        """Serialise a numpy array to a portable base64 dict."""
        arr = np.asarray(arr)
        buf = io.BytesIO()
        np.save(buf, arr)
        return {
            "format": "npy",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        }

    def _run_tflite(tflite_b64_str, input_array):
        """
        Run TFLite inference from a base64-encoded flatbuffer string.

        Tries ``tflite_runtime`` first (lightweight package shipped with
        Chaquopy APKs), then falls back to ``tensorflow.lite`` on PC.

        Returns (output_ndarray, backend_name) or (None, error_string).
        """
        tflite_bytes = base64.b64decode(tflite_b64_str)

        _Interp = None
        _backend = "none"
        try:
            # Android / Chaquopy path
            from tflite_runtime.interpreter import Interpreter as _IR
            _Interp = _IR
            _backend = "tflite_runtime"
        except ImportError:
            try:
                # PC fallback
                import tensorflow as _tf
                _Interp = _tf.lite.Interpreter
                _backend = "tensorflow.lite"
            except ImportError:
                return None, "no_tflite_backend"

        # Prefer loading from bytes (avoids disk I/O).
        # Older tflite_runtime builds only accept a file path, so we fall
        # back to writing a temp file in that case.
        try:
            interp = _Interp(model_content=tflite_bytes)
        except TypeError:
            import tempfile as _tmp
            import os as _os
            _f = _tmp.NamedTemporaryFile(suffix=".tflite", delete=False)
            try:
                _f.write(tflite_bytes)
                _f.close()
                interp = _Interp(model_path=_f.name)
            finally:
                _os.unlink(_f.name)

        interp.allocate_tensors()
        inp_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        arr = np.array(input_array, dtype=inp_det["dtype"]).reshape(inp_det["shape"])
        interp.set_tensor(inp_det["index"], arr)
        interp.invoke()
        return interp.get_tensor(out_det["index"]), _backend

    # -- Unpack task arguments ----------------------------------------------
    start = _time.time()
    progress_percent = 0.0

    upstream = task_input.get("upstream_results", {})
    original_args = task_input.get("original_args", {})
    partition_idx = (
        original_args.get("partition_idx", 0)
        if isinstance(original_args, dict)
        else int(original_args or 0)
    )

    if not upstream:
        return {"error": "No upstream results", "partition_idx": partition_idx}

    # -- Gather tflite_models and the correct input tensor -----------------
    tflite_models = {}
    input_tensor_blob = None
    input_size = 224
    dtype = "float32"

    for _tid, result in upstream.items():
        if isinstance(result, str):
            import json as _json
            try:
                result = _json.loads(result)
            except Exception:
                import ast as _ast
                result = _ast.literal_eval(result)
        if not isinstance(result, dict):
            continue

        # Collect tflite_models from any upstream result that carries them
        if result.get("tflite_models"):
            tflite_models = result["tflite_models"]

        # Partition 0: consume the original image from partition_model
        if partition_idx == 0 and result.get("input_tensor") and input_tensor_blob is None:
            input_tensor_blob = result["input_tensor"]
            input_size = result.get("input_size", input_size)
            dtype = result.get("dtype", dtype)

        # Partition N>0: consume the output tensor from partition N-1
        if partition_idx > 0 and result.get("output_tensor") is not None:
            if result.get("partition_idx") == partition_idx - 1:
                input_tensor_blob = result["output_tensor"]
                dtype = result.get("dtype", dtype)

    # Fallback: accept any output_tensor when the partition_idx tag is absent
    if input_tensor_blob is None and partition_idx > 0:
        for _tid, result in upstream.items():
            if isinstance(result, dict) and result.get("output_tensor") is not None:
                input_tensor_blob = result["output_tensor"]
                break

    if input_tensor_blob is None:
        return {
            "error": f"No input tensor found for partition {partition_idx}",
            "upstream_keys": list(upstream.keys()),
            "partition_idx": partition_idx,
        }

    progress_percent = 10.0

    # -- Decode input tensor -----------------------------------------------
    input_array = _decode_tensor(input_tensor_blob)
    output = None
    backend_used = "simulation"

    # -- TFLite inference --------------------------------------------------
    tflite_info = tflite_models.get(str(partition_idx))
    if tflite_info and tflite_info.get("tflite_b64"):
        output, backend_used = _run_tflite(tflite_info["tflite_b64"], input_array)
        if output is None:
            print(
                f"[run_tflite_partition] p={partition_idx} "
                f"backend failed ({backend_used}), falling back to simulation"
            )

    progress_percent = 80.0

    # -- Simulation fallback (no TFLite backend available) -----------------
    if output is None:
        out_shape_info = (tflite_info or {}).get("output_shape", [])
        if out_shape_info:
            out_shape = tuple([1] + [int(d) for d in out_shape_info])
        else:
            spatial = max(1, input_size // (2 ** partition_idx))
            filters = min(64 * (2 ** partition_idx), 512)
            out_shape = (1, spatial, spatial, filters)
        rng = np.random.default_rng(partition_idx)
        output = rng.standard_normal(out_shape).astype(dtype)

    progress_percent = 90.0
    output_blob = _encode_tensor(output)
    try:
        out_shape_list = list(output.shape)
    except AttributeError:
        out_shape_list = []

    progress_percent = 100.0
    elapsed = _time.time() - start
    print(
        f"[run_tflite_partition] p={partition_idx} backend={backend_used} "
        f"out_shape={out_shape_list} time={elapsed:.3f}s"
    )

    return {
        "partition_idx": partition_idx,
        "output_tensor": output_blob,
        "output_shape": out_shape_list,
        "exec_time": elapsed,
        "backend_used": backend_used,
        # Pass tflite_models through so each subsequent partition stage
        # can retrieve its own sub-model without a round-trip to stage-0.
        "tflite_models": tflite_models,
        "dtype": dtype,
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
        # Also handle the partition_plan passthrough (from stage-0 -> stage-2 skip)
        elif "plan" in result:
            # This is the partition-model output; skip it for fusion
            continue

    if not branch_outputs:
        return {"error": "No branch tensors found", "upstream_keys": list(upstream.keys())}

    print(
        f"[fuse_branches] merging {len(branch_outputs)} branches: "
        + ", ".join(b["layer_id"] for b in branch_outputs)
    )

    # -- Fuse --------------------------------------------------------
    try:
        import numpy as np

        tensors = [np.asarray(b["tensor"]) for b in branch_outputs]

        # Determine fusion op: if shapes match -> add; otherwise -> concat
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
    Run the classification head (Global Average Pooling -> Dense -> Softmax)
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

    # -- Simulate classification -------------------------------------
    # Global average pooling -> dense -> softmax
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
    Submit the TFLite DNN inference pipeline through CROWDio.

    Dynamic stage list (determined by num_partitions):

        Stage 0          -> partition_model     (1 task – PC worker with TF)
        Stage 1..N       -> run_tflite_partition (1 task each – sequential;
                           each stage feeds its output_tensor to the next)
        Stage N+1        -> classify             (1 task)

    Sequential partitioning models real distributed pipeline inference:
    device 0 runs the first sub-model, passes activations to device 1,
    which runs the second sub-model, and so on.

    Each ``run_tflite_partition`` task:
      • Uses ``tflite_runtime.interpreter.Interpreter`` on Android (Chaquopy)
      • Falls back to ``tensorflow.lite.Interpreter`` on PC
      • Passes ``tflite_models`` through in its result so the subsequent
        stage can find its own sub-model bytes without querying stage 0.
    """
    print(f"\n{'='*65}")
    print(f"DNN TFLite Pipeline: {model_name.upper()}")
    print(f"  Input        : {input_size}x{input_size}x3")
    print(f"  Partitions   : {num_partitions}")
    print(f"  Stages       : partition_model -> tflite_partitionx{num_partitions} -> classify")
    print(f"{'='*65}\n")

    config = {
        "model": model_name,
        "input_size": input_size,
        "num_partitions": num_partitions,
        "dtype": dtype,
    }

    start = time.time()

    # Build stage list dynamically: one sequential TFLite stage per partition
    stages = [
        {
            "func": partition_model,
            "args_list": [config],
            "name": "partition_model",
        },
    ]
    for i in range(num_partitions):
        stages.append({
            "func": run_tflite_partition,
            "args_list": [{"partition_idx": i}],
            "pass_upstream_results": True,
            "name": f"tflite_partition_{i}",
        })
    stages.append({
        "func": classify,
        "args_list": [None],
        "pass_upstream_results": True,
        "name": "classify",
    })

    results = await pipeline(stages)
    elapsed = time.time() - start

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
    Run the full TFLite pipeline locally on a single device.

    Executes all stages sequentially in-process.  Useful for:
      - Verifying that TFLite sub-models build and run correctly on the PC
      - Debugging the pipeline without a foreman / workers
      - Single-device latency baselines
    """
    print(f"\n{'='*65}")
    print(f"DNN TFLite Pipeline (LOCAL): {model_name.upper()}")
    print(f"  Input        : {input_size}x{input_size}x3")
    print(f"  Partitions   : {num_partitions} (all on this device)")
    print(f"  Stages       : partition_model -> tflite_partitionx{num_partitions} -> classify")
    print(f"{'='*65}\n")

    config = {
        "model": model_name,
        "input_size": input_size,
        "num_partitions": num_partitions,
        "dtype": dtype,
    }

    start = time.time()

    # -- Stage 0: build TFLite sub-models and partition plan --------------
    print("[local] Stage 0: partition_model")
    _pm = (
        partition_model.__crowdio_original__
        if hasattr(partition_model, "__crowdio_original__")
        else partition_model
    )
    stage0_result = _pm(config)

    # -- Stages 1...N: sequential TFLite partition inference ----------------
    _rtp = (
        run_tflite_partition.__crowdio_original__
        if hasattr(run_tflite_partition, "__crowdio_original__")
        else run_tflite_partition
    )
    prev_result = stage0_result
    for i in range(num_partitions):
        print(f"[local] Stage {i + 1}: run_tflite_partition (partition_idx={i})")
        if i == 0:
            # First partition: stage-0 result has tflite_models + input_tensor
            upstream = {"stage0": stage0_result}
        else:
            # Subsequent partitions: previous result carries tflite_models
            # pass-through + output_tensor
            upstream = {f"partition_{i - 1}": prev_result}
        task_input = {
            "original_args": {"partition_idx": i},
            "upstream_results": upstream,
        }
        prev_result = _rtp(task_input)
        backend = prev_result.get("backend_used", "?")
        shape = prev_result.get("output_shape", "?")
        print(f"           -> backend={backend}  output_shape={shape}")

    # -- Final stage: classification head ---------------------------------
    print(f"[local] Stage {num_partitions + 1}: classify")
    _cls = (
        classify.__crowdio_original__
        if hasattr(classify, "__crowdio_original__")
        else classify
    )
    classify_input = {
        "original_args": None,
        "upstream_results": {f"partition_{num_partitions - 1}": prev_result},
    }
    final = _cls(classify_input)

    elapsed = time.time() - start
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

        # -- Summary -------------------------------------------------
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
