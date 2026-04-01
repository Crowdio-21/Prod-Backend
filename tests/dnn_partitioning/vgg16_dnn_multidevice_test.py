#!/usr/bin/env python3
"""
Run a famous DNN model (VGG16) across multiple devices using CROWDio.

This script builds partitioned TFLite sub-models from VGG16 and submits ONE
pipeline job with multiple tasks under each stage. Each stream follows:

  partition_0 -> partition_1 -> ... -> partition_(N-1) -> classify

Usage (PowerShell):
  uv run python tests/dnn_partitioning/famous_dnn_multidevice_test.py `
    --host localhost `
    --port 9000 `
    --num-devices 2 `
    --tasks-per-stage 4 `
    --weights none

Notes:
- VGG16 with include_top=True requires input-size 224.
- --weights imagenet may download weights from the internet.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import sys
import time
import uuid
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crowdio import crowdio_connect, crowdio_disconnect, CROWDio, crowdio_pipeline


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc


def _require_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise RuntimeError("tensorflow is required") from exc


def _serialize_tensor(arr):
    np = _require_numpy()
    arr = np.asarray(arr)
    buf = io.BytesIO()
    np.save(buf, arr)
    return {
        "format": "npy",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
    }


def _deserialize_tensor(blob):
    np = _require_numpy()
    if blob.get("format", "json") == "npy":
        data_b64 = blob.get("data_b64")
        if data_b64:
            raw = base64.b64decode(data_b64)
        else:
            file_url = blob.get("file_url")
            if not file_url:
                raise ValueError("No tensor data_b64 or file_url in output_tensor")
            import requests

            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            raw = response.content
        return np.load(io.BytesIO(raw), allow_pickle=False)
    arr = np.array(blob.get("data", []), dtype=blob.get("dtype", "float32"))
    if blob.get("shape"):
        arr = arr.reshape(blob["shape"])
    return arr


def _run_tflite_local(tflite_b64_str, input_array):
    np = _require_numpy()
    import tensorflow as tf

    tflite_bytes = base64.b64decode(tflite_b64_str)
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    arr = np.array(input_array, dtype=inp["dtype"]).reshape(inp["shape"])
    interp.set_tensor(inp["index"], arr)
    interp.invoke()
    return interp.get_tensor(out["index"])


def _run_tflite_local_bytes(tflite_bytes, input_array):
    np = _require_numpy()
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    arr = np.array(input_array, dtype=inp["dtype"]).reshape(inp["shape"])
    interp.set_tensor(inp["index"], arr)
    interp.invoke()
    return interp.get_tensor(out["index"])


def _build_vgg16_tflite_partitions(num_devices: int, weights: str):
    """Build VGG16 and split layers into num_devices sequential TFLite parts."""
    tf = _require_tensorflow()

    if weights == "none":
        weights_arg = None
    else:
        weights_arg = "imagenet"

    # VGG16 include_top=True requires 224x224 input
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights=weights_arg,
        input_shape=(224, 224, 3),
        classes=1000,
        classifier_activation="softmax",
    )

    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    total = len(layers)
    chunk = max(1, total // num_devices)

    partitions = {}
    start = 0
    for p in range(num_devices):
        end = total if p == num_devices - 1 else min(total, start + chunk)
        part_layers = layers[start:end]

        if not part_layers:
            raise RuntimeError("Partitioning produced an empty layer chunk")

        in_shape = tuple(int(d) for d in part_layers[0].input.shape[1:])
        inp = tf.keras.Input(shape=in_shape, batch_size=1)
        x = inp
        for layer in part_layers:
            x = layer(x)

        sub_model = tf.keras.Model(inputs=inp, outputs=x)
        converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
        tflite_bytes = converter.convert()

        partitions[p] = {
            "tflite_b64": base64.b64encode(tflite_bytes).decode("utf-8"),
            "input_shape": [int(d) for d in sub_model.input_shape[1:]],
            "output_shape": [int(d) for d in sub_model.output_shape[1:]],
            "layer_start": start,
            "layer_end": end - 1,
        }

        start = end

    return partitions


def _persist_tflite_models_to_store(tflite_models, model_store_dir, model_id=None):
    model_id = model_id or f"vgg16_partitions_{uuid.uuid4().hex[:8]}"
    model_dir = os.path.join(model_store_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    manifest = {"model_id": model_id, "parts": {}}
    for idx, info in tflite_models.items():
        filename = f"partition_{idx}.tflite"
        path = os.path.join(model_dir, filename)
        model_bytes = base64.b64decode(info["tflite_b64"])
        with open(path, "wb") as fh:
            fh.write(model_bytes)
        digest = hashlib.sha256(model_bytes).hexdigest()

        manifest["parts"][str(idx)] = {
            "file": filename,
            "input_shape": info.get("input_shape", []),
            "output_shape": info.get("output_shape", []),
            "layer_start": info.get("layer_start"),
            "layer_end": info.get("layer_end"),
            "sha256": digest,
        }

    with open(os.path.join(model_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return model_id, manifest


def _build_stage0_payload(
    num_devices: int,
    dtype: str,
    weights: str,
    model_store_dir: str,
    model_base_url: str = None,
    worker_model_cache_dir: str = ".worker_model_cache",
):
    np = _require_numpy()

    tflite_models = _build_vgg16_tflite_partitions(num_devices, weights)
    model_store_dir = os.path.abspath(model_store_dir)
    model_id, model_manifest = _persist_tflite_models_to_store(
        tflite_models, model_store_dir=model_store_dir, model_id="vgg16_partitioned"
    )

    rng = np.random.RandomState(42)
    # Typical image input in [0, 255] for VGG preprocessing style
    input_array = (rng.rand(1, 224, 224, 3) * 255.0).astype(dtype)

    return {
        "model": "vgg16",
        "input_size": 224,
        "num_partitions": num_devices,
        "dtype": dtype,
        "layers": [
            {
                "layer_id": f"partition_{i}",
                "op": "tflite_partition",
                "partition_idx": i,
                "layer_start": tflite_models[i]["layer_start"],
                "layer_end": tflite_models[i]["layer_end"],
            }
            for i in range(num_devices)
        ],
        "edges": [(f"partition_{i - 1}", f"partition_{i}") for i in range(1, num_devices)],
        "input_tensor": _serialize_tensor(input_array),
        "plan": {f"partition_{i}": i for i in range(num_devices)},
        "model_id": model_id,
        "model_store_dir": model_store_dir,
        "model_manifest": model_manifest,
        "model_base_url": model_base_url,
        "worker_model_cache_dir": worker_model_cache_dir,
    }


@CROWDio.task()
def emit_payload(payload):
    return payload


@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["partition_idx", "progress_percent"],
)
def run_tflite_partition(task_input):
    import base64
    import hashlib
    import io
    import json as _json
    import os
    import time as _time
    import urllib.request
    from pathlib import Path

    import numpy as np
    import requests

    def _decode(blob):
        if isinstance(blob, str):
            try:
                blob = _json.loads(blob)
            except Exception:
                import ast as _ast

                blob = _ast.literal_eval(blob)
        if blob.get("format", "json") == "npy":
            data_b64 = blob.get("data_b64")
            if data_b64:
                raw = base64.b64decode(data_b64)
            else:
                file_url = blob.get("file_url")
                if not file_url:
                    raise ValueError("No tensor data_b64 or file_url in output_tensor")
                response = requests.get(file_url, timeout=30)
                response.raise_for_status()
                raw = response.content
            return np.load(io.BytesIO(raw), allow_pickle=False)
        arr = np.array(blob.get("data", []), dtype=blob.get("dtype", "float32"))
        if blob.get("shape"):
            arr = arr.reshape(blob["shape"])
        return arr

    def _encode(arr):
        arr = np.asarray(arr)
        buf = io.BytesIO()
        np.save(buf, arr)
        return {
            "format": "npy",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        }

    def _load_tensor_ref(tensor_ref):
        import base64 as _base64

        # Supports http(s) URLs and local filesystem paths.
        if isinstance(tensor_ref, dict):
            if tensor_ref.get("file_url"):
                tensor_ref = tensor_ref["file_url"]
            elif tensor_ref.get("path"):
                tensor_ref = tensor_ref["path"]
            else:
                raise ValueError("tensor_ref dict must include 'file_url' or 'path'")

        if not isinstance(tensor_ref, str) or not tensor_ref.strip():
            raise ValueError("tensor_ref must be a non-empty string or dict")

        ref = tensor_ref.strip()
        if ref.startswith("http://") or ref.startswith("https://"):
            response = requests.get(ref, timeout=30)
            response.raise_for_status()
            raw = response.content
        else:
            path = Path(ref.replace("file://", "")).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"tensor_ref path not found: {path}")
            raw = path.read_bytes()

        # Normal path: raw .npy payload.
        try:
            return np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception:
            # Offloaded payload may be a JSON task_result envelope.
            text = raw.decode("utf-8")
            envelope = _json.loads(text)
            result_obj = (envelope.get("data") or {}).get("result")
            if isinstance(result_obj, str):
                try:
                    result_obj = _json.loads(result_obj)
                except Exception:
                    import ast as _ast

                    result_obj = _ast.literal_eval(result_obj)

            if not isinstance(result_obj, dict):
                raise ValueError("tensor_ref payload did not contain a dict result")

            output_tensor = result_obj.get("output_tensor")
            if not isinstance(output_tensor, dict):
                raise ValueError("tensor_ref payload did not contain output_tensor")

            data_b64 = output_tensor.get("data_b64")
            if data_b64:
                raw_tensor = _base64.b64decode(data_b64)
                return np.load(io.BytesIO(raw_tensor), allow_pickle=False)

            nested_file_url = output_tensor.get("file_url")
            if nested_file_url:
                response = requests.get(nested_file_url, timeout=30)
                response.raise_for_status()
                return np.load(io.BytesIO(response.content), allow_pickle=False)

            raise ValueError("output_tensor missing both data_b64 and file_url")

    def _run_tflite(tflite_b64_str, input_array):
        tflite_bytes = base64.b64decode(tflite_b64_str)
        interp_cls = None
        backend = "none"

        try:
            import importlib

            mod = importlib.import_module("tflite_runtime.interpreter")
            interp_cls = mod.Interpreter
            backend = "tflite_runtime"
        except ImportError:
            try:
                import tensorflow as _tf

                interp_cls = _tf.lite.Interpreter
                backend = "tensorflow.lite"
            except ImportError:
                return None, "no_tflite_backend"

        try:
            interp = interp_cls(model_content=tflite_bytes)
        except TypeError:
            import os as _os
            import tempfile as _tmp

            tmp = _tmp.NamedTemporaryFile(suffix=".tflite", delete=False)
            try:
                tmp.write(tflite_bytes)
                tmp.close()
                interp = interp_cls(model_path=tmp.name)
            finally:
                _os.unlink(tmp.name)

        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        arr = np.array(input_array, dtype=inp["dtype"]).reshape(inp["shape"])
        interp.set_tensor(inp["index"], arr)
        interp.invoke()
        return interp.get_tensor(out["index"]), backend

    def _read_model_bytes_with_cache(
        model_id,
        file_name,
        expected_sha256,
        model_store_dir,
        model_base_url,
        cache_root,
    ):
        cache_root = cache_root or ".worker_model_cache"
        model_cache_dir = os.path.join(cache_root, model_id)
        os.makedirs(model_cache_dir, exist_ok=True)
        cache_path = os.path.join(model_cache_dir, file_name)

        def _valid(data):
            if not expected_sha256:
                return True
            return hashlib.sha256(data).hexdigest() == expected_sha256

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                data = fh.read()
            if _valid(data):
                return data, "cache"

        if model_store_dir:
            shared = os.path.join(model_store_dir, model_id, file_name)
            if os.path.exists(shared):
                with open(shared, "rb") as fh:
                    data = fh.read()
                if _valid(data):
                    with open(cache_path, "wb") as out:
                        out.write(data)
                    return data, "shared"

        if model_base_url:
            url = f"{model_base_url.rstrip('/')}/{model_id}/{file_name}"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            if _valid(data):
                with open(cache_path, "wb") as out:
                    out.write(data)
                return data, "download"

        raise FileNotFoundError(
            f"Unable to load model file '{file_name}' for model_id='{model_id}'"
        )

    start = _time.time()
    progress_percent = 0.0

    upstream = task_input.get("upstream_results", {})
    original_args = task_input.get("original_args", {})

    partition_idx = original_args.get("partition_idx", 0)
    stream_id = original_args.get("stream_id", 0)

    model_id = None
    model_store_dir = None
    model_manifest = {}
    model_base_url = None
    worker_model_cache_dir = ".worker_model_cache"
    input_blob = None
    dtype = "float32"
    trace = []

    for _, result in upstream.items():
        if isinstance(result, str):
            try:
                result = _json.loads(result)
            except Exception:
                import ast as _ast

                result = _ast.literal_eval(result)
        if not isinstance(result, dict):
            continue

        rid = result.get("stream_id")
        if rid is not None and rid != stream_id:
            continue

        if result.get("trace") and isinstance(result.get("trace"), list):
            trace = list(result["trace"])

        if result.get("model_id"):
            model_id = result["model_id"]
        if result.get("model_store_dir"):
            model_store_dir = result["model_store_dir"]
        if result.get("model_manifest"):
            model_manifest = result["model_manifest"]
        if result.get("model_base_url"):
            model_base_url = result["model_base_url"]
        if result.get("worker_model_cache_dir"):
            worker_model_cache_dir = result["worker_model_cache_dir"]

        if partition_idx == 0 and result.get("input_tensor") and input_blob is None:
            input_blob = result["input_tensor"]
            dtype = result.get("dtype", dtype)

        if partition_idx > 0 and result.get("output_tensor") is not None:
            if result.get("partition_idx") == partition_idx - 1:
                input_blob = result["output_tensor"]
                dtype = result.get("dtype", dtype)
        elif partition_idx > 0 and result.get("result_file_url"):
            if result.get("partition_idx") in (None, partition_idx - 1):
                input_blob = {
                    "format": "npy",
                    "file_url": result["result_file_url"],
                }
                dtype = result.get("dtype", dtype)

    if input_blob is None and partition_idx > 0:
        for _, result in upstream.items():
            if isinstance(result, dict) and result.get("output_tensor") is not None:
                input_blob = result["output_tensor"]
                break
            if isinstance(result, dict) and result.get("result_file_url"):
                input_blob = {
                    "format": "npy",
                    "file_url": result["result_file_url"],
                }
                break

    if input_blob is None:
        return {
            "error": f"No input tensor for partition {partition_idx}",
            "stream_id": stream_id,
            "partition_idx": partition_idx,
            "trace": trace,
        }

    progress_percent = 20.0
    x = _decode(input_blob)
    output = None
    backend = "none"

    tensor_ref = task_input.get("tensor_ref")
    if tensor_ref is None and isinstance(original_args, dict):
        tensor_ref = original_args.get("tensor_ref")
    if tensor_ref is not None:
        try:
            output = _load_tensor_ref(tensor_ref)
            backend = "tensor_ref"
        except Exception as exc:
            return {
                "error": f"Failed to load tensor_ref: {exc}",
                "stream_id": stream_id,
                "partition_idx": partition_idx,
                "backend_used": "tensor_ref_error",
                "trace": trace,
            }

    info = (model_manifest.get("parts") or {}).get(str(partition_idx), {})
    if output is None and model_id and model_store_dir and info.get("file"):
        try:
            model_bytes, model_source = _read_model_bytes_with_cache(
                model_id=model_id,
                file_name=info["file"],
                expected_sha256=info.get("sha256"),
                model_store_dir=model_store_dir,
                model_base_url=model_base_url,
                cache_root=worker_model_cache_dir,
            )
            output, backend = _run_tflite(base64.b64encode(model_bytes).decode("utf-8"), x)
            trace.append(
                {
                    "stream_id": stream_id,
                    "partition_idx": partition_idx,
                    "model_source": model_source,
                }
            )
        except Exception:
            output = None

    progress_percent = 80.0
    if output is None:
        error_message = "No real backend output. Refusing simulation fallback."
        if not info.get("file"):
            error_message += f" Missing model manifest entry for partition {partition_idx}."
        else:
            error_message += " Model load or TFLite backend failed."
        return {
            "error": error_message,
            "stream_id": stream_id,
            "partition_idx": partition_idx,
            "backend_used": backend,
            "model_id": model_id,
            "trace": trace,
        }

    output_blob = _encode(output)
    out_shape = list(getattr(output, "shape", []))
    elapsed = _time.time() - start
    progress_percent = 100.0

    trace.append(
        {
            "stream_id": stream_id,
            "partition_idx": partition_idx,
            "backend_used": backend,
            "output_shape": out_shape,
            "exec_time": elapsed,
        }
    )

    return {
        "stream_id": stream_id,
        "partition_idx": partition_idx,
        "output_tensor": output_blob,
        "output_shape": out_shape,
        "exec_time": elapsed,
        "backend_used": backend,
        "dtype": dtype,
        "model_id": model_id,
        "model_store_dir": model_store_dir,
        "model_manifest": model_manifest,
        "model_base_url": model_base_url,
        "worker_model_cache_dir": worker_model_cache_dir,
        "trace": trace,
    }


@CROWDio.task()
def classify_with_trace(task_input):
    import base64
    import io
    import json as _json
    import time as _time

    import numpy as np
    import requests

    def _decode(blob):
        if isinstance(blob, str):
            try:
                blob = _json.loads(blob)
            except Exception:
                import ast as _ast

                blob = _ast.literal_eval(blob)
        if blob.get("format", "json") == "npy":
            data_b64 = blob.get("data_b64")
            if data_b64:
                raw = base64.b64decode(data_b64)
            else:
                file_url = blob.get("file_url")
                if not file_url:
                    raise ValueError("No tensor data_b64 or file_url in output_tensor")
                response = requests.get(file_url, timeout=30)
                response.raise_for_status()
                raw = response.content
            return np.load(io.BytesIO(raw), allow_pickle=False)
        arr = np.array(blob.get("data", []), dtype=blob.get("dtype", "float32"))
        if blob.get("shape"):
            arr = arr.reshape(blob["shape"])
        return arr

    start = _time.time()
    upstream = task_input.get("upstream_results", {})
    stream_id = task_input.get("original_args", {}).get("stream_id", 0)

    trace = []
    x = None
    x_shape = None

    for _, result in upstream.items():
        if isinstance(result, str):
            try:
                result = _json.loads(result)
            except Exception:
                import ast as _ast

                result = _ast.literal_eval(result)
        if not isinstance(result, dict):
            continue

        rid = result.get("stream_id")
        if rid is not None and rid != stream_id:
            continue

        if result.get("trace") and isinstance(result.get("trace"), list):
            trace = list(result["trace"])

        if result.get("output_tensor"):
            x = _decode(result["output_tensor"])
            x_shape = result.get("output_shape", [])
        elif result.get("result_file_url"):
            x = _decode({"format": "npy", "file_url": result["result_file_url"]})
            x_shape = result.get("output_shape", x_shape or [])

    if x is None:
        return {
            "error": "No logits/output found for classification",
            "stream_id": stream_id,
            "trace": trace,
        }

    arr = np.asarray(x).reshape(-1)
    exp = np.exp(arr - arr.max())
    probs = exp / exp.sum()

    top5_idx = probs.argsort()[-5:][::-1].tolist()
    top5 = [{"class": int(i), "prob": float(probs[i])} for i in top5_idx]
    pred = int(top5_idx[0])
    conf = float(probs[pred])

    elapsed = _time.time() - start
    return {
        "stream_id": stream_id,
        "predicted_class": pred,
        "confidence": conf,
        "top5": top5,
        "fused_shape": x_shape,
        "classify_time": elapsed,
        "trace": trace,
    }


def _compute_reference(stage0_payload, num_devices):
    x = _deserialize_tensor(stage0_payload["input_tensor"])
    model_dir = os.path.join(stage0_payload["model_store_dir"], stage0_payload["model_id"])
    manifest = stage0_payload["model_manifest"]
    for i in range(num_devices):
        part_info = manifest["parts"][str(i)]
        with open(os.path.join(model_dir, part_info["file"]), "rb") as fh:
            x = _run_tflite_local_bytes(fh.read(), x)

    import numpy as np

    arr = np.asarray(x).reshape(-1)
    exp = np.exp(arr - arr.max())
    probs = exp / probs.sum() if (probs := exp).sum() else exp
    top5_idx = probs.argsort()[-5:][::-1].tolist()
    pred = int(top5_idx[0])
    conf = float(probs[pred])

    return {
        "predicted_class": pred,
        "confidence": conf,
        "top5": [{"class": int(i), "prob": float(probs[i])} for i in top5_idx],
    }


async def run_single_job(args):
    payload = _build_stage0_payload(
        args.num_devices,
        args.dtype,
        args.weights,
        args.model_store_dir,
        args.model_base_url,
        args.worker_model_cache_dir,
    )

    stages = [
        {"func": emit_payload, "args_list": [payload], "name": "emit_payload"}
    ]
    for p in range(args.num_devices):
        stages.append(
            {
                "func": run_tflite_partition,
                "args_list": [
                    {"partition_idx": p, "stream_id": s}
                    for s in range(args.tasks_per_stage)
                ],
                "pass_upstream_results": True,
                "name": f"tflite_partition_{p}",
            }
        )

    stages.append(
        {
            "func": classify_with_trace,
            "args_list": [{"stream_id": s} for s in range(args.tasks_per_stage)],
            "pass_upstream_results": True,
            "name": "classify",
        }
    )

    t0 = time.time()
    results = await crowdio_pipeline(stages)
    pipeline_wall = time.time() - t0

    parsed = []
    for r in results or []:
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                import ast

                r = ast.literal_eval(r)
        parsed.append(r)

    return parsed, pipeline_wall, payload


def parse_args():
    p = argparse.ArgumentParser(description="Famous DNN multi-device partition test")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument(
        "--num-devices",
        "--num-partitions",
        dest="num_devices",
        type=int,
        default=2,
        help="Number of devices/partitions (>=2)",
    )
    p.add_argument("--tasks-per-stage", type=int, default=4)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--weights", choices=["none", "imagenet"], default="none")
    p.add_argument(
        "--model-store-dir",
        default=os.path.join(".", ".model_store"),
        help="Shared directory containing partitioned model files",
    )
    p.add_argument(
        "--model-base-url",
        default=None,
        help="Optional base URL for workers to download model files: <base>/<model_id>/<partition>.tflite",
    )
    p.add_argument(
        "--worker-model-cache-dir",
        default=os.path.join(".", ".worker_model_cache"),
        help="Worker-local cache directory for downloaded/copied model files",
    )
    p.add_argument(
        "--output-json",
        default=os.path.join(
            os.path.dirname(__file__), "dnn_output", "famous_dnn_multidevice_result.json"
        ),
    )

    args = p.parse_args()
    if args.num_devices < 2:
        p.error("--num-devices must be >= 2")
    if args.tasks_per_stage < 1:
        p.error("--tasks-per-stage must be >= 1")
    return args


async def main():
    args = parse_args()

    print("=" * 72)
    print("FAMOUS DNN MULTI-DEVICE TEST (VGG16)")
    print("=" * 72)
    print(f"Foreman            : {args.host}:{args.port}")
    print(f"Model              : vgg16")
    print(f"Weights            : {args.weights}")
    print("Input size         : 224x224x3")
    print(f"Devices per job    : {args.num_devices}")
    print(f"Tasks per stage    : {args.tasks_per_stage}")

    await crowdio_connect(args.host, args.port)
    try:
        started = time.time()
        results, pipeline_wall, stage0_payload = await run_single_job(args)
        total_wall = time.time() - started
    finally:
        await crowdio_disconnect()

    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    backend_counter = Counter()
    for r in ok:
        for step in r.get("trace", []):
            backend_counter[step.get("backend_used", "unknown")] += 1

    accuracy = None
    if ok:
        try:
            ref = _compute_reference(stage0_payload, args.num_devices)
            ref_class = ref["predicted_class"]
            ref_top5 = {item["class"] for item in ref["top5"]}

            top1_hits = 0
            top5_hits = 0
            for r in ok:
                if r.get("predicted_class") == ref_class:
                    top1_hits += 1
                s5 = {item.get("class") for item in r.get("top5", [])}
                if ref_class in s5 or len(ref_top5.intersection(s5)) > 0:
                    top5_hits += 1

            accuracy = {
                "mode": "reference_agreement",
                "reference_predicted_class": ref_class,
                "streams_evaluated": len(ok),
                "top1_matches": top1_hits,
                "top1_accuracy": top1_hits / len(ok),
                "top5_matches": top5_hits,
                "top5_accuracy": top5_hits / len(ok),
            }
        except Exception as exc:
            accuracy = {"mode": "reference_agreement", "error": str(exc)}

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Successful streams : {len(ok)}/{len(results)}")
    print(f"Failed streams     : {len(failed)}")
    print(f"Pipeline wall time : {pipeline_wall:.2f}s")
    print(f"Total wall time    : {total_wall:.2f}s")
    print("Backend usage      : " + ", ".join(f"{k}={v}" for k, v in backend_counter.items()))
    if accuracy and "error" not in accuracy:
        print(f"Top-1 accuracy     : {accuracy['top1_accuracy']:.4f} ({accuracy['top1_matches']}/{accuracy['streams_evaluated']})")
        print(f"Top-5 accuracy     : {accuracy['top5_accuracy']:.4f} ({accuracy['top5_matches']}/{accuracy['streams_evaluated']})")
    elif accuracy and "error" in accuracy:
        print(f"Accuracy check     : {accuracy['error']}")

    if failed:
        print("\nFailures:")
        for r in failed[:10]:
            print(f"  stream={r.get('stream_id')} error={r.get('error')}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    out = {
        "config": vars(args),
        "successful_streams": len(ok),
        "failed_streams": len(failed),
        "total_streams": len(results),
        "pipeline_wall_time": pipeline_wall,
        "total_wall_time": total_wall,
        "backend_usage": dict(backend_counter),
        "accuracy": accuracy,
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nSaved summary: {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
