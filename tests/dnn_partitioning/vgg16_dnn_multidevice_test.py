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
import io
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import connect, disconnect, crowdio, pipeline


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
        raw = base64.b64decode(blob["data_b64"])
        return np.load(io.BytesIO(raw))
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


def _build_stage0_payload(num_devices: int, dtype: str, weights: str):
    np = _require_numpy()

    tflite_models = _build_vgg16_tflite_partitions(num_devices, weights)

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
        "tflite_models": {str(k): v for k, v in tflite_models.items()},
    }


@crowdio.task()
def emit_payload(payload):
    return payload


@crowdio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["partition_idx", "progress_percent"],
)
def run_tflite_partition(task_input):
    import base64
    import io
    import json as _json
    import time as _time

    import numpy as np

    def _decode(blob):
        if isinstance(blob, str):
            try:
                blob = _json.loads(blob)
            except Exception:
                import ast as _ast

                blob = _ast.literal_eval(blob)
        if blob.get("format", "json") == "npy":
            raw = base64.b64decode(blob["data_b64"])
            return np.load(io.BytesIO(raw))
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

    start = _time.time()
    progress_percent = 0.0

    upstream = task_input.get("upstream_results", {})
    original_args = task_input.get("original_args", {})

    partition_idx = original_args.get("partition_idx", 0)
    stream_id = original_args.get("stream_id", 0)

    tflite_models = {}
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

        if result.get("tflite_models"):
            tflite_models = result["tflite_models"]

        if partition_idx == 0 and result.get("input_tensor") and input_blob is None:
            input_blob = result["input_tensor"]
            dtype = result.get("dtype", dtype)

        if partition_idx > 0 and result.get("output_tensor") is not None:
            if result.get("partition_idx") == partition_idx - 1:
                input_blob = result["output_tensor"]
                dtype = result.get("dtype", dtype)

    if input_blob is None and partition_idx > 0:
        for _, result in upstream.items():
            if isinstance(result, dict) and result.get("output_tensor") is not None:
                input_blob = result["output_tensor"]
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
    backend = "simulation"

    info = tflite_models.get(str(partition_idx))
    if info and info.get("tflite_b64"):
        output, backend = _run_tflite(info["tflite_b64"], x)

    progress_percent = 80.0
    if output is None:
        out_shape = (1,) + tuple(info.get("output_shape", [1000]))
        output = np.random.default_rng(partition_idx).standard_normal(out_shape).astype(dtype)

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
        "tflite_models": tflite_models,
        "trace": trace,
    }


@crowdio.task()
def classify_with_trace(task_input):
    import base64
    import io
    import json as _json
    import time as _time

    import numpy as np

    def _decode(blob):
        if isinstance(blob, str):
            try:
                blob = _json.loads(blob)
            except Exception:
                import ast as _ast

                blob = _ast.literal_eval(blob)
        if blob.get("format", "json") == "npy":
            raw = base64.b64decode(blob["data_b64"])
            return np.load(io.BytesIO(raw))
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
    models = stage0_payload["tflite_models"]
    for i in range(num_devices):
        x = _run_tflite_local(models[str(i)]["tflite_b64"], x)

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
    payload = _build_stage0_payload(args.num_devices, args.dtype, args.weights)

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
    results = await pipeline(stages)
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

    await connect(args.host, args.port)
    try:
        started = time.time()
        results, pipeline_wall, stage0_payload = await run_single_job(args)
        total_wall = time.time() - started
    finally:
        await disconnect()

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
