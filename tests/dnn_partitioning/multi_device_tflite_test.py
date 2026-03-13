#!/usr/bin/env python3
"""
Multi-device TFLite partition inference test for CROWDio.

This test is designed to validate distributed execution across multiple workers:
- Build partitioned TFLite sub-models on the client machine (TensorFlow required)
- Dispatch multiple tasks under each stage inside a single pipeline job
- Collect per-partition backend telemetry (tflite_runtime / tensorflow.lite / simulation)

Typical usage:
    uv run python tests/dnn_partitioning/multi_device_tflite_test.py \
        --host localhost --port 9000 --num-devices 3 --jobs 4 --concurrency 2
        
        uv run python tests/dnn_partitioning/multi_device_tflite_test.py --host localhost --port 9000 --num-devices 2 --jobs 1 --concurrency 1 --tasks-per-stage 4
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
        raise RuntimeError("numpy is required for this test") from exc


def _require_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required on the client to build TFLite partitions"
        ) from exc


def _serialize_tensor_client(tensor):
    np = _require_numpy()
    buf = io.BytesIO()
    arr = np.asarray(tensor)
    np.save(buf, arr)
    return {
        "format": "npy",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
    }


def _deserialize_tensor_client(blob):
    np = _require_numpy()
    if blob.get("format", "json") == "npy":
        raw = base64.b64decode(blob["data_b64"])
        return np.load(io.BytesIO(raw))
    arr = np.array(blob.get("data", []), dtype=blob.get("dtype", "float32"))
    if blob.get("shape"):
        arr = arr.reshape(blob["shape"])
    return arr


def _run_tflite_client(tflite_b64_str, input_array):
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


def _compute_reference_prediction(stage0_payload, num_devices):
    """
    Compute a deterministic local reference prediction from the same partitioned
    TFLite chain used by distributed tasks.
    """
    np = _require_numpy()

    x = _deserialize_tensor_client(stage0_payload["input_tensor"])
    tflite_models = stage0_payload["tflite_models"]

    for i in range(num_devices):
        info = tflite_models[str(i)]
        x = _run_tflite_client(info["tflite_b64"], x)

    feature = np.asarray(x)
    if feature.ndim == 4:
        gap = feature.mean(axis=(1, 2)).flatten()
    else:
        gap = feature.flatten()

    num_classes = 1000
    rng = np.random.RandomState(0)
    W = rng.randn(len(gap), num_classes).astype(np.float32) * 0.01
    logits = gap @ W
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    top5_idx = probs.argsort()[-5:][::-1].tolist()
    pred_class = int(top5_idx[0])
    confidence = float(probs[pred_class])
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top5": [{"class": int(i), "prob": float(probs[i])} for i in top5_idx],
    }


def _build_tflite_sub_models_client(num_partitions, input_size, num_classes=1000):
    tf = _require_tensorflow()

    partitions = {}
    h, w, c = input_size, input_size, 3

    for i in range(num_partitions):
        filters = min(64 * (2 ** i), 512)
        stride = 2 if i < num_partitions - 1 else 1
        is_last = i == num_partitions - 1

        model_inp = tf.keras.Input(shape=(h, w, c), batch_size=1)
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

        partitions[i] = {
            "tflite_b64": base64.b64encode(tflite_bytes).decode("utf-8"),
            "input_shape": [h, w, c],
            "output_shape": list(model.output_shape[1:]),
        }

        h = (h + stride - 1) // stride
        w = (w + stride - 1) // stride
        c = filters

    return partitions


def _build_stage0_payload(model_name, input_size, num_partitions, dtype="float32"):
    np = _require_numpy()

    shape = (1, input_size, input_size, 3)
    rng = np.random.RandomState(42)
    input_array = rng.randn(*shape).astype(dtype)

    tflite_models = _build_tflite_sub_models_client(num_partitions, input_size)

    layers = [
        {
            "layer_id": f"partition_{i}",
            "op": "tflite_partition",
            "partition_idx": i,
            "output_spatial": tflite_models[i]["output_shape"][0]
            if len(tflite_models[i]["output_shape"]) >= 2
            else 1,
        }
        for i in range(num_partitions)
    ]
    edges = [(f"partition_{i - 1}", f"partition_{i}") for i in range(1, num_partitions)]

    return {
        "model": model_name,
        "input_size": input_size,
        "num_partitions": num_partitions,
        "dtype": dtype,
        "layers": layers,
        "edges": edges,
        "input_tensor": _serialize_tensor_client(input_array),
        "plan": {f"partition_{i}": i for i in range(num_partitions)},
        "tflite_models": {str(k): v for k, v in tflite_models.items()},
    }


@crowdio.task()
def emit_payload(payload):
    """Simple stage-0 carrier task to inject prebuilt partition payload."""
    return payload


@crowdio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["partition_idx", "progress_percent"],
)
def run_tflite_partition_traced(task_input):
    """Run one TFLite partition and carry trace metadata forward."""
    import base64
    import io
    import json as _json
    import time as _time

    import numpy as np

    def _decode_tensor(blob):
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

    def _encode_tensor(arr):
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

            tflite_mod = importlib.import_module("tflite_runtime.interpreter")
            interp_cls = tflite_mod.Interpreter
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
    partition_idx = (
        original_args.get("partition_idx", 0)
        if isinstance(original_args, dict)
        else int(original_args or 0)
    )
    stream_id = (
        original_args.get("stream_id", 0)
        if isinstance(original_args, dict)
        else 0
    )

    if not upstream:
        return {"error": "No upstream results", "partition_idx": partition_idx}

    tflite_models = {}
    input_tensor_blob = None
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

        result_stream_id = result.get("stream_id")
        if result_stream_id is not None and result_stream_id != stream_id:
            continue

        if result.get("trace") and isinstance(result.get("trace"), list):
            trace = list(result["trace"])

        if result.get("tflite_models"):
            tflite_models = result["tflite_models"]

        if partition_idx == 0 and result.get("input_tensor") and input_tensor_blob is None:
            input_tensor_blob = result["input_tensor"]
            dtype = result.get("dtype", dtype)

        if partition_idx > 0 and result.get("output_tensor") is not None:
            if result.get("partition_idx") == partition_idx - 1:
                input_tensor_blob = result["output_tensor"]
                dtype = result.get("dtype", dtype)

    if input_tensor_blob is None and partition_idx > 0:
        for _, result in upstream.items():
            if isinstance(result, dict) and result.get("output_tensor") is not None:
                input_tensor_blob = result["output_tensor"]
                break

    if input_tensor_blob is None:
        return {
            "error": f"No input tensor found for partition {partition_idx}",
            "partition_idx": partition_idx,
            "stream_id": stream_id,
            "upstream_keys": list(upstream.keys()),
            "trace": trace,
        }

    progress_percent = 20.0
    input_array = _decode_tensor(input_tensor_blob)
    output = None
    backend_used = "simulation"

    tflite_info = tflite_models.get(str(partition_idx))
    if tflite_info and tflite_info.get("tflite_b64"):
        output, backend_used = _run_tflite(tflite_info["tflite_b64"], input_array)

    progress_percent = 80.0
    if output is None:
        out_shape_info = (tflite_info or {}).get("output_shape", [])
        if out_shape_info:
            out_shape = tuple([1] + [int(d) for d in out_shape_info])
        else:
            out_shape = (1, 56, 56, 64)
        output = np.random.default_rng(partition_idx).standard_normal(out_shape).astype(dtype)

    output_blob = _encode_tensor(output)
    out_shape_list = list(getattr(output, "shape", []))
    elapsed = _time.time() - start
    progress_percent = 100.0

    trace.append(
        {
            "partition_idx": partition_idx,
            "stream_id": stream_id,
            "backend_used": backend_used,
            "output_shape": out_shape_list,
            "exec_time": elapsed,
        }
    )

    return {
        "partition_idx": partition_idx,
        "stream_id": stream_id,
        "output_tensor": output_blob,
        "output_shape": out_shape_list,
        "exec_time": elapsed,
        "backend_used": backend_used,
        "tflite_models": tflite_models,
        "dtype": dtype,
        "trace": trace,
    }


@crowdio.task()
def classify_with_trace(task_input):
    """Final stage classifier that also returns partition trace metadata."""
    import base64
    import io
    import json as _json
    import time as _time

    import numpy as np

    def _decode_tensor(blob):
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
    original_args = task_input.get("original_args", {})
    stream_id = (
        original_args.get("stream_id", 0)
        if isinstance(original_args, dict)
        else 0
    )
    if not upstream:
        return {
            "error": "No upstream results for classification",
            "stream_id": stream_id,
            "trace": [],
        }

    fused_tensor = None
    fused_shape = None
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

        result_stream_id = result.get("stream_id")
        if result_stream_id is not None and result_stream_id != stream_id:
            continue

        if result.get("trace") and isinstance(result.get("trace"), list):
            trace = list(result["trace"])

        if "fused_tensor" in result:
            fused_tensor = _decode_tensor(result["fused_tensor"])
            fused_shape = result.get("fused_shape", [])
        elif "output_tensor" in result:
            fused_tensor = _decode_tensor(result["output_tensor"])
            fused_shape = result.get("output_shape", [])

    if fused_tensor is None:
        return {
            "error": "No feature map found for classification",
            "stream_id": stream_id,
            "trace": trace,
        }

    feature = np.asarray(fused_tensor)
    if feature.ndim == 4:
        gap = feature.mean(axis=(1, 2)).flatten()
    else:
        gap = feature.flatten()

    num_classes = 1000
    rng = np.random.RandomState(0)
    W = rng.randn(len(gap), num_classes).astype(np.float32) * 0.01
    logits = gap @ W

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    top5_idx = probs.argsort()[-5:][::-1].tolist()
    top5 = [{"class": int(i), "prob": float(probs[i])} for i in top5_idx]
    pred_class = int(top5_idx[0])
    confidence = float(probs[pred_class])

    elapsed = _time.time() - start
    return {
        "stream_id": stream_id,
        "predicted_class": pred_class,
        "confidence": confidence,
        "top5": top5,
        "fused_shape": fused_shape,
        "classify_time": elapsed,
        "trace": trace,
    }


async def run_single_pipeline_job(args):
    """
    Submit one pipeline job with multiple tasks under each stage.

    "tasks_per_stage" controls the number of parallel streams processed inside
    the same job. Every stream follows partition_0 -> ... -> partition_N-1 and
    is disambiguated using stream_id.
    """
    stage0_payload = _build_stage0_payload(
        args.model, args.input_size, args.num_devices, args.dtype
    )

    # Send heavyweight stage-0 payload only once; downstream stream tasks share it.
    emit_args = [stage0_payload]
    stages = [
        {
            "func": emit_payload,
            "args_list": emit_args,
            "name": "emit_payload",
        }
    ]
    for i in range(args.num_devices):
        stages.append(
            {
                "func": run_tflite_partition_traced,
                "args_list": [
                    {"partition_idx": i, "stream_id": s}
                    for s in range(args.tasks_per_stage)
                ],
                "pass_upstream_results": True,
                "name": f"tflite_partition_{i}",
            }
        )

    stages.append(
        {
            "func": classify_with_trace,
            "args_list": [{"stream_id": s} for s in range(args.tasks_per_stage)],
            "pass_upstream_results": True,
            "name": "classify_with_trace",
        }
    )

    t0 = time.time()
    results = await pipeline(stages)
    wall = time.time() - t0

    parsed = []
    for r in results or []:
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:
                import ast

                r = ast.literal_eval(r)
        parsed.append(r)

    return parsed, wall, stage0_payload


async def run_pipeline_jobs(args):
    """Run one or more pipeline jobs with bounded concurrency."""
    semaphore = asyncio.Semaphore(args.concurrency)

    async def _run_one(job_index):
        async with semaphore:
            parsed, wall, stage0_payload = await run_single_pipeline_job(args)
            for item in parsed:
                if isinstance(item, dict):
                    item.setdefault("job_index", job_index)
            return {
                "job_index": job_index,
                "results": parsed,
                "wall": wall,
                "stage0_payload": stage0_payload,
                "error": None,
            }

    tasks = [asyncio.create_task(_run_one(i)) for i in range(args.jobs)]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    per_job_wall = []
    stage0_payload = None

    for idx, outcome in enumerate(outcomes):
        if isinstance(outcome, Exception):
            all_results.append(
                {
                    "job_index": idx,
                    "error": f"pipeline submission failed: {outcome}",
                }
            )
            continue

        if stage0_payload is None:
            stage0_payload = outcome["stage0_payload"]

        per_job_wall.append(outcome["wall"])
        all_results.extend(outcome["results"])

    return all_results, per_job_wall, stage0_payload


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-device TFLite partition test with distributed workers"
    )
    p.add_argument("--host", default="localhost", help="Foreman host")
    p.add_argument("--port", type=int, default=9000, help="Foreman port")
    p.add_argument("--model", default="resnet", choices=["resnet", "inception"])
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument(
        "--num-devices",
        "--num-partitions",
        dest="num_devices",
        type=int,
        default=3,
        help="Number of devices/partitions per job (must be >= 2)",
    )
    p.add_argument("--dtype", default="float32")
    p.add_argument(
        "--tasks-per-stage",
        type=int,
        default=6,
        help="Number of parallel tasks under each stage in one pipeline job",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of pipeline jobs to submit",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of in-flight pipeline jobs",
    )
    p.add_argument(
        "--output-json",
        default=os.path.join(
            os.path.dirname(__file__),
            "dnn_output",
            "multi_device_tflite_test_result.json",
        ),
    )
    p.add_argument(
        "--require-non-simulation",
        action="store_true",
        help="Fail if any partition fell back to simulation backend",
    )
    p.add_argument(
        "--skip-accuracy-check",
        action="store_true",
        help="Skip local reference accuracy/agreement calculation",
    )

    args = p.parse_args()

    if args.num_devices < 2:
        p.error("--num-devices must be >= 2")
    if args.tasks_per_stage < 1:
        p.error("--tasks-per-stage must be >= 1")
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")

    return args


async def main():
    args = parse_args()

    print("=" * 72)
    print("MULTI-DEVICE TFLITE PARTITION TEST")
    print("=" * 72)
    print(f"Foreman            : {args.host}:{args.port}")
    print(f"Model              : {args.model}")
    print(f"Input size         : {args.input_size}x{args.input_size}x3")
    print(f"Devices per job    : {args.num_devices}")
    print(f"Tasks per stage    : {args.tasks_per_stage}")
    print(f"Jobs               : {args.jobs}")
    print(f"Concurrency        : {args.concurrency}")

    await connect(args.host, args.port)
    try:
        started = time.time()
        results, per_job_wall, stage0_payload = await run_pipeline_jobs(args)
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
    if not args.skip_accuracy_check and ok and stage0_payload is not None:
        try:
            ref = _compute_reference_prediction(stage0_payload, args.num_devices)
            ref_class = ref["predicted_class"]
            ref_top5 = {item["class"] for item in ref["top5"]}

            top1_hits = 0
            top5_hits = 0
            for r in ok:
                pred = r.get("predicted_class")
                if pred == ref_class:
                    top1_hits += 1
                stream_top5 = {item.get("class") for item in r.get("top5", [])}
                if ref_class in stream_top5 or len(ref_top5.intersection(stream_top5)) > 0:
                    top5_hits += 1

            accuracy = {
                "mode": "reference_agreement",
                "reference_predicted_class": ref_class,
                "reference_confidence": ref.get("confidence"),
                "streams_evaluated": len(ok),
                "top1_matches": top1_hits,
                "top1_accuracy": top1_hits / len(ok),
                "top5_matches": top5_hits,
                "top5_accuracy": top5_hits / len(ok),
            }
        except Exception as exc:
            accuracy = {
                "mode": "reference_agreement",
                "error": f"accuracy check failed: {exc}",
            }

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Successful streams : {len(ok)}/{len(results)}")
    print(f"Failed streams     : {len(failed)}")
    if per_job_wall:
        print(f"Pipeline wall time : min={min(per_job_wall):.2f}s max={max(per_job_wall):.2f}s avg={sum(per_job_wall)/len(per_job_wall):.2f}s")
    print(f"Total wall time    : {total_wall:.2f}s")
    print("Backend usage      : " + ", ".join(f"{k}={v}" for k, v in backend_counter.items()))
    if accuracy and "error" not in accuracy:
        print(f"Accuracy mode      : {accuracy['mode']}")
        print(f"Reference class    : {accuracy['reference_predicted_class']}")
        print(f"Top-1 accuracy     : {accuracy['top1_accuracy']:.4f} ({accuracy['top1_matches']}/{accuracy['streams_evaluated']})")
        print(f"Top-5 accuracy     : {accuracy['top5_accuracy']:.4f} ({accuracy['top5_matches']}/{accuracy['streams_evaluated']})")
    elif accuracy and "error" in accuracy:
        print(f"Accuracy check     : {accuracy['error']}")

    if failed:
        print("\nFailures:")
        for r in failed[:10]:
            print(f"  stream={r.get('stream_id')} error={r.get('error')}")

    if args.require_non_simulation and backend_counter.get("simulation", 0) > 0:
        raise RuntimeError(
            "Simulation fallback detected while --require-non-simulation was set"
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    summary = {
        "config": vars(args),
        "successful_streams": len(ok),
        "failed_streams": len(failed),
        "total_streams": len(results),
        "per_job_wall_time": per_job_wall,
        "total_wall_time": total_wall,
        "backend_usage": dict(backend_counter),
        "accuracy": accuracy,
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved summary: {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
