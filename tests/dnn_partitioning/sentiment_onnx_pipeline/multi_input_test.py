#!/usr/bin/env python3
"""Run multiple independent texts through the streaming ONNX pipeline."""

from __future__ import annotations

import asyncio
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from common.serializer import deserialize_tensor, serialize_tensor
try:
    from crowdio import CROWDio, crowdio_connect, crowdio_disconnect, crowdio_pipeline
except ImportError:
    from developer_sdk import CROWDio, crowdio_connect, crowdio_disconnect, crowdio_pipeline

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS = [str(ROOT / "tools" / "onnx_partitions_sentiment" / name) for name in ("cell_a.onnx", "cell_b.onnx", "cell_c.onnx")]
DEFAULT_TOKENIZER_DIR = str(ROOT / "tools" / "onnx_partitions_sentiment" / "tokenizer")
DEFAULT_LIMIT = 700


def load_texts(dataset_path: str, text_column: str, limit: int) -> list[str]:
    """Load up to ``limit`` sentences from common local dataset formats."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        rows = path.read_text(encoding="utf-8").splitlines()
        return [row.strip() for row in rows if row.strip()][:limit]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or text_column not in rows.fieldnames:
                raise ValueError(
                    f"CSV must contain a '{text_column}' column; found {rows.fieldnames}"
                )
            return [
                value.strip()
                for row in rows
                if (value := str(row.get(text_column, ""))).strip()
            ][:limit]

    if suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("data", records.get("records", records.get("texts", [])))
    else:
        raise ValueError("Dataset must be .csv, .json, .jsonl, .ndjson, or .txt")

    texts = []
    for record in records:
        value = record if isinstance(record, str) else record.get(text_column, "")
        if str(value).strip():
            texts.append(str(value).strip())
        if len(texts) == limit:
            break
    if not texts:
        raise ValueError(f"No text rows found in {path} using column '{text_column}'")
    return texts


def tokenize_texts(texts: list[str], tokenizer_dir: str, max_length: int) -> list[dict]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encoded = tokenizer(texts, return_tensors="np", padding="max_length", truncation=True, max_length=max_length)
    return [
        {
            "transport": "tensor_transport",
            "tensor_payload": serialize_tensor(input_ids[None, :].astype(np.int64), compression="zlib"),
        }
        for input_ids in encoded["input_ids"]
    ]


def extract_tensor_payload(node: Any) -> Optional[dict]:
    if isinstance(node, dict):
        if node.get("transport") == "tensor_transport" and isinstance(node.get("tensor_payload"), dict):
            return node["tensor_payload"]
        for value in node.values():
            payload = extract_tensor_payload(value)
            if payload is not None:
                return payload
    elif isinstance(node, list):
        for item in node:
            payload = extract_tensor_payload(item)
            if payload is not None:
                return payload
    return None


def run_onnx_partition(stage_input: dict) -> dict:
    """Execute one ONNX partition on a worker and return a tensor payload."""
    import numpy as np
    import onnxruntime as ort
    from common.serializer import deserialize_tensor, serialize_tensor

    def find_payload(node):
        if isinstance(node, dict):
            if node.get("transport") == "tensor_transport" and isinstance(
                node.get("tensor_payload"), dict
            ):
                return node["tensor_payload"]
            for value in node.values():
                payload = find_payload(value)
                if payload is not None:
                    return payload
        elif isinstance(node, list):
            for item in node:
                payload = find_payload(item)
                if payload is not None:
                    return payload
        return None

    original_args = stage_input.get("original_args", stage_input)
    if not isinstance(original_args, dict):
        raise RuntimeError("Pipeline stage arguments must be a dictionary")

    model_path = original_args["model_path"]
    payload = original_args.get("tensor_payload")
    if payload is None:
        upstream = stage_input.get("upstream_results", {})
        if not upstream:
            raise RuntimeError("No upstream tensor payload was provided")
        payload = find_payload(next(iter(upstream.values())))
    if payload is None:
        raise RuntimeError("Could not find upstream tensor payload")

    tensor = deserialize_tensor(payload)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: np.asarray(tensor)})[0]
    return {
        "transport": "tensor_transport",
        "tensor_payload": serialize_tensor(np.asarray(output), compression="zlib"),
    }


def classify_result(result: Any, text: str, index: int) -> None:
    payload = extract_tensor_payload(result)
    if payload is None:
        print(f"  Input {index}: tensor payload not found")
        return
    logits = np.asarray(deserialize_tensor(payload), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)[0]
    probabilities /= np.sum(probabilities)
    prediction = int(np.argmax(probabilities))
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f'  Input {index}: {sentiment} (neg={probabilities[0]:.4f}, pos={probabilities[1]:.4f}) "{text[:50]}..."')


async def run(host: str, port: int, models: list[str], tokenizer_dir: str, max_length: int, texts: list[str]) -> None:
    stage0_inputs = [
        {"model_path": models[0], "tensor_payload": item["tensor_payload"]}
        for item in tokenize_texts(texts, tokenizer_dir, max_length)
    ]
    stage1_inputs = [{"model_path": models[1]} for _ in texts]
    stage2_inputs = [{"model_path": models[2]} for _ in texts]
    await crowdio_connect(host, port)
    try:
        started = time.perf_counter()
        results = await crowdio_pipeline(
            stages=[
                {"name": "cell_a", "func": run_onnx_partition, "args_list": stage0_inputs},
                {"name": "cell_b", "func": run_onnx_partition, "args_list": stage1_inputs, "pass_upstream_results": True},
                {"name": "cell_c", "func": run_onnx_partition, "args_list": stage2_inputs, "pass_upstream_results": True},
            ],
        )
        elapsed = time.perf_counter() - started
        raw_results = results.get("raw_results", results) if isinstance(results, dict) else results
        print(f"Pipeline completed in {elapsed:.2f}s; got {len(raw_results)} results for {len(texts)} inputs")
        for index, (text, result) in enumerate(zip(texts, raw_results)):
            classify_result(result, text, index)
    finally:
        await crowdio_disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="CSV, JSON, JSONL, or TXT dataset containing sentences")
    parser.add_argument("--text-column", default="text", help="Text column/key for CSV or JSON records")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of sentences to process (default: 700)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--max-length", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_texts = load_texts(args.dataset, args.text_column, args.limit)
    print(f"Loaded {len(dataset_texts)} sentences from {Path(args.dataset).resolve()}")
    asyncio.run(run(args.host, args.port, DEFAULT_MODELS, DEFAULT_TOKENIZER_DIR, args.max_length, dataset_texts))
