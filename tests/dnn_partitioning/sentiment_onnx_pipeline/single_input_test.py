#!/usr/bin/env python3
"""Run one text through the three ONNX partitions."""

from __future__ import annotations

import asyncio
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


def tokenize_texts(texts: list[str], tokenizer_dir: str, max_length: int) -> np.ndarray:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required when using text input") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encoded = tokenizer(texts, return_tensors="np", padding="max_length", truncation=True, max_length=max_length)
    return np.asarray(encoded["input_ids"], dtype=np.int64)


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


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


async def run(host: str, port: int, models: list[str], tokenizer_dir: str, max_length: int, texts: list[str]) -> None:
    input_ids = tokenize_texts(texts, tokenizer_dir, max_length)
    stage0_input = {"transport": "tensor_transport", "tensor_payload": serialize_tensor(input_ids, compression="zlib")}
    await crowdio_connect(host, port)
    try:
        results = await crowdio_pipeline(
            stages=[
                {"name": "cell_a", "model": models[0], "args_list": [stage0_input]},
                {"name": "cell_b", "model": models[1], "args_list": [None], "pass_upstream_results": True},
                {"name": "cell_c", "model": models[2], "args_list": [None], "pass_upstream_results": True},
            ],
            pipeline_mode="streaming",
        )
        payload = extract_tensor_payload(results)
        if payload is None:
            raise RuntimeError(f"Final tensor payload not found in result: {results!r}")
        logits = np.asarray(deserialize_tensor(payload), dtype=np.float64)
        if logits.ndim == 1:
            logits = np.expand_dims(logits, axis=0)
        probabilities = softmax_numpy(logits)[0]
        prediction = int(np.argmax(probabilities))
        print(f"Pipeline completed successfully; input shape={list(input_ids.shape)}")
        print(f"Logits shape={list(logits.shape)}, probabilities={probabilities.tolist()}")
        print(f"Prediction: {'positive' if prediction == 1 else 'negative'} (class={prediction})")
    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(run("localhost", 9000, DEFAULT_MODELS, DEFAULT_TOKENIZER_DIR, 64, ["The movie was absolutely garbage"]))
