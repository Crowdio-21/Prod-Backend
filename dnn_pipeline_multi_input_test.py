"""
Multi-input DNN pipeline test for verifying multi-worker streaming behaviour.

Unlike the single-input smoke test, this script sends N separate inputs
through the 3-stage ONNX pipeline.  With pipeline_mode="streaming", each
input flows independently:

    Input-0:  cell_a → cell_b → cell_c
    Input-1:  cell_a → cell_b → cell_c
    Input-2:  cell_a → cell_b → cell_c
              ↑                       ↑
       Worker-A starts here     Worker-B can start cell_a for Input-1
       while Worker-A moves      while Input-0 is still in cell_b

Expected with 2+ workers:
  - Multiple workers process different inputs at different stages concurrently.
  - Model affinity scheduler prefers the worker that already has a model loaded.
  - ONNX session caching on mobile means no session re-creation between inputs.
  - from_cache skips model re-download on repeated runs.

Run:
    uv run python dnn_pipeline_multi_input_test.py
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from common.serializer import deserialize_tensor, serialize_tensor
from developer_sdk import connect, disconnect, dnn_pipeline

DEFAULT_MODELS = [
    str(Path("tools") / "onnx_partitions_sentiment" / "cell_a.onnx"),
    str(Path("tools") / "onnx_partitions_sentiment" / "cell_b.onnx"),
    str(Path("tools") / "onnx_partitions_sentiment" / "cell_c.onnx"),
]
DEFAULT_TOKENIZER_DIR = str(Path("tools") / "onnx_partitions_sentiment" / "tokenizer")

# ── Texts to classify (each becomes its own pipeline input) ──────────
DEFAULT_TEXTS = [
    "The movie was absolutely wonderful and heartwarming",
    "The food was terrible and the service was even worse",
    "I had an amazing experience at the new restaurant downtown",
]


def tokenize_single(text: str, tokenizer_dir: str, max_length: int) -> dict:
    """Tokenize one text and wrap it as a tensor_transport payload."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required. Install with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encoded = tokenizer(
        [text],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    return {
        "transport": "tensor_transport",
        "tensor_payload": serialize_tensor(ids, compression="zlib"),
    }


def extract_tensor_payload(node: Any) -> Optional[dict]:
    """Recursively find a tensor_transport payload in nested result data."""
    if isinstance(node, dict):
        if node.get("transport") == "tensor_transport" and isinstance(
            node.get("tensor_payload"), dict
        ):
            return node["tensor_payload"]
        for value in node.values():
            payload = extract_tensor_payload(value)
            if payload is not None:
                return payload
        return None
    if isinstance(node, list):
        for item in node:
            payload = extract_tensor_payload(item)
            if payload is not None:
                return payload
    return None


def softmax_numpy(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def classify_result(result: Any, text: str, index: int) -> None:
    """Print classification for one pipeline result."""
    tensor_payload = extract_tensor_payload(result)
    if tensor_payload is None:
        print(f"  Input {index}: tensor payload not found in result")
        return

    tensor = deserialize_tensor(tensor_payload)
    logits = np.asarray(tensor, dtype=np.float64)
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)

    probs = softmax_numpy(logits, axis=-1)[0]
    pred = int(np.argmax(probs))

    if probs.shape[0] >= 2:
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(
            f"  Input {index}: {sentiment} "
            f"(neg={float(probs[0]):.4f}, pos={float(probs[1]):.4f}) "
            f'"{text[:50]}..."'
        )
    else:
        print(f"  Input {index}: class={pred} probs={probs.tolist()}")


async def run(
    host: str,
    port: int,
    models: list[str],
    tokenizer_dir: str,
    max_length: int,
    texts: list[str],
) -> None:
    n = len(texts)
    print(f"Tokenizing {n} texts individually...")

    # Each text becomes a separate stage-0 input
    stage0_inputs = [tokenize_single(text, tokenizer_dir, max_length) for text in texts]

    await connect(host, port)
    try:
        print(f"Submitting pipeline: 3 stages × {n} inputs (streaming mode)")
        print(
            "With 2+ workers, different inputs should run on different workers concurrently.\n"
        )

        t0 = time.perf_counter()

        results = await dnn_pipeline(
            stages=[
                {
                    "name": "cell_a",
                    "model": models[0],
                    "args_list": stage0_inputs,
                },
                {
                    "name": "cell_b",
                    "model": models[1],
                    "args_list": [None] * n,
                    "pass_upstream_results": True,
                },
                {
                    "name": "cell_c",
                    "model": models[2],
                    "args_list": [None] * n,
                    "pass_upstream_results": True,
                },
            ],
            pipeline_mode="streaming",
        )

        elapsed = time.perf_counter() - t0

        # dnn_pipeline returns an aggregation wrapper dict, not a plain list.
        # Unwrap to get the per-input results.
        if isinstance(results, dict) and "raw_results" in results:
            raw_results = results["raw_results"]
        else:
            raw_results = results

        print(f"Pipeline completed in {elapsed:.2f}s")
        print(f"Got {len(raw_results)} results for {n} inputs\n")
        print("Classifications:")

        for i, (text, result) in enumerate(zip(texts, raw_results)):
            classify_result(result, text, i)

        print(f"\nTotal wall-clock time: {elapsed:.2f}s")
        if n > 1:
            print(f"Average per input: {elapsed / n:.2f}s")
            print(
                "\nTip: Compare this to running the same texts one-at-a-time with the\n"
                "single-input smoke test. If multi-worker parallelism is working,\n"
                "the wall-clock time should be significantly less than N × single-input time."
            )
    finally:
        await disconnect()


if __name__ == "__main__":
    host = "localhost"
    port = 9000
    models = DEFAULT_MODELS
    tokenizer_dir = DEFAULT_TOKENIZER_DIR
    max_length = 64
    texts = DEFAULT_TEXTS

    asyncio.run(
        run(
            host=host,
            port=port,
            models=models,
            tokenizer_dir=tokenizer_dir,
            max_length=max_length,
            texts=texts,
        )
    )
