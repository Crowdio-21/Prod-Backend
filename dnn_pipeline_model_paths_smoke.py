import asyncio
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


def tokenize_texts(texts: list[str], tokenizer_dir: str, max_length: int) -> np.ndarray:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required when using --texts. Install with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encoded = tokenizer(
        texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return np.asarray(encoded["input_ids"], dtype=np.int64)


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
    """Numerically stable softmax with NumPy."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


async def run(
    host: str,
    port: int,
    models: list[str],
    tokenizer_dir: str,
    max_length: int,
    texts: list[str],
) -> None:
    input_ids = tokenize_texts(
        texts=texts,
        tokenizer_dir=tokenizer_dir,
        max_length=max_length,
    )

    stage0_input = {
        "transport": "tensor_transport",
        "tensor_payload": serialize_tensor(input_ids, compression="zlib"),
    }

    await connect(host, port)
    try:
        results = await dnn_pipeline(
            stages=[
                {
                    "name": "cell_a",
                    "model": models[0],
                    "args_list": [stage0_input],
                },
                {
                    "name": "cell_b",
                    "model": models[1],
                    "args_list": [None],
                    "pass_upstream_results": True,
                },
                {
                    "name": "cell_c",
                    "model": models[2],
                    "args_list": [None],
                    "pass_upstream_results": True,
                },
            ],
            pipeline_mode="streaming",
        )

        print("Pipeline completed successfully")
        print(f"Input IDs shape: {list(input_ids.shape)}")
        if texts is not None:
            print(f"Input texts count: {len(texts)}")
        final_tensor_payload = extract_tensor_payload(results)
        if final_tensor_payload is not None:
            final_tensor = deserialize_tensor(final_tensor_payload)
            print(f"Final tensor dtype: {final_tensor.dtype}")
            print(f"Final tensor shape: {list(final_tensor.shape)}")
            print("Final tensor values:")
            print(final_tensor)

            logits = np.asarray(final_tensor, dtype=np.float64)
            if logits.ndim == 1:
                logits = np.expand_dims(logits, axis=0)

            probs = softmax_numpy(logits, axis=-1)[0]
            pred = int(np.argmax(probs))

            if probs.shape[0] >= 2:
                negative_prob = float(probs[0])
                positive_prob = float(probs[1])
                sentiment = "positive" if pred == 1 else "negative"
                print(
                    f"Softmax probs: negative={negative_prob:.6f}, positive={positive_prob:.6f}"
                )
                print(f"Prediction: {sentiment} (class={pred})")
            else:
                print(f"Softmax probs: {probs.tolist()}")
                print(f"Prediction class: {pred}")
        else:
            print("Final tensor payload not found in results")
        print(results)
    finally:
        await disconnect()


if __name__ == "__main__":
    # Edit these variables directly, then run the script.
    host = "localhost"
    port = 9000
    models = DEFAULT_MODELS
    tokenizer_dir = DEFAULT_TOKENIZER_DIR
    max_length = 64
    texts = None
    # Example:
    texts = [
        # "The movie was absolutely wonderful",
        "The movie was absolutely garbage",
    ]

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
