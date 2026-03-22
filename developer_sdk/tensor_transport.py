"""Tensor serialization helpers for DNN feature transport."""

import base64
import zlib
from typing import Any, Dict

import numpy as np


DEFAULT_COMPRESS_LEVEL = 6


def serialize_tensor(
    tensor: np.ndarray,
    compression: str = "zlib",
    compress_level: int = DEFAULT_COMPRESS_LEVEL,
) -> Dict[str, Any]:
    """Serialize numpy tensor to transport-friendly dict payload."""
    if not isinstance(tensor, np.ndarray):
        raise TypeError("tensor must be a numpy.ndarray")

    raw = tensor.tobytes(order="C")
    codec = compression or "none"
    encoded_bytes = raw

    if codec == "zlib":
        encoded_bytes = zlib.compress(raw, level=compress_level)
    elif codec != "none":
        raise ValueError(f"Unsupported compression codec: {codec}")

    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "order": "C",
        "compression": codec,
        "payload_b64": base64.b64encode(encoded_bytes).decode("ascii"),
    }


def deserialize_tensor(payload: Dict[str, Any]) -> np.ndarray:
    """Deserialize transport payload back into numpy tensor."""
    dtype = payload["dtype"]
    shape = tuple(payload["shape"])
    compression = payload.get("compression", "none")
    payload_b64 = payload["payload_b64"]

    encoded = base64.b64decode(payload_b64)
    raw = encoded
    if compression == "zlib":
        raw = zlib.decompress(encoded)
    elif compression != "none":
        raise ValueError(f"Unsupported compression codec: {compression}")

    array = np.frombuffer(raw, dtype=np.dtype(dtype))
    return array.reshape(shape)
