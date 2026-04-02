"""
Serialization utilities for CrowdCompute
"""

import base64
import inspect
import re
import sys
import types
import zlib
from typing import Any, Callable, List, Dict

try:
    import numpy as np
except Exception:
    np = None


DEFAULT_COMPRESS_LEVEL = 6


def _env_info() -> str:
    """Return a concise runtime environment string for diagnostics"""
    return f"python={sys.version.split()[0]}"


def get_runtime_info() -> str:
    """Public helper to expose runtime info to other modules"""
    return _env_info()


def _strip_decorators(source: str) -> str:
    """
    Strip decorator lines from function source code.

    Removes @decorator(...) lines that precede function definitions,
    including multi-line decorators with parentheses.

    Args:
        source: Function source code string

    Returns:
        Source code with decorators removed
    """
    lines = source.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this is a decorator line
        if stripped.startswith("@"):
            # Skip decorator lines (including multi-line decorators)
            paren_count = stripped.count("(") - stripped.count(")")
            i += 1

            # Continue skipping if we're inside parentheses
            while paren_count > 0 and i < len(lines):
                paren_count += lines[i].count("(") - lines[i].count(")")
                i += 1
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def serialize_function(func: Callable) -> str:
    """
    Serialize a Python function as a str

    Strips any decorators from the source code so the function
    can be executed on workers without needing decorator dependencies.

    Args:
        func: Function to serialize

    Returns:
        Function source code string (without decorators)
    """
    try:
        source = inspect.getsource(func)
        # Strip decorators so workers don't need decorator dependencies
        source = _strip_decorators(source)
        return source
    except Exception as e:
        raise ValueError(f"Failed to serialize function ({_env_info()}): {e}")


def deserialize_function_for_PC(func_code: str):
    """
    Turn function source code string into a callable function.

    Handles code that includes a task control wrapper (pause/resume/kill
    functions prepended by the SDK) by using a shared namespace and
    selecting the user's task function (skipping internal wrapper functions).
    """

    # Use a single namespace so wrapper globals (paused, killed, time)
    # are accessible from the function's __globals__
    namespace = {"__builtins__": __builtins__}
    exec(func_code, namespace)

    # Internal wrapper function names to skip
    _internal_names = {"pause", "resume", "kill"}

    # Find the user's function (skip wrapper functions)
    func = None
    for name, val in namespace.items():
        if isinstance(val, types.FunctionType) and name not in _internal_names:
            func = val
            break

    if func is None:
        raise ValueError("No function could be deserialized from code string")

    return func


def serialize_data(data: Any) -> bytes:
    """Serialize arbitrary data using _"""
    raise NotImplementedError(
        "serialize_data is not implemented; use explicit serializers for supported payload types"
    )


def deserialize_data(data_bytes: bytes) -> Any:
    """Deserialize arbitrary data using _"""
    raise NotImplementedError(
        "deserialize_data is not implemented; use explicit deserializers for supported payload types"
    )


def hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string back to bytes"""
    return bytes.fromhex(hex_str)


def bytes_to_hex(data_bytes: bytes) -> str:
    """Convert bytes to hex string"""
    return data_bytes.hex()


def serialize_tensor(
    tensor: "np.ndarray",
    compression: str = "zlib",
    compress_level: int = DEFAULT_COMPRESS_LEVEL,
) -> Dict[str, Any]:
    """Serialize numpy tensor to transport-friendly dict payload."""
    if np is None:
        raise ImportError("numpy is required for tensor serialization")
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


def deserialize_tensor(payload: Dict[str, Any]) -> "np.ndarray":
    """Deserialize transport payload back into numpy tensor."""
    if np is None:
        raise ImportError("numpy is required for tensor deserialization")

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


def encode_feature_payload(payload: Any) -> Dict[str, Any]:
    """
    Encode intermediate feature payload with tensor-aware serialization.

    Recursively encodes nested structures (dicts, lists) and converts
    numpy arrays to transport-friendly format using zlib compression.

    Args:
        payload: Payload to encode (may contain numpy arrays)

    Returns:
        Encoded payload dict with tensors as serialized dicts
    """
    if np is not None and isinstance(payload, np.ndarray):
        return {
            "transport": "tensor_transport",
            "tensor": serialize_tensor(payload),
        }

    if isinstance(payload, dict):
        return {k: encode_feature_payload(v) for k, v in payload.items()}

    if isinstance(payload, list):
        return [encode_feature_payload(v) for v in payload]

    return payload


def decode_feature_payload(payload: Any) -> Any:
    """
    Decode intermediate feature payload and materialize serialized tensors.

    Recursively decodes nested structures (dicts, lists) and reconstructs
    numpy arrays from their serialized format.

    Args:
        payload: Payload to decode

    Returns:
        Decoded payload with tensors materialized as numpy arrays
    """
    if isinstance(payload, dict):
        if payload.get("transport") == "tensor_transport" and "tensor" in payload:
            return deserialize_tensor(payload["tensor"])
        return {k: decode_feature_payload(v) for k, v in payload.items()}

    if isinstance(payload, list):
        return [decode_feature_payload(v) for v in payload]

    return payload
