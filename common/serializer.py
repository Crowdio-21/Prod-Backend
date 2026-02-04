"""
Serialization utilities for CrowdCompute
"""

import inspect
import sys
import types
from typing import Any, Callable, List


def _env_info() -> str:
    """Return a concise runtime environment string for diagnostics"""
    return f"python={sys.version.split()[0]}"


def get_runtime_info() -> str:
    """Public helper to expose runtime info to other modules"""
    return _env_info()


def serialize_function(func: str):
    """Serialize a Python function as a str, stripping any decorators"""
    try:
        source = inspect.getsource(func)
        # Strip only @...remote decorator lines to avoid issues on worker
        lines = source.split("\n")
        filtered_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("@") and ".remote" in stripped:
                continue  # Skip @crowdio.remote decorator lines
            filtered_lines.append(line)
        return "\n".join(filtered_lines)
    except Exception as e:
        raise ValueError(f"Failed to serialize function ({_env_info()}): {e}")


def deserialize_function_for_PC(func_code: str):
    """Turn function source code string into a callable function"""

    # Create a local namespace for the exec
    local_vars = {}
    exec(func_code, {}, local_vars)

    # Find the function object in local_vars
    func = None
    for val in local_vars.values():
        if isinstance(val, types.FunctionType):
            func = val
            break

    if func is None:
        raise ValueError("No function could be deserialized from code string")

    return func


def serialize_data(data: Any) -> bytes:
    """Serialize arbitrary data using _"""
    pass


def deserialize_data(data_bytes: bytes) -> Any:
    """Deserialize arbitrary data using _"""
    pass


def hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string back to bytes"""
    return bytes.fromhex(hex_str)


def bytes_to_hex(data_bytes: bytes) -> str:
    """Convert bytes to hex string"""
    return data_bytes.hex()
