"""Model artifact storage and URI utilities for DNN partition distribution."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional


MODEL_STORE_DIR = Path(".model_store")


def _safe_component(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_", "."))


def get_model_version_dir(model_version_id: str) -> Path:
    safe_version = _safe_component(model_version_id)
    version_dir = MODEL_STORE_DIR / safe_version
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def store_partition_blob(
    model_version_id: str,
    model_partition_id: str,
    content_b64: str,
    file_name: Optional[str] = None,
) -> Dict[str, str]:
    """Persist base64 partition artifact and return metadata with checksum."""
    version_dir = get_model_version_dir(model_version_id)

    suggested_name = file_name or f"{_safe_component(model_partition_id)}.tflite"
    safe_name = _safe_component(suggested_name)
    if not safe_name:
        safe_name = f"{_safe_component(model_partition_id)}.bin"

    raw_bytes = base64.b64decode(content_b64)
    checksum = hashlib.sha256(raw_bytes).hexdigest()

    partition_path = version_dir / safe_name
    partition_path.write_bytes(raw_bytes)

    return {
        "model_version_id": model_version_id,
        "model_partition_id": model_partition_id,
        "file_name": safe_name,
        "checksum": checksum,
        "storage_path": str(partition_path),
    }


def resolve_partition_path(model_version_id: str, file_name: str) -> Path:
    return get_model_version_dir(model_version_id) / _safe_component(file_name)


def list_model_manifest(model_version_id: str) -> List[Dict[str, str]]:
    """List artifacts under a model version directory with checksums."""
    version_dir = get_model_version_dir(model_version_id)
    manifest: List[Dict[str, str]] = []

    for file_path in sorted(version_dir.glob("*")):
        if not file_path.is_file():
            continue
        raw = file_path.read_bytes()
        manifest.append(
            {
                "file_name": file_path.name,
                "checksum": hashlib.sha256(raw).hexdigest(),
                "size_bytes": str(file_path.stat().st_size),
            }
        )

    return manifest


def build_model_artifact_url(model_version_id: str, file_name: str) -> str:
    host = os.getenv("FOREMAN_PUBLIC_HOST", "localhost")
    port = os.getenv("FOREMAN_API_PORT", "8000")
    return f"http://{host}:{port}/model-artifacts/{_safe_component(model_version_id)}/{_safe_component(file_name)}"
