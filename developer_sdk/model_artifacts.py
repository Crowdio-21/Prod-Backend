"""Helpers for packaging model partition artifacts for dnn_pipeline submissions."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List


def build_partition_artifact(
    model_partition_id: str,
    file_path: str,
    assigned_device_id: str,
) -> Dict[str, str]:
    """Package one partition artifact into the dnn_config model_artifacts format."""
    src = Path(file_path)
    raw = src.read_bytes()
    return {
        "model_partition_id": model_partition_id,
        "file_name": src.name,
        "assigned_device_id": assigned_device_id,
        "content_b64": base64.b64encode(raw).decode("ascii"),
    }


def build_partition_artifacts(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Batch helper for model artifacts.

    Each entry requires: model_partition_id, file_path, assigned_device_id.
    """
    artifacts: List[Dict[str, str]] = []
    for item in entries:
        artifacts.append(
            build_partition_artifact(
                model_partition_id=item["model_partition_id"],
                file_path=item["file_path"],
                assigned_device_id=item["assigned_device_id"],
            )
        )
    return artifacts
