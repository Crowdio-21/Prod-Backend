"""Helpers for packaging model partition artifacts for dnn_pipeline submissions."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional


def _read_artifact_bytes(src: Path) -> bytes:
    """Read artifact bytes, inlining ONNX external tensor data when present.

    Some exported ONNX models use sidecar files (for example, `<model>.onnx.data`).
    Mobile runtimes often receive only the primary `.onnx` file, which then fails to
    load due to missing external data. To make transport robust, inline tensors into
    a single ONNX payload at packaging time.
    """
    if src.suffix.lower() != ".onnx":
        return src.read_bytes()

    sidecar = Path(f"{src.as_posix()}.data")
    if not sidecar.exists():
        return src.read_bytes()

    try:
        import onnx
        from onnx import external_data_helper
    except Exception as exc:
        raise RuntimeError(
            "Detected ONNX external data sidecar, but 'onnx' is not available to inline it. "
            "Install with: pip install onnx"
        ) from exc

    try:
        model = onnx.load(src.as_posix(), load_external_data=True)
        external_data_helper.convert_model_from_external_data(model)
        return model.SerializeToString()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to inline external ONNX data for artifact: {src}"
        ) from exc


def build_partition_artifact(
    model_partition_id: str,
    file_path: str,
    assigned_device_id: Optional[str] = None,
) -> Dict[str, str]:
    """Package one partition artifact into the dnn_config model_artifacts format."""
    src = Path(file_path)
    raw = _read_artifact_bytes(src)
    artifact = {
        "model_partition_id": model_partition_id,
        "file_name": src.name,
        "content_b64": base64.b64encode(raw).decode("ascii"),
    }
    if assigned_device_id:
        artifact["assigned_device_id"] = assigned_device_id
    return artifact


def build_partition_artifacts(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Batch helper for model artifacts.

    Each entry requires: model_partition_id and file_path.
    Optional: assigned_device_id.
    """
    artifacts: List[Dict[str, str]] = []
    for item in entries:
        artifacts.append(
            build_partition_artifact(
                model_partition_id=item["model_partition_id"],
                file_path=item["file_path"],
                assigned_device_id=item.get("assigned_device_id"),
            )
        )
    return artifacts
