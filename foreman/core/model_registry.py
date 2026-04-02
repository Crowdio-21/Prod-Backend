"""Model artifact storage and URI utilities for DNN partition distribution."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

MODEL_STORE_DIR = Path(".model_store")
_ENV_LOADED = False


def _load_foreman_env_once() -> None:
    """Load env vars from foreman/.env once, without overriding process env."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("export "):
                    line = line[len("export ") :].strip()

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue

                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                os.environ.setdefault(key, value)
        except Exception:
            # Keep URL generation resilient; fall back to built-in defaults.
            pass

    _ENV_LOADED = True


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


def _normalize_public_host(host_value: str) -> str:
    """Normalize host value by removing scheme and optional port."""
    host = (host_value or "").strip()
    if not host:
        return ""

    if "://" in host:
        parsed = urlparse(host)
        host = parsed.hostname or ""
    elif host.startswith("[") and "]" in host:
        # IPv6 host with optional port, e.g. [::1]:9000
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:
        maybe_host, maybe_port = host.rsplit(":", 1)
        if maybe_port.isdigit():
            host = maybe_host

    return host.strip()


def build_model_artifact_url(
    model_version_id: str,
    file_name: str,
    host_override: Optional[str] = None,
    port_override: Optional[str] = None,
    scheme_override: Optional[str] = None,
) -> str:
    """Build a download URL for model artifacts reachable by workers."""
    _load_foreman_env_once()

    host = _normalize_public_host(
        host_override or os.getenv("FOREMAN_PUBLIC_HOST", "localhost")
    )
    if not host:
        host = "localhost"

    port = str(port_override or os.getenv("FOREMAN_API_PORT", "8000")).strip()
    if not port:
        port = "8000"

    scheme = (scheme_override or os.getenv("FOREMAN_PUBLIC_SCHEME", "http")).strip()
    if not scheme:
        scheme = "http"

    return (
        f"{scheme}://{host}:{port}/model-artifacts/"
        f"{_safe_component(model_version_id)}/{_safe_component(file_name)}"
    )
