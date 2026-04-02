"""Model partition loading and local cache management for workers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlopen


class ModelLoader:
    """Downloads and caches model partitions for worker runtimes."""

    def __init__(self, cache_dir: str = ".worker_model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_partitions: Dict[str, Dict[str, str]] = {}

    def _cache_path(
        self, model_version_id: str, model_partition_id: str, extension: str = "bin"
    ) -> Path:
        safe_version = "".join(
            ch for ch in model_version_id if ch.isalnum() or ch in ("-", "_", ".")
        )
        safe_partition = "".join(
            ch for ch in model_partition_id if ch.isalnum() or ch in ("-", "_", ".")
        )
        target_dir = self.cache_dir / safe_version
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{safe_partition}.{extension}"

    @staticmethod
    def _sha256(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def load_from_uri(
        self,
        model_version_id: str,
        model_partition_id: str,
        model_uri: str,
        checksum: Optional[str] = None,
    ) -> Dict[str, str]:
        with urlopen(model_uri) as resp:
            raw = resp.read()

        if checksum:
            digest = self._sha256(raw)
            if digest != checksum:
                raise ValueError(
                    f"Checksum mismatch for {model_partition_id}: expected {checksum}, got {digest}"
                )

        extension = "tflite" if model_uri.endswith(".tflite") else "bin"
        cache_path = self._cache_path(
            model_version_id, model_partition_id, extension=extension
        )
        cache_path.write_bytes(raw)

        entry = {
            "model_version_id": model_version_id,
            "model_partition_id": model_partition_id,
            "model_uri": model_uri,
            "local_path": str(cache_path),
            "checksum": checksum or self._sha256(raw),
        }
        self.loaded_partitions[model_partition_id] = entry
        return entry

    def get_loaded_partition(self, model_partition_id: str) -> Optional[Dict[str, str]]:
        return self.loaded_partitions.get(model_partition_id)
