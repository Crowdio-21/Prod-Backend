"""Track model partition load readiness for worker-side native runtimes."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple


class ModelLoadTracker:
    """In-memory readiness tracker for worker model partitions."""

    def __init__(self, warmup_seconds: float = 5.0):
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self._dispatched_at: Dict[Tuple[str, str], float] = {}
        self._ready_at: Dict[Tuple[str, str], float] = {}
        self._ready: Set[Tuple[str, str]] = set()
        # Deferred model loads: job_id → {stage_index → [load_info_dict]}
        self._deferred_loads: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        # Track which stage was last loaded for a job: job_id → stage_index
        self._loaded_stage: Dict[str, int] = {}
        # Track model_version_id per job for UNLOAD_MODEL messages
        self._job_model_version: Dict[str, str] = {}
        # Track partition_id per stage per job: job_id → {stage_index → partition_id}
        self._stage_partition: Dict[str, Dict[int, str]] = {}

        # --- Model residency tracking ---
        # Model currently loaded in worker memory (ONNX session / active)
        self._worker_resident_model: Dict[str, str] = {}  # worker_id → partition_id
        # Models cached on worker disk (survives unload, fast reload)
        self._worker_cached_models: Dict[str, Set[str]] = (
            {}
        )  # worker_id → {partition_ids}

    @classmethod
    def from_env(cls) -> "ModelLoadTracker":
        """Create tracker using MODEL_LOAD_WARMUP_SECONDS from environment."""
        raw_value = os.getenv("MODEL_LOAD_WARMUP_SECONDS", "5.0")
        try:
            warmup = float(raw_value)
        except (TypeError, ValueError):
            warmup = 5.0
        return cls(warmup_seconds=warmup)

    @staticmethod
    def _key(worker_id: str, model_partition_id: str) -> Tuple[str, str]:
        return (str(worker_id), str(model_partition_id))

    def _resolve_warmup_seconds(self, warmup_seconds: Optional[float]) -> float:
        if warmup_seconds is None:
            return self.warmup_seconds
        try:
            return max(0.0, float(warmup_seconds))
        except (TypeError, ValueError):
            return self.warmup_seconds

    def mark_dispatched(
        self,
        worker_id: str,
        model_partition_id: str,
        warmup_seconds: Optional[float] = None,
    ) -> None:
        """Record that LOAD_MODEL was sent to worker for a partition."""
        if not worker_id or not model_partition_id:
            return
        key = self._key(worker_id, model_partition_id)
        dispatched_at = time.monotonic()
        self._dispatched_at[key] = dispatched_at
        wait_seconds = self._resolve_warmup_seconds(warmup_seconds)
        self._ready_at[key] = dispatched_at + wait_seconds
        # A fresh dispatch means we should wait for readiness again.
        self._ready.discard(key)

    def mark_ready(self, worker_id: str, model_partition_id: str) -> None:
        """Record explicit readiness for a worker partition."""
        if not worker_id or not model_partition_id:
            return
        key = self._key(worker_id, model_partition_id)
        self._ready.add(key)
        now = time.monotonic()
        self._dispatched_at.setdefault(key, now)
        self._ready_at[key] = now
        # Update residency: this partition is now active in worker memory
        self._worker_resident_model[worker_id] = model_partition_id
        self._worker_cached_models.setdefault(worker_id, set()).add(model_partition_id)

    def has_dispatch(self, worker_id: str, model_partition_id: str) -> bool:
        """Return True when a LOAD_MODEL dispatch has been recorded."""
        if not worker_id or not model_partition_id:
            return False
        return self._key(worker_id, model_partition_id) in self._dispatched_at

    def seconds_until_ready(
        self, worker_id: str, model_partition_id: str
    ) -> Optional[float]:
        """Return remaining warmup time; None means no dispatch is known."""
        if not worker_id or not model_partition_id:
            return None

        key = self._key(worker_id, model_partition_id)
        if key in self._ready:
            return 0.0

        dispatched_at = self._dispatched_at.get(key)
        if dispatched_at is None:
            return None

        ready_at = self._ready_at.get(key, dispatched_at + self.warmup_seconds)
        remaining = ready_at - time.monotonic()
        return remaining if remaining > 0.0 else 0.0

    def is_ready(self, worker_id: str, model_partition_id: str) -> bool:
        """Return True when a partition can be used for execution."""
        remaining = self.seconds_until_ready(worker_id, model_partition_id)
        # No dispatch record means Foreman is not managing this partition; do not block.
        if remaining is None:
            return True
        return remaining <= 0.0

    # ------------------------------------------------------------------ #
    #  Deferred model load management
    # ------------------------------------------------------------------ #

    def store_deferred_load(
        self,
        job_id: str,
        stage_index: int,
        partition_id: str,
        worker_id: str,
        model_url: str,
        model_version_id: str,
        checksum: Optional[str] = None,
        warmup_seconds: float = 5.0,
    ) -> None:
        """Store model load info to be dispatched when the target stage is unblocked."""
        if job_id not in self._deferred_loads:
            self._deferred_loads[job_id] = {}
        stage_loads = self._deferred_loads[job_id].setdefault(stage_index, [])
        stage_loads.append(
            {
                "partition_id": partition_id,
                "worker_id": worker_id,
                "model_url": model_url,
                "model_version_id": model_version_id,
                "checksum": checksum,
                "warmup_seconds": warmup_seconds,
            }
        )
        self._job_model_version[job_id] = model_version_id
        self._stage_partition.setdefault(job_id, {})[stage_index] = partition_id

    def pop_deferred_loads_for_stage(
        self, job_id: str, stage_index: int
    ) -> List[Dict[str, Any]]:
        """Pop and return deferred loads for a given stage. Returns [] if none."""
        job_loads = self._deferred_loads.get(job_id)
        if not job_loads:
            return []
        return job_loads.pop(stage_index, [])

    def has_deferred_loads(self, job_id: str, stage_index: int) -> bool:
        """Check if there are deferred model loads for a given stage."""
        job_loads = self._deferred_loads.get(job_id)
        if not job_loads:
            return False
        return bool(job_loads.get(stage_index))

    def set_loaded_stage(self, job_id: str, stage_index: int) -> None:
        """Record which stage was last loaded for a job."""
        self._loaded_stage[job_id] = stage_index

    def get_loaded_stage(self, job_id: str) -> Optional[int]:
        """Return the last loaded stage index for a job, or None."""
        return self._loaded_stage.get(job_id)

    def get_job_model_version(self, job_id: str) -> Optional[str]:
        """Return the model_version_id for a job."""
        return self._job_model_version.get(job_id)

    def set_job_model_version(self, job_id: str, model_version_id: str) -> None:
        """Record model_version_id for a job."""
        if not job_id or not model_version_id:
            return
        self._job_model_version[job_id] = model_version_id

    def set_stage_partition_id(
        self, job_id: str, stage_index: int, partition_id: str
    ) -> None:
        """Record partition_id for a specific stage of a job."""
        if not job_id or not partition_id:
            return
        self._stage_partition.setdefault(job_id, {})[stage_index] = partition_id

    def get_stage_partition_id(self, job_id: str, stage_index: int) -> Optional[str]:
        """Return the partition_id for a given stage of a job."""
        return self._stage_partition.get(job_id, {}).get(stage_index)

    def cleanup_job(self, job_id: str) -> None:
        """Clean up all deferred/staging state for a completed job."""
        self._deferred_loads.pop(job_id, None)
        self._loaded_stage.pop(job_id, None)
        self._job_model_version.pop(job_id, None)
        self._stage_partition.pop(job_id, None)

    # ------------------------------------------------------------------ #
    #  Model residency tracking
    # ------------------------------------------------------------------ #

    def mark_model_unloaded(self, worker_id: str, model_partition_id: str) -> None:
        """Record that a partition was unloaded from worker memory.

        The partition remains in the cached set (still on disk) unless
        explicitly removed.
        """
        if not worker_id or not model_partition_id:
            return
        current = self._worker_resident_model.get(worker_id)
        if current == model_partition_id:
            del self._worker_resident_model[worker_id]
        # Clear readiness so we don't skip a re-load
        key = self._key(worker_id, model_partition_id)
        self._ready.discard(key)
        self._dispatched_at.pop(key, None)
        self._ready_at.pop(key, None)

    def mark_model_evicted(self, worker_id: str, model_partition_id: str) -> None:
        """Record that a partition was removed from worker disk cache."""
        self.mark_model_unloaded(worker_id, model_partition_id)
        cached = self._worker_cached_models.get(worker_id)
        if cached:
            cached.discard(model_partition_id)

    def clear_worker_residency(self, worker_id: str) -> None:
        """Clear all residency state for a disconnected worker."""
        self._worker_resident_model.pop(worker_id, None)
        self._worker_cached_models.pop(worker_id, None)
        # Clean up readiness entries for this worker
        to_remove = [k for k in self._dispatched_at if k[0] == worker_id]
        for k in to_remove:
            self._dispatched_at.pop(k, None)
            self._ready_at.pop(k, None)
            self._ready.discard(k)

    def register_cached_models(self, worker_id: str, partition_ids: List[str]) -> None:
        """Register model partitions a worker reports as cached on disk."""
        if not worker_id:
            return
        cached = self._worker_cached_models.setdefault(worker_id, set())
        for pid in partition_ids:
            if pid:
                cached.add(pid)

    def get_resident_model(self, worker_id: str) -> Optional[str]:
        """Return the partition currently loaded in a worker's memory, or None."""
        return self._worker_resident_model.get(worker_id)

    def workers_with_model_resident(self, partition_id: str) -> List[str]:
        """Return worker IDs that have *partition_id* loaded in memory."""
        return [
            wid
            for wid, pid in self._worker_resident_model.items()
            if pid == partition_id
        ]

    def workers_with_model_cached(self, partition_id: str) -> List[str]:
        """Return worker IDs that have *partition_id* cached on disk."""
        return [
            wid
            for wid, pids in self._worker_cached_models.items()
            if partition_id in pids
        ]

    def is_model_resident(self, worker_id: str, partition_id: str) -> bool:
        """Check if a specific partition is currently loaded in worker memory."""
        return self._worker_resident_model.get(worker_id) == partition_id

    def is_model_cached(self, worker_id: str, partition_id: str) -> bool:
        """Check if a specific partition is cached on worker disk."""
        return partition_id in self._worker_cached_models.get(worker_id, set())
