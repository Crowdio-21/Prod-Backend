"""
Feature router for DNN pipeline/topology execution.

This module tracks intermediate feature payloads keyed by
(job_id, target_task_id, source_task_id) and provides a simple routing API.
"""

from typing import Any, Dict, List, Optional, Tuple


class FeatureRouter:
    """In-memory intermediate feature routing for topology-driven jobs."""

    def __init__(self):
        self._buffer: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def store_feature(
        self,
        job_id: str,
        source_task_id: str,
        target_task_id: str,
        payload: Any,
        payload_format: str = "json",
    ) -> None:
        key = (job_id, target_task_id, source_task_id)
        self._buffer[key] = {
            "payload": payload,
            "payload_format": payload_format,
        }

    def pop_feature(
        self,
        job_id: str,
        source_task_id: str,
        target_task_id: str,
    ) -> Optional[Dict[str, Any]]:
        key = (job_id, target_task_id, source_task_id)
        return self._buffer.pop(key, None)

    def route_completion_to_dependents(
        self,
        job_id: str,
        source_task_id: str,
        dependent_task_ids: List[str],
        result_payload: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Produce per-dependent routed payloads.

        If a worker already provided a dedicated intermediate feature message
        for a target, that payload is used. Otherwise, the task result is used
        as the fallback payload.
        """
        routed: Dict[str, Dict[str, Any]] = {}

        for target_task_id in dependent_task_ids:
            explicit = self.pop_feature(job_id, source_task_id, target_task_id)
            if explicit is not None:
                routed[target_task_id] = explicit
            else:
                routed[target_task_id] = {
                    "payload": result_payload,
                    "payload_format": "result",
                }

        return routed

    def clear_job(self, job_id: str) -> None:
        keys_to_remove = [k for k in self._buffer.keys() if k[0] == job_id]
        for key in keys_to_remove:
            self._buffer.pop(key, None)
