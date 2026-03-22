"""Dynamic topology replanning for DNN jobs after worker failures."""

from typing import Any, Dict, List, Optional

from foreman.db.base import db_session
from foreman.db.models import TaskModel
from sqlalchemy import select


class DynamicTopologyPlanner:
    """Recompute topology assignments for affected DNN jobs."""

    def __init__(self, connection_manager, task_dispatcher, job_manager):
        self.connection_manager = connection_manager
        self.task_dispatcher = task_dispatcher
        self.job_manager = job_manager

    async def replan_job(
        self,
        job_id: str,
        inference_graph_id: str,
        recovered_task_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        graph = self.job_manager.topology_manager.get_graph(inference_graph_id)
        if not graph:
            return None

        partition_requirements = await self._collect_partition_requirements(job_id)
        all_workers = await self.task_dispatcher._get_all_workers()
        connected_worker_ids = self.connection_manager.get_all_worker_ids()

        if not connected_worker_ids:
            return graph

        used_workers = set()
        assignments: Dict[str, str] = {}

        for node in graph.get("nodes", []):
            node_id = node.get("node_id")
            if not node_id:
                continue

            partition_id = node.get("model_partition_id")
            req = partition_requirements.get(partition_id, {}) if partition_id else {}
            worker_id = self._select_best_worker(
                all_workers=all_workers,
                connected_worker_ids=connected_worker_ids,
                used_workers=used_workers,
                requirements=req,
            )
            if worker_id:
                assignments[node_id] = worker_id
                used_workers.add(worker_id)

        if assignments:
            graph = self.job_manager.topology_manager.update_assignments(
                inference_graph_id=inference_graph_id,
                node_assignments=assignments,
            )

        # Best-effort redispatch for recovered work after replanning.
        if recovered_task_ids:
            func_code = self.job_manager.get_func_code(job_id) or ""
            await self.task_dispatcher.assign_tasks_for_job(job_id, func_code, [])

        return graph

    async def _collect_partition_requirements(
        self, job_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Collect merged device requirements per model partition id."""
        import json

        requirements_by_partition: Dict[str, Dict[str, Any]] = {}

        async with db_session() as session:
            result = await session.execute(
                select(TaskModel).where(TaskModel.job_id == job_id)
            )
            tasks = result.scalars().all()

        for row in tasks:
            partition_key = row.model_partition_id
            if not partition_key:
                continue
            req_text = row.device_requirements
            if not req_text:
                continue
            try:
                req = json.loads(req_text)
            except Exception:
                req = {}

            existing = requirements_by_partition.get(partition_key, {})
            requirements_by_partition[partition_key] = self._merge_requirements(
                existing, req
            )

        return requirements_by_partition

    def _merge_requirements(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(left)
        for key, value in right.items():
            if key in ("min_ram_mb", "min_battery"):
                merged[key] = max(float(merged.get(key, 0.0)), float(value))
            elif key in (
                "allowed_os_types",
                "allowed_runtimes",
                "allowed_model_runtimes",
            ):
                existing = set(merged.get(key, []))
                merged[key] = sorted(existing.union(set(value or [])))
            else:
                merged[key] = value
        return merged

    def _select_best_worker(
        self,
        all_workers,
        connected_worker_ids,
        used_workers,
        requirements,
    ) -> Optional[str]:
        candidates = []
        for worker_id in connected_worker_ids:
            if worker_id in used_workers:
                continue
            worker = all_workers.get(worker_id)
            if not worker:
                continue
            if not self.task_dispatcher._worker_meets_requirements(
                worker, requirements
            ):
                continue

            score = (
                (worker.success_rate or 0.0) * 100.0
                + (worker.ram_available_mb or 0.0) / 256.0
                + (10.0 if worker.is_charging else 0.0)
                + (5.0 if worker.gpu_available else 0.0)
            )
            candidates.append((score, worker_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]
