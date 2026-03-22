"""
Dependency Manager for Pipeline Execution

Manages task dependency counters and pipeline stage progression.
When a task completes, this manager:
1. Finds all downstream (dependent) tasks
2. Decrements their dependency counters
3. Unblocks tasks whose counters reach zero (blocked → pending)
4. Optionally injects upstream results into downstream task arguments

Works transparently with the existing scheduler/dispatcher:
- Blocked tasks have status="blocked" and are invisible to the scheduler
- Once unblocked (status="pending"), they enter the normal dispatch flow

Similar to how CheckpointManager handles checkpoint lifecycle, this class
handles the full dependency lifecycle from job creation through completion.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from foreman.db.base import db_session
from foreman.db.models import TaskModel, JobModel
from foreman.core.payload_store import store_text_if_large, resolve_text_ref
from foreman.db.crud import (
    create_pipeline_job,
    create_task_with_dependencies,
    decrement_dependency_count,
    get_task_dependents,
    get_task_by_id,
    update_task_args,
)
from .feature_router import FeatureRouter


class PipelineStage:
    """
    Defines a single stage in a pipeline.

    Each stage has:
    - A function (serialised code) to execute
    - A list of input arguments (one per task in the stage)
    - An optional flag to pass upstream results as arguments

    Usage (SDK-side, see developer_sdk/client.py):
        stage_a = PipelineStage(func_code_a, args_list_a)
        stage_b = PipelineStage(func_code_b, args_list_b, pass_upstream_results=True)
    """

    def __init__(
        self,
        func_code: str,
        args_list: List[Any],
        pass_upstream_results: bool = False,
        task_metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            func_code: Serialised function source code for this stage
            args_list: List of arguments – one entry per task in this stage.
                       If pass_upstream_results is True and this is not the
                       first stage, each entry may be a *template* that will
                       be augmented with upstream results at runtime.
            pass_upstream_results: When True the system will inject the
                       results of upstream dependency tasks into the
                       downstream task's arguments automatically.
            task_metadata: Optional checkpoint / task configuration dict
                       (same shape as the declarative checkpointing metadata).
            name: Human-readable stage label (for logging / dashboard).
        """
        self.func_code = func_code
        self.args_list = args_list
        self.pass_upstream_results = pass_upstream_results
        self.task_metadata = task_metadata
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "func_code": self.func_code,
            "args_list": self.args_list,
            "pass_upstream_results": self.pass_upstream_results,
            "task_metadata": self.task_metadata,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStage":
        return cls(
            func_code=data["func_code"],
            args_list=data["args_list"],
            pass_upstream_results=data.get("pass_upstream_results", False),
            task_metadata=data.get("task_metadata"),
            name=data.get("name"),
        )


class DependencyManager:
    """
    Manages task dependency counters for pipeline execution.

    Lifecycle:
    1. create_pipeline_tasks()   – called once when a pipeline job is submitted
    2. on_task_completed()       – called every time a task finishes
       • decrements counters of downstream tasks
       • optionally injects upstream result into downstream args
       • returns list of newly-unblocked task IDs (ready for dispatch)

    The DependencyManager does NOT dispatch tasks itself.  It only
    manipulates statuses and returns the IDs of unblocked tasks so that
    the caller (message handler / completion handler) can ask the
    TaskDispatcher to schedule them.
    """

    def __init__(self):
        # In-memory cache: job_id → { stage_index → func_code }
        self._pipeline_func_codes: Dict[str, Dict[int, str]] = {}
        # In-memory cache: job_id → pipeline metadata
        self._pipeline_metadata: Dict[str, Dict[str, Any]] = {}
        self.feature_router = FeatureRouter()

    # ------------------------------------------------------------------ #
    #  Pipeline Job Creation
    # ------------------------------------------------------------------ #

    async def create_pipeline_tasks(
        self,
        job_id: str,
        stages: List[PipelineStage],
        dependency_map: Optional[Dict[str, List[str]]] = None,
        dnn_config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create all tasks for a pipeline job and wire their dependencies.

        Two dependency modes are supported:

        **Sequential stages (default)** – every task in stage N depends on
        *all* tasks in stage N-1.  Works like a barrier: stage N cannot
        start until the entire previous stage is done.

        **Custom dependency map** – a dict mapping ``task_id → [upstream_task_ids]``
        that lets callers express arbitrary DAGs (fan-in, fan-out, diamond, …).

        Args:
            job_id:  Unique job identifier (already created in DB).
            stages:  Ordered list of PipelineStage objects.
            dependency_map:  Optional explicit dependency mapping.
                             Keys are task IDs, values are lists of
                             task IDs they depend on.

        Returns:
            Total number of tasks created.
        """
        total_tasks = 0

        # ----- build task-id lists per stage (so we can wire deps) ----- #
        stage_task_ids: List[List[str]] = []
        for stage_idx, stage in enumerate(stages):
            ids = [
                f"{job_id}_task_{stage_idx}_{i}" for i in range(len(stage.args_list))
            ]
            stage_task_ids.append(ids)
            total_tasks += len(ids)

        # Cache func_codes per stage for later dispatch
        self._pipeline_func_codes[job_id] = {
            idx: stage.func_code for idx, stage in enumerate(stages)
        }
        self._pipeline_metadata[job_id] = {
            "total_stages": len(stages),
            "stages": [s.to_dict() for s in stages],
        }

        # ----- create tasks with dependency wiring ----- #
        node_by_stage: Dict[int, Dict[str, Any]] = {}
        node_lookup: Dict[str, Dict[str, Any]] = {}
        if dnn_config:
            nodes = dnn_config.get("topology_nodes", [])
            edges = dnn_config.get("topology_edges", [])
            for node in nodes:
                node_id = node.get("node_id")
                if node_id:
                    node_lookup[node_id] = node

            for stage_idx, stage in enumerate(stages):
                stage_name = stage.name
                if not stage_name:
                    continue
                for node in nodes:
                    if (
                        node.get("model_partition_id") == stage_name
                        or node.get("node_id") == stage_name
                    ):
                        node_by_stage[stage_idx] = node
                        break

        async with db_session() as session:
            for stage_idx, stage in enumerate(stages):
                stage_node = node_by_stage.get(stage_idx)
                stage_node_id = stage_node.get("node_id") if stage_node else None
                stage_role = stage_node.get("role") if stage_node else None
                stage_partition = (
                    stage_node.get("model_partition_id") if stage_node else None
                )
                stage_requirements = (
                    stage_node.get("requirements", {}) if stage_node else {}
                )

                feature_sources = []
                feature_targets = []
                if dnn_config and stage_node_id:
                    for edge in edges:
                        if edge.get("target") == stage_node_id:
                            feature_sources.append(edge.get("source"))
                        if edge.get("source") == stage_node_id:
                            feature_targets.append(edge.get("target"))

                for task_local_idx, args in enumerate(stage.args_list):
                    task_id = stage_task_ids[stage_idx][task_local_idx]

                    # Determine upstream dependencies
                    if dependency_map and task_id in dependency_map:
                        # Explicit dependency map
                        depends_on = dependency_map[task_id]
                    elif stage_idx > 0:
                        # Default: depend on ALL tasks in previous stage
                        depends_on = stage_task_ids[stage_idx - 1]
                    else:
                        depends_on = []

                    # Determine downstream dependents
                    if stage_idx < len(stages) - 1:
                        # Default: all tasks in next stage depend on me
                        dependents = stage_task_ids[stage_idx + 1]
                    else:
                        dependents = []

                    # Override dependents if explicit map provided
                    if dependency_map:
                        dependents = [
                            tid
                            for tid, deps in dependency_map.items()
                            if task_id in deps
                        ]

                    dependency_count = len(depends_on)
                    initial_status = "blocked" if dependency_count > 0 else "pending"

                    task = TaskModel(
                        id=task_id,
                        job_id=job_id,
                        args=store_text_if_large(
                            json.dumps(args), "task_args", task_id
                        ),
                        status=initial_status,
                        stage=stage_idx,
                        dependency_count=dependency_count,
                        depends_on=json.dumps(depends_on) if depends_on else None,
                        dependents=json.dumps(dependents) if dependents else None,
                        stage_func_code=stage.func_code,
                        topology_role=stage_role,
                        feature_sources=(
                            json.dumps(feature_sources) if feature_sources else None
                        ),
                        feature_targets=(
                            json.dumps(feature_targets) if feature_targets else None
                        ),
                        device_requirements=(
                            json.dumps(stage_requirements)
                            if stage_requirements
                            else None
                        ),
                        model_partition_id=stage_partition,
                    )
                    session.add(task)

            await session.commit()

        stage_summary = " → ".join(
            f"Stage {i}({stage.name or 'unnamed'}: {len(stage.args_list)} tasks)"
            for i, stage in enumerate(stages)
        )
        print(
            f"DependencyManager: Created pipeline for job {job_id}: "
            f"{stage_summary} | total={total_tasks} tasks"
        )

        return total_tasks

    # ------------------------------------------------------------------ #
    #  On Task Completion – decrement counters & unblock
    # ------------------------------------------------------------------ #

    async def on_task_completed(
        self,
        task_id: str,
        job_id: str,
        result: Any,
    ) -> List[str]:
        """
        Called when a task finishes successfully.

        Decrements the dependency counter of every downstream task.
        If a downstream task's counter reaches 0, it becomes "pending"
        and its ID is returned so the caller can dispatch it.

        If the completed stage was configured with pass_upstream_results,
        the result is injected into downstream task arguments.

        Args:
            task_id:  The completed task's ID
            job_id:   The job the task belongs to
            result:   The task's result value

        Returns:
            List of task IDs that were just unblocked (newly pending).
        """
        newly_unblocked: List[str] = []

        async with db_session() as session:
            # Get the completed task to find its dependents
            dependents_json = await get_task_dependents(session, task_id)

            if not dependents_json:
                return newly_unblocked

            routed_payloads = self.feature_router.route_completion_to_dependents(
                job_id=job_id,
                source_task_id=task_id,
                dependent_task_ids=dependents_json,
                result_payload=result,
            )

            for dependent_id in dependents_json:
                # Inject upstream result into downstream args BEFORE
                # decrementing so that results from ALL upstream tasks
                # accumulate (not just the one that triggers unblock).
                await self._maybe_inject_upstream_result(
                    session,
                    job_id,
                    dependent_id,
                    task_id,
                    routed_payloads.get(dependent_id, {}).get("payload", result),
                )

                # Decrement counter
                new_count, new_status = await decrement_dependency_count(
                    session, dependent_id
                )

                if new_count == 0 and new_status == "pending":
                    newly_unblocked.append(dependent_id)
                    print(
                        f"DependencyManager: ✅ Task {dependent_id} unblocked "
                        f"(all dependencies met)"
                    )

                elif new_count > 0:
                    print(
                        f"DependencyManager: Task {dependent_id} still blocked "
                        f"({new_count} dependencies remaining)"
                    )

        if newly_unblocked:
            print(
                f"DependencyManager: 🚀 {len(newly_unblocked)} tasks unblocked "
                f"for job {job_id}: {newly_unblocked}"
            )

        return newly_unblocked

    def register_intermediate_feature(
        self,
        job_id: str,
        source_task_id: str,
        target_task_id: str,
        payload: Any,
        payload_format: str = "json",
    ) -> None:
        """Store an explicit worker-provided intermediate feature for a target task."""
        self.feature_router.store_feature(
            job_id=job_id,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            payload=payload,
            payload_format=payload_format,
        )

    # ------------------------------------------------------------------ #
    #  Upstream result injection
    # ------------------------------------------------------------------ #

    def _normalize_upstream_result_for_pipeline(self, upstream_result: Any) -> Any:
        """Normalize special transport payloads for downstream pipeline tasks.

        Supports results offloaded to HTTP storage in this shape:
        {
            "transport": "http_url",
            "result_file_url": "http://.../file.npy",
            ...
        }

        Downstream DNN stages expect ``output_tensor`` blobs, so we attach
        a compatible ``output_tensor`` with ``file_url`` and a ``tensor_ref``.
        """
        parsed = upstream_result
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                try:
                    import ast as _ast

                    parsed = _ast.literal_eval(parsed)
                except Exception:
                    return upstream_result

        if not isinstance(parsed, dict):
            return parsed

        result_file_url = parsed.get("result_file_url")
        if result_file_url and parsed.get("transport") in {"http_url", "http", "url"}:
            extracted_result = None
            local_tensor_path = None

            # Map URL like http://host:8001/files/<name>.npy to local tensor_store path.
            try:
                from pathlib import Path
                from urllib.parse import urlparse

                url_path = urlparse(result_file_url).path
                file_name = Path(url_path).name
                candidate = Path.cwd() / "tensor_store" / file_name
                if candidate.exists():
                    local_tensor_path = str(candidate)
                    # Offloaded file may contain a JSON task_result envelope.
                    try:
                        envelope = json.loads(candidate.read_text(encoding="utf-8"))
                        extracted_result = (envelope.get("data") or {}).get("result")
                        if isinstance(extracted_result, str):
                            try:
                                extracted_result = json.loads(extracted_result)
                            except Exception:
                                import ast as _ast

                                extracted_result = _ast.literal_eval(extracted_result)
                    except Exception:
                        extracted_result = None
            except Exception:
                extracted_result = None

            if isinstance(extracted_result, dict):
                merged = dict(parsed)
                merged.update(extracted_result)
                return merged

            normalized = dict(parsed)
            normalized.setdefault(
                "output_tensor",
                {
                    "format": "npy",
                    "file_url": result_file_url,
                },
            )
            normalized.setdefault(
                "tensor_ref",
                (
                    {"path": local_tensor_path}
                    if local_tensor_path
                    else {"file_url": result_file_url}
                ),
            )
            return normalized

        return parsed

    async def _maybe_inject_upstream_result(
        self,
        session,
        job_id: str,
        downstream_task_id: str,
        upstream_task_id: str,
        upstream_result: Any,
    ):
        """
        If the downstream task's stage has pass_upstream_results=True,
        augment the downstream task's args with the upstream result.

        The injection strategy: downstream args become a dict with
        ``{"original_args": <existing>, "upstream_results": {upstream_task_id: result}}``
        so the worker function can access upstream outputs.
        """
        pipeline_meta = self._pipeline_metadata.get(job_id)
        if not pipeline_meta:
            return

        # Find the downstream task's stage index
        task = await get_task_by_id(session, downstream_task_id)
        if not task:
            return

        stage_idx = task.stage or 0
        stages_data = pipeline_meta.get("stages", [])
        if stage_idx >= len(stages_data):
            return

        stage_info = stages_data[stage_idx]
        if not stage_info.get("pass_upstream_results", False):
            return

        # Read current args and augment with upstream result
        try:
            resolved_args = resolve_text_ref(task.args)
            current_args = json.loads(resolved_args) if resolved_args else {}
        except (json.JSONDecodeError, TypeError):
            current_args = {}

        normalized_result = self._normalize_upstream_result_for_pipeline(
            upstream_result
        )

        # Build augmented args structure
        if isinstance(current_args, dict) and "upstream_results" in current_args:
            # Already has upstream_results — append
            current_args["upstream_results"][upstream_task_id] = normalized_result
        else:
            current_args = {
                "original_args": current_args,
                "upstream_results": {upstream_task_id: normalized_result},
            }

        updated_args = store_text_if_large(
            json.dumps(current_args), "task_args", downstream_task_id
        )
        await update_task_args(session, downstream_task_id, updated_args)
        print(
            f"DependencyManager: Injected result from {upstream_task_id} "
            f"into {downstream_task_id}"
        )

    # ------------------------------------------------------------------ #
    #  Pipeline helpers
    # ------------------------------------------------------------------ #

    def get_stage_func_code(self, job_id: str, stage: int) -> Optional[str]:
        """Get the function code for a specific pipeline stage."""
        codes = self._pipeline_func_codes.get(job_id, {})
        return codes.get(stage)

    def is_pipeline_job(self, job_id: str) -> bool:
        """Check if a job is a pipeline job."""
        return job_id in self._pipeline_metadata

    def get_pipeline_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline metadata for a job."""
        return self._pipeline_metadata.get(job_id)

    def cleanup_job(self, job_id: str):
        """Clean up cached pipeline data for a completed job."""
        self._pipeline_func_codes.pop(job_id, None)
        self._pipeline_metadata.pop(job_id, None)
        self.feature_router.clear_job(job_id)

    def __repr__(self) -> str:
        return f"DependencyManager(active_pipelines={len(self._pipeline_metadata)})"
