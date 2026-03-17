"""
Message handling logic separated by client type.

Changes from original
---------------------
WorkerMessageHandler now accepts an optional ``reconnection_handler``
(WorkerReconnectionHandler).  Two integration points:

1. ``_handle_worker_ready``  – calls ``reconnection_handler.handle_worker_reconnected``
   first; if it returns True the worker already received a RESUME_TASK message
   and normal task assignment is skipped.

2. ``handle_disconnection``  – NEW public method, called by the WebSocket
   server's close/disconnect path instead of (or after) any existing cleanup.
   Delegates to ``reconnection_handler.handle_worker_disconnected``.

All other behaviour is unchanged.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from websockets.server import WebSocketServerProtocol

from .utils import (
    _create_worker_in_database,
    _record_worker_failure,
    _update_worker_status,
    _update_worker_task_stats,
)
from common.protocol import (
    Message,
    MessageType,
    create_job_accepted_message,
    create_load_model_message,
    create_unload_model_message,
)
from common.serializer import get_runtime_info, bytes_to_hex, hex_to_bytes
from .staged_results_manager.checkpoint_manager import CheckpointManager
from .model_registry import build_model_artifact_url, store_partition_blob


class ClientMessageHandler:
    """
    Handles messages from client SDK

    Responsibilities:
    - Handle job submissions
    - Handle client disconnections
    - Send job acknowledgments
    """

    def __init__(
        self,
        connection_manager,
        job_manager,
        task_dispatcher,
        model_load_tracker=None,
    ):
        """
        Initialize client message handler

        Args:
            connection_manager: ConnectionManager instance
            job_manager: JobManager instance
            task_dispatcher: TaskDispatcher instance
            model_load_tracker: Optional tracker for model load readiness
        """
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher
        self.model_load_tracker = model_load_tracker

    @staticmethod
    def _infer_public_host_for_worker(
        worker_ws: WebSocketServerProtocol,
    ) -> Optional[str]:
        """Infer the foreman host a worker used to establish its WebSocket connection."""
        try:
            headers = getattr(worker_ws, "request_headers", None)
            if headers is None:
                return None
            host_header = headers.get("Host")
            if not host_header:
                return None
            return str(host_header)
        except Exception:
            return None

    @staticmethod
    def _estimate_model_load_warmup_seconds(storage_path: Optional[str]) -> float:
        """Estimate warmup wait based on artifact size and configured throughput."""

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        base_seconds = max(0.0, _env_float("MODEL_LOAD_BASE_SECONDS", 1.0))
        min_warmup = max(0.0, _env_float("MODEL_LOAD_MIN_WARMUP_SECONDS", 5.0))
        max_warmup = max(min_warmup, _env_float("MODEL_LOAD_MAX_WARMUP_SECONDS", 90.0))
        min_mbps = max(0.25, _env_float("MODEL_LOAD_MIN_MBPS", 8.0))

        size_bytes = 0
        if storage_path:
            try:
                size_bytes = os.path.getsize(storage_path)
            except OSError:
                size_bytes = 0

        size_mb = float(size_bytes) / (1024.0 * 1024.0)
        estimated = base_seconds + (size_mb / min_mbps)
        bounded = max(min_warmup, estimated)
        return min(max_warmup, bounded)

    @staticmethod
    def _estimate_model_load_dispatch_delay_seconds(warmup_seconds: float) -> float:
        """Estimate inter-message delay to avoid concurrent model loads on a worker."""

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        fixed_delay = max(0.0, _env_float("MODEL_LOAD_DISPATCH_DELAY_SECONDS", 0.0))
        if fixed_delay > 0.0:
            return fixed_delay

        delay_fraction = max(0.0, _env_float("MODEL_LOAD_DISPATCH_DELAY_FRACTION", 1.0))
        min_delay = max(0.0, _env_float("MODEL_LOAD_DISPATCH_MIN_DELAY_SECONDS", 1.0))
        max_delay = max(
            min_delay, _env_float("MODEL_LOAD_DISPATCH_MAX_DELAY_SECONDS", 30.0)
        )

        if warmup_seconds <= 0.0:
            return 0.0

        derived = warmup_seconds * delay_fraction
        bounded = max(min_delay, derived)
        return min(max_delay, bounded)

    async def _dispatch_load_model_instructions(
        self,
        job_id: str,
        model_version_id: str,
        topology_nodes,
        model_artifacts,
        stages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Persist artifacts, dispatch stage-0 loads, and defer later-stage loads.

        Returns:
            True when at least one stage-0/source LOAD_MODEL was dispatched.
        """
        if not model_artifacts:
            return False

        stage_index_by_name: Dict[str, int] = {}
        for stage_index, stage in enumerate(stages or []):
            stage_name = stage.get("name")
            if stage_name:
                stage_index_by_name[str(stage_name)] = stage_index

            stage_partition = stage.get("model_partition_id")
            if stage_partition:
                stage_index_by_name[str(stage_partition)] = stage_index

        partition_meta: Dict[str, Dict[str, Any]] = {}
        for node_index, node in enumerate(topology_nodes or []):
            partition_id = node.get("model_partition_id")
            if not partition_id:
                continue

            role = str(node.get("role", "intermediate")).lower()
            node_id = node.get("node_id")
            stage_index = None
            if node_id in stage_index_by_name:
                stage_index = stage_index_by_name[node_id]
            elif partition_id in stage_index_by_name:
                stage_index = stage_index_by_name[partition_id]
            elif role == "source":
                stage_index = 0

            partition_meta[partition_id] = {
                "role": role,
                "assigned_device_id": node.get("assigned_device_id"),
                "stage_index": stage_index,
                "node_index": node_index,
            }

        ordered_index_artifacts = sorted(
            enumerate(model_artifacts),
            key=lambda item: (
                partition_meta.get(item[1].get("model_partition_id"), {}).get(
                    "stage_index", 999
                ),
                (
                    0
                    if partition_meta.get(item[1].get("model_partition_id"), {}).get(
                        "role"
                    )
                    == "source"
                    else 1
                ),
                item[0],
            ),
        )

        stage0_load_dispatched = False

        for ordered_index, (_, artifact) in enumerate(ordered_index_artifacts):
            partition_id = artifact.get("model_partition_id")
            content_b64 = artifact.get("content_b64")
            if not partition_id or not content_b64:
                continue

            stored = store_partition_blob(
                model_version_id=model_version_id,
                model_partition_id=partition_id,
                content_b64=content_b64,
                file_name=artifact.get("file_name"),
            )

            meta = partition_meta.get(partition_id, {})
            role = str(meta.get("role", "intermediate")).lower()
            stage_index = meta.get("stage_index")
            if stage_index is None:
                stage_index = ordered_index

            assigned_worker = artifact.get("assigned_device_id") or meta.get(
                "assigned_device_id"
            )
            if not assigned_worker:
                available_workers = sorted(self.connection_manager.get_all_worker_ids())
                if not available_workers:
                    print(
                        f"ClientMessageHandler: No connected workers available for partition {partition_id}; skipping model dispatch"
                    )
                    continue

                # Prefer a worker that already has this model resident or cached
                best_worker = None
                if self.model_load_tracker:
                    resident = self.model_load_tracker.workers_with_model_resident(
                        partition_id
                    )
                    if resident:
                        best_worker = next(
                            (w for w in resident if w in available_workers), None
                        )
                    if not best_worker:
                        cached = self.model_load_tracker.workers_with_model_cached(
                            partition_id
                        )
                        if cached:
                            best_worker = next(
                                (w for w in cached if w in available_workers), None
                            )

                if best_worker:
                    assigned_worker = best_worker
                    print(
                        f"ClientMessageHandler: Affinity-assigned partition {partition_id} to worker {assigned_worker} (model already available)"
                    )
                else:
                    # Pick the worker with fewest resident models to spread load
                    worker_load = {
                        w: len(
                            self.model_load_tracker.workers_with_model_resident(w)
                            if self.model_load_tracker
                            else []
                        )
                        for w in available_workers
                    }
                    assigned_worker = min(
                        available_workers, key=lambda w: worker_load.get(w, 0)
                    )
                    print(
                        f"ClientMessageHandler: No affinity for partition {partition_id}; selected least-loaded worker {assigned_worker}"
                    )

            warmup_seconds = self._estimate_model_load_warmup_seconds(
                stored.get("storage_path")
            )

            if self.model_load_tracker:
                self.model_load_tracker.set_job_model_version(job_id, model_version_id)
                self.model_load_tracker.set_stage_partition_id(
                    job_id, stage_index, partition_id
                )

            # Stage-0 / source partitions: broadcast LOAD_MODEL to ALL workers
            if stage_index == 0 or role == "source":
                all_stage0_workers = sorted(
                    self.connection_manager.get_all_worker_ids()
                )
                for target_wid in all_stage0_workers:
                    target_ws = self.connection_manager.get_worker_websocket(target_wid)
                    if not target_ws:
                        continue
                    host_hint = self._infer_public_host_for_worker(target_ws)
                    target_model_url = build_model_artifact_url(
                        model_version_id,
                        stored["file_name"],
                        host_override=host_hint,
                    )
                    use_cache = bool(
                        self.model_load_tracker
                        and (
                            self.model_load_tracker.is_model_resident(
                                target_wid, partition_id
                            )
                            or self.model_load_tracker.is_model_cached(
                                target_wid, partition_id
                            )
                        )
                    )
                    load_msg = create_load_model_message(
                        job_id=job_id,
                        model_version_id=model_version_id,
                        model_partition_id=partition_id,
                        model_uri=target_model_url,
                        checksum=stored["checksum"],
                        from_cache=use_cache,
                    )
                    await target_ws.send(load_msg.to_json())
                    if self.model_load_tracker:
                        self.model_load_tracker.mark_dispatched(
                            target_wid,
                            partition_id,
                            warmup_seconds=warmup_seconds,
                        )
                    print(
                        f"ClientMessageHandler: Sent stage-0 LOAD_MODEL for "
                        f"partition {partition_id} to worker {target_wid}"
                    )

                if self.model_load_tracker:
                    self.model_load_tracker.set_loaded_stage(job_id, 0)
                stage0_load_dispatched = True
                continue

            # For later stages, need the assigned worker's websocket
            worker_ws = self.connection_manager.get_worker_websocket(assigned_worker)
            if not worker_ws:
                print(
                    f"ClientMessageHandler: Worker {assigned_worker} unavailable for partition {partition_id}; skipping dispatch"
                )
                continue

            host_hint = self._infer_public_host_for_worker(worker_ws)
            model_url = build_model_artifact_url(
                model_version_id,
                stored["file_name"],
                host_override=host_hint,
            )

            if self.model_load_tracker:
                self.model_load_tracker.store_deferred_load(
                    job_id=job_id,
                    stage_index=stage_index,
                    partition_id=partition_id,
                    worker_id=assigned_worker,
                    model_url=model_url,
                    model_version_id=model_version_id,
                    checksum=stored.get("checksum"),
                    warmup_seconds=warmup_seconds,
                )
                print(
                    f"ClientMessageHandler: Deferred LOAD_MODEL for stage {stage_index} partition {partition_id} on worker {assigned_worker}"
                )
                continue

            # Fallback path if tracker is unavailable: dispatch immediately.
            load_msg = create_load_model_message(
                job_id=job_id,
                model_version_id=model_version_id,
                model_partition_id=partition_id,
                model_uri=model_url,
                checksum=stored["checksum"],
            )
            await worker_ws.send(load_msg.to_json())

            dispatch_delay = self._estimate_model_load_dispatch_delay_seconds(
                warmup_seconds
            )
            if dispatch_delay > 0.0:
                await asyncio.sleep(dispatch_delay)

        return stage0_load_dispatched

    async def dispatch_stage_model_load(self, job_id: str, stage_index: int) -> int:
        """Dispatch deferred model loads for a newly unblocked pipeline stage."""
        if not self.model_load_tracker:
            return 0

        deferred_loads = self.model_load_tracker.pop_deferred_loads_for_stage(
            job_id, stage_index
        )
        if not deferred_loads:
            return 0

        previous_stage = self.model_load_tracker.get_loaded_stage(job_id)
        previous_partition_id = None
        if previous_stage is not None:
            previous_partition_id = self.model_load_tracker.get_stage_partition_id(
                job_id, previous_stage
            )
        if previous_partition_id is None and stage_index > 0:
            previous_partition_id = self.model_load_tracker.get_stage_partition_id(
                job_id, stage_index - 1
            )

        dispatched = 0
        unload_sent_workers = set()

        for load_info in deferred_loads:
            partition_id = load_info.get("partition_id")
            model_url = load_info.get("model_url")
            model_version_id = load_info.get(
                "model_version_id"
            ) or self.model_load_tracker.get_job_model_version(job_id)
            checksum = load_info.get("checksum")
            warmup_seconds = load_info.get("warmup_seconds")

            if not partition_id or not model_url or not model_version_id:
                continue

            worker_id = load_info.get("worker_id")
            if not worker_id:
                available_workers = sorted(self.connection_manager.get_all_worker_ids())
                if not available_workers:
                    print(
                        f"ClientMessageHandler: No workers available to dispatch deferred stage load for partition {partition_id}"
                    )
                    continue
                worker_id = available_workers[0]

            worker_ws = self.connection_manager.get_worker_websocket(worker_id)
            if not worker_ws:
                print(
                    f"ClientMessageHandler: Worker {worker_id} unavailable for deferred partition {partition_id}"
                )
                continue

            # Smart skip: if the worker already has this partition loaded,
            # skip both unload and load — zero cost.
            if self.model_load_tracker.is_model_resident(worker_id, partition_id):
                self.model_load_tracker.mark_ready(worker_id, partition_id)
                self.model_load_tracker.set_stage_partition_id(
                    job_id, stage_index, partition_id
                )
                dispatched += 1
                print(
                    f"ClientMessageHandler: Worker {worker_id} already has partition "
                    f"{partition_id} resident — skipping unload/load"
                )
                continue

            # Smart skip: if the worker has this partition cached on disk,
            # send LOAD_MODEL with from_cache flag to skip HTTP download.
            from_cache = self.model_load_tracker.is_model_cached(
                worker_id, partition_id
            )

            if (
                previous_partition_id
                and worker_id not in unload_sent_workers
                and model_version_id
            ):
                unload_msg = create_unload_model_message(
                    job_id=job_id,
                    model_version_id=model_version_id,
                    model_partition_id=previous_partition_id,
                )
                await worker_ws.send(unload_msg.to_json())
                unload_sent_workers.add(worker_id)
                self.model_load_tracker.mark_model_unloaded(
                    worker_id, previous_partition_id
                )
                print(
                    f"ClientMessageHandler: Sent UNLOAD_MODEL for partition {previous_partition_id} to worker {worker_id}"
                )

            load_msg = create_load_model_message(
                job_id=job_id,
                model_version_id=model_version_id,
                model_partition_id=partition_id,
                model_uri=model_url,
                checksum=checksum,
                from_cache=from_cache,
            )
            await worker_ws.send(load_msg.to_json())
            self.model_load_tracker.mark_dispatched(
                worker_id,
                partition_id,
                warmup_seconds=warmup_seconds,
            )
            self.model_load_tracker.set_stage_partition_id(
                job_id, stage_index, partition_id
            )
            dispatched += 1

            cache_note = " (from cache)" if from_cache else ""
            print(
                f"ClientMessageHandler: Sent deferred LOAD_MODEL for stage {stage_index} "
                f"partition {partition_id} to worker {worker_id}{cache_note}"
            )

        if dispatched > 0:
            self.model_load_tracker.set_loaded_stage(job_id, stage_index)

        return dispatched

    async def handle_message(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        if message.type == MessageType.SUBMIT_JOB:
            await self._handle_job_submission(message, websocket)
        elif message.type == MessageType.SUBMIT_PIPELINE_JOB:
            await self._handle_pipeline_submission(message, websocket)
        elif message.type == MessageType.DISCONNECT:
            await websocket.close()
        else:
            print(f"ClientMessageHandler: Unknown message type: {message.type}")

    async def _handle_job_submission(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            func_code = message.data["func_code"]
            args_list = message.data["args_list"]
            total_tasks = message.data["total_tasks"]
            
            # Extract task metadata for declarative checkpointing
            task_metadata = message.data.get("task_metadata")

            print(
                f"ClientMessageHandler: Received job {job_id} with {total_tasks} tasks | foreman_runtime={get_runtime_info()}"
            )
            
            # Log checkpoint configuration if present
            if task_metadata and task_metadata.get("checkpoint_enabled"):
                print(
                    f"ClientMessageHandler: Job {job_id} has checkpointing enabled "
                    f"(interval={task_metadata.get('checkpoint_interval')}s, "
                    f"state_vars={task_metadata.get('checkpoint_state', [])})"
                )

            self.connection_manager.add_client(job_id, websocket)
            await self.job_manager.create_job(
                job_id, func_code, args_list, total_tasks, task_metadata=task_metadata
            )
            assigned = await self.task_dispatcher.assign_tasks_for_job(
                job_id, func_code, args_list
            )
            print(f"ClientMessageHandler: Assigned {assigned} tasks immediately for job {job_id}")

            response = create_job_accepted_message(job_id)
            await websocket.send(response.to_json())
            print(f"ClientMessageHandler: Job {job_id} accepted and acknowledged")

        except KeyError as e:
            print(f"ClientMessageHandler: Missing required field in job submission: {e}")
            error_msg = Message(MessageType.JOB_ERROR, {"error": f"Missing required field: {e}"}, message.job_id)
            await websocket.send(error_msg.to_json())
        except Exception as e:
            print(f"ClientMessageHandler: Error handling job submission: {e}")
            import traceback; traceback.print_exc()
            error_msg = Message(MessageType.JOB_ERROR, {"error": str(e)}, message.job_id)
            await websocket.send(error_msg.to_json())

    async def _handle_pipeline_submission(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            stages = message.data["stages"]
            total_tasks = message.data["total_tasks"]
            total_stages = message.data["total_stages"]
            dependency_map = message.data.get("dependency_map")
            task_metadata = message.data.get("task_metadata")
            dnn_config = message.data.get("dnn_config")
            pipeline_mode = message.data.get("pipeline_mode", "barrier")

            print(
                f"ClientMessageHandler: Received pipeline job {job_id} "
                f"with {total_stages} stages, {total_tasks} tasks | "
                f"foreman_runtime={get_runtime_info()}"
            )

            for i, stage in enumerate(stages):
                stage_name = stage.get("name", f"stage_{i}")
                stage_tasks = len(stage["args_list"])
                pass_results = stage.get("pass_upstream_results", False)
                print(
                    f"  Stage {i} ({stage_name}): {stage_tasks} tasks"
                    f"{' [receives upstream results]' if pass_results else ''}"
                )

            self.connection_manager.add_client(job_id, websocket)

            # Create pipeline job with dependency wiring
            created = await self.job_manager.create_pipeline_job(
                job_id, stages, dependency_map, task_metadata
            )

            # Dispatch Stage 0 tasks (they are "pending" from creation)
            stage_0_func_code = stages[0]["func_code"]
            assigned = await self.task_dispatcher.assign_tasks_for_job(
                job_id, stage_0_func_code, stages[0]["args_list"]
            )

            print(
                f"ClientMessageHandler: Pipeline job {job_id} — "
                f"Assigned {assigned} stage-0 tasks immediately"
            )

            response = create_job_accepted_message(job_id)
            await websocket.send(response.to_json())

            print(f"ClientMessageHandler: Pipeline job {job_id} accepted and acknowledged")

        except KeyError as e:
            print(f"ClientMessageHandler: Missing required field in pipeline submission: {e}")
            error_msg = Message(MessageType.JOB_ERROR, {"error": f"Missing required field: {e}"}, message.job_id)
            await websocket.send(error_msg.to_json())
        except Exception as e:
            print(f"ClientMessageHandler: Error handling pipeline submission: {e}")
            import traceback
            traceback.print_exc()

            error_msg = Message(
                MessageType.JOB_ERROR, {"error": str(e)}, message.job_id
            )
            await websocket.send(error_msg.to_json())


class WorkerMessageHandler:
    """
    Handles messages from workers.

    Responsibilities:
    - Handle worker registration (with transparent checkpoint-based resume on reconnect)
    - Handle task results / errors / checkpoints / pongs
    - Graceful disconnection → reconnection lifecycle via reconnection_handler
    """

    def __init__(
        self,
        connection_manager,
        job_manager,
        task_dispatcher,
        completion_handler,
        checkpoint_manager: CheckpointManager = None,
    ):
        """
        Initialize worker message handler

        Args:
            connection_manager: ConnectionManager instance
            job_manager: JobManager instance
            task_dispatcher: TaskDispatcher instance
            completion_handler: JobCompletionHandler instance
            checkpoint_manager: CheckpointManager instance for checkpoint handling
        """
        self.connection_manager = connection_manager
        self.job_manager = job_manager
        self.task_dispatcher = task_dispatcher
        self.completion_handler = completion_handler
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

    async def handle_message(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        if message.type == MessageType.WORKER_READY:
            await self._handle_worker_ready(message, websocket)
        elif message.type == MessageType.TASK_RESULT:
            await self._handle_task_result(message, websocket)
        elif message.type == MessageType.TASK_ERROR:
            await self._handle_task_error(message, websocket)
        elif message.type == MessageType.PONG:
            await self._handle_pong(message, websocket)
        elif message.type == MessageType.WORKER_HEARTBEAT:
            await self._handle_pong(message, websocket)
        elif message.type == MessageType.MODEL_LOADED:
            await self._handle_model_loaded(message, websocket)
        elif message.type == MessageType.TASK_CHECKPOINT:
            await self._handle_task_checkpoint(message, websocket)
        elif message.type == MessageType.INTERMEDIATE_FEATURE:
            await self._handle_intermediate_feature(message, websocket)
        else:
            print(f"WorkerMessageHandler: Unknown message type: {message.type}")

    # ------------------------------------------------------------------
    # Individual handlers
    # ------------------------------------------------------------------

    async def _handle_pong(self, message: Message, websocket: WebSocketServerProtocol):
        worker_id = self.connection_manager.find_worker_by_websocket(websocket)
        if worker_id:
            await _update_worker_status(worker_id, "online")
            if "performance_metrics" in message.data:
                await self._update_worker_performance_metrics(worker_id, message.data["performance_metrics"])

    async def _update_worker_performance_metrics(self, worker_id: str, metrics: dict):
        try:
            from foreman.db.base import async_session
            from foreman.db.models import WorkerModel
            from sqlalchemy import update
            from datetime import datetime

            async with async_session() as session:
                stmt = (
                    update(WorkerModel)
                    .where(WorkerModel.id == worker_id)
                    .values(
                        cpu_usage_percent=metrics.get("cpu_usage_percent"),
                        ram_available_mb=metrics.get("ram_available_mb"),
                        battery_level=metrics.get("battery_level"),
                        is_charging=metrics.get("is_charging"),
                        network_speed_mbps=metrics.get("network_speed_mbps"),
                        storage_available_gb=metrics.get("storage_available_gb"),
                        last_performance_update=datetime.now(),
                    )
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            print(f"WorkerMessageHandler: Error updating performance metrics for {worker_id}: {e}")

    @staticmethod
    def _is_model_not_loaded_error(error: str) -> bool:
        if not error:
            return False
        lowered = str(error).lower()
        return "model partition" in lowered and "not loaded" in lowered

    @staticmethod
    def _extract_model_partition_id(error: str) -> Optional[str]:
        if not error:
            return None

        match = re.search(
            r"model partition\s+[\"']?([^\"'\s]+)[\"']?",
            str(error),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _extract_stage_index_from_task_id(task_id: str) -> Optional[int]:
        """Extract stage index from task IDs formatted as <job>_task_<stage>_<idx>."""
        if not task_id:
            return None

        match = re.search(r"_task_(\d+)_\d+$", str(task_id))
        if not match:
            return None

        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    async def _handle_model_loaded(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """Mark a worker partition as ready and trigger assignment."""
        worker_id = self.connection_manager.find_worker_by_websocket(websocket)
        if not worker_id:
            print("WorkerMessageHandler: Could not map websocket for MODEL_LOADED")
            return

        model_partition_id = message.data.get("model_partition_id")
        if not model_partition_id:
            print("WorkerMessageHandler: MODEL_LOADED missing model_partition_id field")
            return

        if self.model_load_tracker:
            self.model_load_tracker.mark_ready(worker_id, model_partition_id)

        print(
            f"WorkerMessageHandler: Worker {worker_id} reported model partition ready: {model_partition_id}"
        )

        # Opportunistically assign work now that the partition is confirmed ready.
        await self.task_dispatcher.assign_task_to_available_worker(worker_id)

    async def _handle_worker_ready(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """
        Handle worker ready message (worker registration)
        
        Extended to extract worker capabilities for mobile support.
        Workers report their platform, runtime, and capabilities, which
        determines if code instrumentation is needed.

        Args:
            message: Worker ready message
            websocket: Worker websocket connection
        """
        try:
            worker_id = message.data["worker_id"]
            device_specs = message.data.get("device_specs", {})
            
            # Extract platform/runtime info for mobile worker support
            worker_type = message.data.get("worker_type", "pc_python")
            platform = message.data.get("platform", "unknown")
            runtime = message.data.get("runtime", "cpython")
            model_runtime = message.data.get("model_runtime")
            capabilities = message.data.get("capabilities", {})

            # Keep runtime metadata in device specs for DB persistence.
            if runtime and "runtime" not in device_specs:
                device_specs["runtime"] = runtime
            if model_runtime and "model_runtime" not in device_specs:
                device_specs["model_runtime"] = model_runtime

            print(f"WorkerMessageHandler: Worker {worker_id} connected")
            print(f"  Type: {worker_type} | Platform: {platform} | Runtime: {runtime}")

            if device_specs:
                device_type = device_specs.get("device_type", "Unknown")
                os_info = f"{device_specs.get('os_type', 'Unknown')} {device_specs.get('os_version', '')}"
                cpu = device_specs.get("cpu_model", "Unknown")
                ram = device_specs.get("ram_total_mb")
                ram_info = f"{ram:.0f} MB" if ram else "Unknown"
                print(f"  📱 Device: {device_type} | OS: {os_info}")
                print(f"  🖥️  CPU: {cpu}")
                print(f"  💾 RAM: {ram_info}")
            
            # Log capabilities for mobile workers
            if capabilities:
                settrace_support = capabilities.get("supports_settrace", True)
                frame_support = capabilities.get("supports_frame_introspection", True)
                if not settrace_support or not frame_support:
                    print(f"  ⚠️  Limited capabilities: settrace={settrace_support}, frame_introspection={frame_support}")

            # ---- Register worker in connection manager ----
            from .connection_manager import WorkerInfo

            worker_info = WorkerInfo.from_worker_ready_message(
                worker_id=worker_id,
                websocket=websocket,
                message_data=message.data
            )
            
            # Register worker with capabilities
            self.connection_manager.add_worker(worker_id, websocket, worker_info=worker_info)

            # Create or update worker in database with device specs
            await _create_worker_in_database(worker_id, device_specs)
            await _update_worker_status(worker_id, "online")

            # Assign any pending tasks to this worker
            assigned = await self.task_dispatcher.assign_task_to_available_worker(
                worker_id
            )

            if assigned:
                print(
                    f"WorkerMessageHandler: Worker {worker_id} reconnected with DB-assigned task "
                    f"{db_task_id} (job {db_job_id}) \u2013 no new task assigned."
                )
                return

            # No explicit RESUME_TASK dispatch here; worker continues execution
            # on its own after reconnect and keeps sending checkpoints/results.

            # ---- Normal path: assign a pending task ----
            assigned = await self.task_dispatcher.assign_task_to_available_worker(worker_id)
            if assigned:
                print(f"WorkerMessageHandler: Assigned task to newly connected worker {worker_id}")
            else:
                print(f"WorkerMessageHandler: No tasks available for worker {worker_id}")

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in worker ready: {e}")
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling worker ready: {e}")
            import traceback; traceback.print_exc()

    async def _handle_task_result(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            result = message.data["result"]

            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if not worker_id:
                print(f"[RESULT DEBUG] Could not find worker for task result – task_id={task_id}, job_id={job_id}")
                return

            print(f"[Task Result] Received from worker {worker_id} | Task: {task_id} | Job: {job_id}")

            # Normalise result payload
            normalized_result = result
            if isinstance(normalized_result, str):
                try:
                    normalized_result = json.loads(normalized_result)
                except Exception:
                    try:
                        import ast as _ast
                        normalized_result = _ast.literal_eval(normalized_result)
                    except Exception:
                        normalized_result = result

            if isinstance(normalized_result, dict):
                normalized_result.setdefault("worker_id", worker_id)
                trace = normalized_result.get("trace")
                if isinstance(trace, list) and trace and isinstance(trace[-1], dict):
                    trace[-1].setdefault("worker_id", worker_id)
                    trace[-1].setdefault("task_id", task_id)
                try:
                    result_str = json.dumps(normalized_result)
                except Exception:
                    result_str = str(result) if result is not None else ""
            else:
                result_str = str(result) if result is not None else ""
            
            print(f"[Task Result] Result preview: {result_str[:100]}..." if len(result_str) > 100 else f"📊 [Task Result] Result: {result_str}")

            # Mark task as completed in job manager (idempotent)
            accepted, job_complete = await self.job_manager.mark_task_completed(
                task_id, job_id, worker_id, result_str
            )

            if not accepted:
                print(f"[RESULT DEBUG] Ignoring duplicate/stale completion for task {task_id} on worker {worker_id}")
                self.connection_manager.clear_worker_active_task(worker_id)
                self.connection_manager.mark_worker_available(worker_id)
                await _update_worker_status(worker_id, "online", current_task_id=None)
                return

            await _update_worker_task_stats(worker_id, task_completed=True)
            self.connection_manager.clear_worker_active_task(worker_id)
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)

            # Pipeline dependency resolution
            pipeline_batch_dispatched = False
            if self.job_manager.is_pipeline_job(job_id):
                dep_mgr = self.job_manager.dependency_manager
                newly_unblocked = await dep_mgr.on_task_completed(task_id, job_id, normalized_result)
                if newly_unblocked:
                    print(
                        f"[PIPELINE] 🚀 Stage barrier lifted! {len(newly_unblocked)} "
                        f"downstream tasks unblocked for job {job_id}"
                    )
                    # Batch-dispatch ALL unblocked tasks to the best available
                    # workers (including the current worker and any idle ones).
                    func_code = self.job_manager.get_func_code(job_id)
                    batch_assigned = await self.task_dispatcher.assign_tasks_for_job(
                        job_id, func_code or "", []
                    )
                    print(
                        f"[PIPELINE] Batch-dispatched {batch_assigned} next-stage "
                        f"tasks across available workers"
                    )
                    pipeline_batch_dispatched = True

            if job_complete:
                print(f"[RESULT DEBUG] Job {job_id} completed, triggering completion handler")
                await self.completion_handler.handle_job_completion(job_id)

            if not pipeline_batch_dispatched:
                assigned = await self.task_dispatcher.assign_task_to_available_worker(worker_id)
                if assigned:
                    print(f"[RESULT DEBUG] Assigned next task to worker {worker_id}")

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in task result: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"[RESULT DEBUG] Error handling task result: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_task_error(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            error = message.data["error"]

            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if not worker_id:
                print("WorkerMessageHandler: Could not find worker for task error")
                return

            print(f"WorkerMessageHandler: Task {task_id} failed on worker {worker_id} for job {job_id}: {error}")

            await self.job_manager.mark_task_failed(task_id, job_id, worker_id, error)
            await _update_worker_task_stats(worker_id, task_completed=False)

            try:
                await _record_worker_failure(worker_id, task_id, error, job_id)
            except Exception as ex:
                print(f"WorkerMessageHandler: Error recording worker failure for {worker_id}/{task_id}: {ex}")

            self.connection_manager.clear_worker_active_task(worker_id)
            self.connection_manager.mark_worker_available(worker_id)
            await _update_worker_status(worker_id, "online", current_task_id=None)

            # Assign next task to this worker (the failed task will be retried by another worker)
            assigned = await self.task_dispatcher.assign_task_to_available_worker(
                worker_id
            )

            if assigned:
                print(f"WorkerMessageHandler: Assigned next task to worker {worker_id} after failure")

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in task error: {e}")
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling task error: {e}")
            import traceback; traceback.print_exc()

    async def _handle_task_checkpoint(
        self, message: Message, websocket: WebSocketServerProtocol
    ):
        """
        Handle checkpoint message from worker (incremental state save)
        
        Supports both legacy checkpoints and declarative checkpointing with
        additional metadata (checkpoint_type, checkpoint_state_vars, state_size_bytes).

        Args:
            message: Task checkpoint message
            websocket: Worker websocket connection
        """
        try:
            job_id = message.job_id
            task_id = message.data["task_id"]
            is_base = message.data["is_base"]
            delta_data_hex = message.data["delta_data_hex"]
            progress_percent = message.data["progress_percent"]
            checkpoint_id = message.data["checkpoint_id"]
            compression_type = message.data.get("compression_type", "gzip")
            
            # Extract declarative checkpointing metadata (optional)
            checkpoint_type = message.data.get("checkpoint_type")
            checkpoint_state_vars = message.data.get("checkpoint_state_vars")
            state_size_bytes = message.data.get("state_size_bytes")

            worker_id = self.connection_manager.find_worker_by_websocket(websocket)
            if not worker_id:
                print("WorkerMessageHandler: Could not find worker for checkpoint")
                return

            checkpoint_data_bytes = hex_to_bytes(delta_data_hex)
            
            # Build log message with declarative checkpoint details if present
            state_vars_info = ""
            if checkpoint_state_vars:
                state_vars_info = f", state_vars: {checkpoint_state_vars}"

            print(
                f"WorkerMessageHandler: Received checkpoint {checkpoint_id} from worker {worker_id} "
                f"for task {task_id} (size: {len(checkpoint_data_bytes)} bytes, "
                f"progress: {progress_percent}%{state_vars_info})"
            )

            from foreman.db.base import get_db_session
            async with get_db_session() as session:
                success = await self.checkpoint_manager.store_checkpoint(
                    session=session,
                    task_id=task_id,
                    job_id=job_id,
                    is_base=is_base,
                    delta_data_bytes=checkpoint_data_bytes,
                    progress_percent=progress_percent,
                    checkpoint_id=checkpoint_id,
                    compression_type=compression_type,
                    checkpoint_type=checkpoint_type,
                    checkpoint_state_vars=checkpoint_state_vars,
                    state_size_bytes=state_size_bytes,
                )

            if success:
                print(f"WorkerMessageHandler: Checkpoint {checkpoint_id} stored for task {task_id}")
                from common.protocol import create_checkpoint_ack_message
                ack_msg = create_checkpoint_ack_message(task_id, job_id, checkpoint_id)
                await websocket.send(ack_msg.to_json())
            else:
                print(f"WorkerMessageHandler: Failed to store checkpoint {checkpoint_id} for task {task_id}")

        except KeyError as e:
            print(f"WorkerMessageHandler: Missing required field in checkpoint message: {e}")
        except Exception as e:
            print(f"WorkerMessageHandler: Error handling checkpoint: {e}")
            import traceback

            traceback.print_exc()
