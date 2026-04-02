import asyncio
import json
import uuid
from pathlib import Path
import websockets
from typing import Any, Callable, List, Optional, Dict
from common.protocol import (
    Message,
    MessageType,
    create_submit_job_message,
    create_submit_job_message_with_metadata,
    create_submit_pipeline_job_message,
    create_ping_message,
    create_pong_message,
    create_intermediate_feature_message,
)
from common.serializer import (
    serialize_function,
    encode_feature_payload,
    decode_feature_payload,
)
from common.code_instrumenter import (
    instrument_for_task_control,
    generate_task_control_wrapper,
)
from .decorators import get_task_metadata, TaskMetadata
from .model_artifacts import build_partition_artifact
from .topology import validate_topology


def _native_model_stage_passthrough(task_input: Any) -> Any:
    """Fallback stage callable for native runtimes that execute model partitions directly."""
    return task_input


class CrowdComputeClient:
    """Main client for interacting with CrowdCompute foreman"""

    def __init__(self):
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.foreman_host: Optional[str] = None
        self.foreman_port: Optional[int] = None
        self.connected = False
        self.pending_jobs: Dict[str, asyncio.Future] = {}
        self._listen_task: Optional[asyncio.Task] = None
        self._submitted_jobs: Dict[str, Dict[str, Any]] = {}

    async def connect(self, host: str, port: int = 9000):
        """Connect to foreman server"""
        try:
            self.foreman_host = host
            self.foreman_port = port
            uri = f"ws://{host}:{port}"

            print(f"Connecting to foreman at {uri}...")
            # Increase max_size and disable built-in ping to match foreman server settings.
            # Large job results (e.g., base64 encoded images) routinely exceed the
            # 1 MiB default limit, which previously caused the foreman to close
            # the connection while streaming results back to the client.
            self.websocket = await websockets.connect(
                uri,
                max_size=None,  # Allow arbitrarily large result payloads
                ping_interval=None,
                ping_timeout=None,
                close_timeout=30,
            )
            self.connected = True

            # Start listening for responses
            self._listen_task = asyncio.create_task(self._listen_for_messages())

            print(f"Connected to foreman at {uri}")

        except Exception as e:
            print(f"Failed to connect to foreman: {e}")
            raise

    async def disconnect(self):
        """Disconnect from foreman server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False

        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None

        print("Disconnected from foreman")

    async def _listen_for_messages(self):
        """Listen for messages from foreman"""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("Connection to foreman closed")
            self.connected = False
        except Exception as e:
            print(f"Error in message listener: {e}")
            self.connected = False

    async def _handle_message(self, message_str: str):
        """Handle incoming message from foreman"""
        try:
            message = Message.from_json(message_str)

            if message.type == MessageType.JOB_RESULTS:
                job_id = message.job_id
                if job_id in self.pending_jobs:
                    future = self.pending_jobs.pop(job_id)
                    future.set_result(message.data["results"])

            elif message.type == MessageType.JOB_ERROR:
                job_id = message.job_id
                if job_id in self.pending_jobs:
                    future = self.pending_jobs.pop(job_id)
                    future.set_exception(Exception(message.data["error"]))

            elif message.type == MessageType.PING:
                # Respond to ping
                pong = create_pong_message()
                await self.websocket.send(pong.to_json())

        except Exception as e:
            print(f"Error handling message: {e}")

    async def map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """
        Map function over iterable using distributed workers

        Extracts checkpoint metadata from @task decorated functions and
        sends it with the job submission for checkpoint-aware execution.

        Args:
            func: Function to execute (optionally decorated with @task)
            iterable: List of arguments to map over
            **kwargs: Additional options:
                - checkpoint: Override checkpoint setting
                - checkpoint_interval: Override checkpoint interval

        Returns:
            List of results from workers
        """
        if not self.connected:
            raise RuntimeError("Not connected to foreman. Call connect() first.")

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create future for this job
        future = asyncio.Future()
        self.pending_jobs[job_id] = future

        try:
            # Extract metadata from decorated function (if present)
            metadata = get_task_metadata(func)

            # Apply any overrides from kwargs
            if metadata:
                task_metadata = TaskMetadata(
                    checkpoint_enabled=kwargs.get(
                        "checkpoint", metadata.checkpoint_enabled
                    ),
                    checkpoint_interval=kwargs.get(
                        "checkpoint_interval", metadata.checkpoint_interval
                    ),
                    checkpoint_state=kwargs.get(
                        "checkpoint_state", metadata.checkpoint_state
                    ),
                    parallel=kwargs.get("parallel", metadata.parallel),
                    retry_on_failure=kwargs.get(
                        "retry_on_failure", metadata.retry_on_failure
                    ),
                    max_retries=kwargs.get("max_retries", metadata.max_retries),
                    timeout=kwargs.get("timeout", metadata.timeout),
                    _func_name=metadata._func_name,
                )
            else:
                # Create default metadata if not decorated
                task_metadata = TaskMetadata(
                    checkpoint_enabled=kwargs.get("checkpoint", False),
                    checkpoint_interval=kwargs.get("checkpoint_interval", 10.0),
                    checkpoint_state=kwargs.get("checkpoint_state", []),
                    parallel=kwargs.get("parallel", True),
                    retry_on_failure=kwargs.get("retry_on_failure", True),
                    max_retries=kwargs.get("max_retries", 3),
                    timeout=kwargs.get("timeout", None),
                    _func_name=func.__name__,
                )

            # Serialize function (use original if decorated)
            if hasattr(func, "__crowdio_original__"):
                func_code = serialize_function(func.__crowdio_original__)
            else:
                func_code = serialize_function(func)

            # Pre-instrument with task control (pause/kill) at SDK level
            # This is a static transformation that reduces foreman load
            checkpoint_state_vars = task_metadata.checkpoint_state or []
            func_code, num_funcs, num_ctrl_loops = instrument_for_task_control(
                func_code, checkpoint_state_vars=checkpoint_state_vars
            )
            control_wrapper = generate_task_control_wrapper()
            func_code = control_wrapper + "\n" + func_code
            if num_funcs > 0:
                print(
                    f"[SDK] Task control: instrumented {num_funcs} functions, "
                    f"{num_ctrl_loops} loops"
                )

            # Create submission message with metadata
            message = create_submit_job_message_with_metadata(
                func_code, iterable, job_id, task_metadata.to_dict()
            )

            # Store job metadata for tracking
            self._submitted_jobs[job_id] = {
                "metadata": task_metadata,
                "func_name": task_metadata._func_name,
                "total_tasks": len(iterable),
                "checkpoint_enabled": task_metadata.checkpoint_enabled,
            }

            # Send to foreman
            await self.websocket.send(message.to_json())

            if task_metadata.checkpoint_enabled:
                print(
                    f"[SDK] Job {job_id[:8]}... submitted with checkpointing enabled "
                    f"(interval: {task_metadata.checkpoint_interval}s, "
                    f"state vars: {task_metadata.checkpoint_state or 'all'})"
                )
            else:
                print(f"[SDK] Job {job_id[:8]}... submitted (checkpointing disabled)")

            # Wait for results
            results = await future
            return results

        except Exception as e:
            # Clean up on error
            if job_id in self.pending_jobs:
                del self.pending_jobs[job_id]
            if job_id in self._submitted_jobs:
                del self._submitted_jobs[job_id]
            raise

    async def submit(self, func: Callable, iterable: List[Any], **kwargs) -> str:
        """
        Submit a job asynchronously without waiting for results

        Args:
            func: Function to execute
            iterable: List of arguments for tasks
            **kwargs: Additional options

        Returns:
            job_id: Identifier to retrieve results later
        """
        if not self.connected:
            raise RuntimeError("Not connected to foreman. Call connect() first.")

        job_id = str(uuid.uuid4())

        # Extract metadata
        metadata = get_task_metadata(func)
        if metadata:
            task_metadata = TaskMetadata(
                checkpoint_enabled=kwargs.get(
                    "checkpoint", metadata.checkpoint_enabled
                ),
                checkpoint_interval=kwargs.get(
                    "checkpoint_interval", metadata.checkpoint_interval
                ),
                checkpoint_state=kwargs.get(
                    "checkpoint_state", metadata.checkpoint_state
                ),
                parallel=metadata.parallel,
                retry_on_failure=metadata.retry_on_failure,
                max_retries=metadata.max_retries,
                timeout=metadata.timeout,
                _func_name=metadata._func_name,
            )
        else:
            task_metadata = TaskMetadata(
                checkpoint_enabled=kwargs.get("checkpoint", False),
                checkpoint_interval=kwargs.get("checkpoint_interval", 10.0),
                _func_name=func.__name__,
            )

        # Serialize function
        if hasattr(func, "__crowdio_original__"):
            func_code = serialize_function(func.__crowdio_original__)
        else:
            func_code = serialize_function(func)

        # Pre-instrument with task control (pause/kill) at SDK level
        checkpoint_state_vars = task_metadata.checkpoint_state or []
        func_code, num_funcs, num_ctrl_loops = instrument_for_task_control(
            func_code, checkpoint_state_vars=checkpoint_state_vars
        )
        control_wrapper = generate_task_control_wrapper()
        func_code = control_wrapper + "\n" + func_code
        if num_funcs > 0:
            print(
                f"[SDK] Task control: instrumented {num_funcs} functions, "
                f"{num_ctrl_loops} loops"
            )

        # Create future for this job
        future = asyncio.Future()
        self.pending_jobs[job_id] = future

        # Store job metadata
        self._submitted_jobs[job_id] = {
            "metadata": task_metadata,
            "func_name": task_metadata._func_name,
            "total_tasks": len(iterable),
            "checkpoint_enabled": task_metadata.checkpoint_enabled,
            "future": future,
        }

        # Create and send message
        message = create_submit_job_message_with_metadata(
            func_code, iterable, job_id, task_metadata.to_dict()
        )
        await self.websocket.send(message.to_json())

        print(f"[SDK] Job {job_id[:8]}... submitted asynchronously")
        return job_id

    async def get_results(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get results for a submitted job

        Args:
            job_id: Job identifier from submit()
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            List of results

        Raises:
            TimeoutError: If timeout exceeded
            KeyError: If job_id not found
        """
        if job_id not in self.pending_jobs:
            raise KeyError(f"Job {job_id} not found")

        future = self.pending_jobs[job_id]

        if timeout:
            try:
                results = await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )
        else:
            results = await future

        # Clean up
        del self.pending_jobs[job_id]
        if job_id in self._submitted_jobs:
            del self._submitted_jobs[job_id]

        return results

    async def pipeline(
        self,
        stages: List[Dict[str, Any]],
        dependency_map: Optional[Dict[str, List[str]]] = None,
        dnn_config: Optional[Dict[str, Any]] = None,
        pipeline_mode: str = "barrier",
        **kwargs,
    ) -> List[Any]:
        """
        Execute a pipeline of dependent stages on distributed workers.

        Each stage defines a function and its input arguments.  Tasks in
        later stages are automatically blocked until their upstream
        dependencies complete.  The system tracks a *dependency counter*
        per task and decrements it as upstream tasks finish; when the
        counter reaches zero the task becomes eligible for scheduling.

        This is the pipeline equivalent of ``map()`` – it submits the
        whole pipeline as a single job and waits for the final results.

        Args:
            stages: Ordered list of stage definitions.  Each stage is a dict:
                {
                    "func": <callable>,        # optional for native model stages
                    "args_list": [arg1, ...],  # one entry per task
                    "pass_upstream_results": False,  # inject upstream results?
                    "name": "stage_name",      # optional human label
                }
            dependency_map: Optional explicit mapping of
                task_id → [upstream_task_ids] for arbitrary DAGs.
                If omitted, sequential barriers between stages are used.
            **kwargs: Additional options (checkpoint, checkpoint_interval, etc.)

        Returns:
            List of results from the final stage.

        Example (sequential pipeline)::

            results = await client.pipeline([
                {"func": preprocess,  "args_list": raw_data},
                {"func": compute,     "args_list": [None]*len(raw_data),
                 "pass_upstream_results": True},
                {"func": aggregate,   "args_list": [None]*1,
                 "pass_upstream_results": True},
            ])
        """
        if not self.connected:
            raise RuntimeError("Not connected to foreman. Call connect() first.")

        job_id = str(uuid.uuid4())

        # Create future
        future = asyncio.Future()
        self.pending_jobs[job_id] = future

        try:
            # Build stage definitions for the protocol message
            stage_defs = []
            for stage_idx, stage_def in enumerate(stages):
                func = stage_def.get("func")
                args_list = stage_def["args_list"]
                stage_name = stage_def.get("name")
                if not stage_name:
                    if func is not None:
                        stage_name = func.__name__
                    else:
                        stage_name = f"stage_{stage_idx}"

                execution_mode = stage_def.get("execution_mode")
                model_partition_id = stage_def.get("model_partition_id") or stage_name

                # Serialise function (or internal passthrough for native model stages)
                if func is None:
                    if not execution_mode:
                        execution_mode = "native_model"
                    func_code = serialize_function(_native_model_stage_passthrough)
                    metadata = None
                else:
                    if hasattr(func, "__crowdio_original__"):
                        func_code = serialize_function(func.__crowdio_original__)
                    else:
                        func_code = serialize_function(func)
                    metadata = get_task_metadata(func)

                # Pre-instrument with task control (pause/kill) at SDK level
                stage_ckpt_vars = []
                if metadata:
                    stage_ckpt_vars = metadata.checkpoint_state or []
                func_code, _, _ = instrument_for_task_control(
                    func_code, checkpoint_state_vars=stage_ckpt_vars
                )
                ctrl_wrapper = generate_task_control_wrapper()
                func_code = ctrl_wrapper + "\n" + func_code
                task_meta_dict = None
                if metadata:
                    task_meta_dict = metadata.to_dict()

                stage_defs.append(
                    {
                        "func_code": func_code,
                        "args_list": args_list,
                        "pass_upstream_results": stage_def.get(
                            "pass_upstream_results", False
                        ),
                        "task_metadata": task_meta_dict,
                        "name": stage_name,
                        "execution_mode": execution_mode or "python",
                        "model_partition_id": model_partition_id,
                    }
                )

            # Global task metadata from kwargs
            global_meta = None
            if kwargs.get("checkpoint"):
                global_meta = {
                    "checkpoint_enabled": True,
                    "checkpoint_interval": kwargs.get("checkpoint_interval", 10.0),
                    "checkpoint_state": kwargs.get("checkpoint_state", []),
                }

            # Build and send protocol message
            message = create_submit_pipeline_job_message(
                job_id=job_id,
                stages=stage_defs,
                dependency_map=dependency_map,
                task_metadata=global_meta,
                dnn_config=dnn_config,
                pipeline_mode=pipeline_mode,
            )

            self._submitted_jobs[job_id] = {
                "type": "pipeline",
                "stages": len(stage_defs),
                "total_tasks": sum(len(s["args_list"]) for s in stage_defs),
            }

            await self.websocket.send(message.to_json())

            total_tasks = sum(len(s["args_list"]) for s in stage_defs)
            print(
                f"[SDK] Pipeline job {job_id[:8]}... submitted "
                f"({len(stage_defs)} stages, {total_tasks} tasks)"
            )

            # Wait for results (same flow as map)
            results = await future
            return results

        except Exception as e:
            if job_id in self.pending_jobs:
                del self.pending_jobs[job_id]
            if job_id in self._submitted_jobs:
                del self._submitted_jobs[job_id]
            raise

    async def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run a single function with arguments in one worker."""
        # Reuse map() flow for consistency with metadata/checkpoint handling.
        task_payload: Any = args
        call_kwargs = dict(kwargs)
        if call_kwargs:
            task_payload = [list(args), call_kwargs]

        results = await self.map(func, [task_payload])
        if not results:
            return None
        return results[0]

    async def dnn_pipeline(
        self,
        stages: List[Dict[str, Any]],
        inference_graph_id: Optional[str] = None,
        topology_nodes: Optional[List[Dict[str, Any]]] = None,
        topology_edges: Optional[List[Dict[str, Any]]] = None,
        model_paths: Optional[List[str]] = None,
        model_partition_ids: Optional[List[str]] = None,
        model_version_id: Optional[str] = None,
        model_artifacts: Optional[List[Dict[str, Any]]] = None,
        aggregation_strategy: str = "average",
        dependency_map: Optional[Dict[str, List[str]]] = None,
        pipeline_mode: str = "streaming",
        **kwargs,
    ) -> List[Any]:
        """
        Submit a topology-aware DNN pipeline job.

        You can provide explicit topology metadata (topology_nodes/topology_edges)
        or model paths (either model_paths=[...] or per-stage "model" fields),
        in which case the SDK builds a linear topology from stage order
        automatically.
        """
        if not stages:
            raise ValueError("stages must not be empty")

        stage_model_paths: List[Optional[str]] = []
        stage_model_count = 0
        for stage in stages:
            model_path = stage.get("model")
            if model_path:
                stage_model_paths.append(str(model_path))
                stage_model_count += 1
            else:
                stage_model_paths.append(None)

        if stage_model_count and stage_model_count != len(stages):
            raise ValueError(
                "When using per-stage model fields, every stage must include a model path"
            )

        resolved_model_paths = list(model_paths or [])
        if not resolved_model_paths and stage_model_count:
            resolved_model_paths = [p for p in stage_model_paths if p is not None]

        if resolved_model_paths and stage_model_count:
            explicit_stage_models = [p for p in stage_model_paths if p is not None]
            if len(explicit_stage_models) != len(resolved_model_paths):
                raise ValueError(
                    "model_paths length must match the number of stage model entries"
                )
            for idx, (from_stage, from_arg) in enumerate(
                zip(explicit_stage_models, resolved_model_paths)
            ):
                if from_stage != from_arg:
                    raise ValueError(
                        f"Conflicting model path at stage index {idx}: stage model '{from_stage}' != model_paths '{from_arg}'"
                    )

        normalized_stages: List[Dict[str, Any]] = []
        used_stage_names = set()
        for idx, stage in enumerate(stages):
            stage_copy = dict(stage)
            stage_name = stage_copy.get("name")
            if not stage_name:
                if idx < len(stage_model_paths) and stage_model_paths[idx]:
                    stage_name = Path(stage_model_paths[idx]).stem
                elif idx < len(resolved_model_paths):
                    stage_name = Path(resolved_model_paths[idx]).stem
                else:
                    func = stage_copy.get("func")
                    stage_name = getattr(func, "__name__", f"stage_{idx}")

            base_name = str(stage_name)
            unique_name = base_name
            suffix = 1
            while unique_name in used_stage_names:
                unique_name = f"{base_name}_{suffix}"
                suffix += 1

            stage_copy["name"] = unique_name
            stage_has_func = stage_copy.get("func") is not None
            if not stage_has_func and idx < len(resolved_model_paths):
                stage_copy["execution_mode"] = stage_copy.get(
                    "execution_mode", "native_model"
                )
                stage_copy["model_partition_id"] = stage_copy.get(
                    "model_partition_id", unique_name
                )
            stage_copy.pop("model", None)
            used_stage_names.add(unique_name)
            normalized_stages.append(stage_copy)

        if resolved_model_paths and len(resolved_model_paths) != len(normalized_stages):
            raise ValueError("model_paths length must match stages length")

        resolved_partition_ids = list(model_partition_ids or [])
        if resolved_partition_ids and not resolved_model_paths:
            raise ValueError(
                "model_partition_ids requires model paths (from model_paths or per-stage model)"
            )

        if resolved_partition_ids and len(resolved_partition_ids) != len(
            normalized_stages
        ):
            raise ValueError("model_partition_ids length must match stages length")

        if resolved_model_paths and not resolved_partition_ids:
            # Keep partition IDs aligned with stage names for topology validation.
            resolved_partition_ids = [stage["name"] for stage in normalized_stages]

        resolved_topology_nodes = topology_nodes
        resolved_topology_edges = topology_edges

        if resolved_topology_nodes is None or resolved_topology_edges is None:
            if not resolved_model_paths:
                raise ValueError(
                    "Provide topology_nodes/topology_edges or model_paths for auto-topology"
                )

            resolved_topology_nodes = []
            for idx, stage in enumerate(normalized_stages):
                if idx == 0:
                    role = "source"
                elif idx == len(normalized_stages) - 1:
                    role = "sink"
                else:
                    role = "intermediate"

                requirements = stage.get("requirements") or {}
                if not isinstance(requirements, dict):
                    requirements = {}

                resolved_topology_nodes.append(
                    {
                        "node_id": stage["name"],
                        "role": role,
                        "model_partition_id": resolved_partition_ids[idx],
                        "requirements": requirements,
                    }
                )

            resolved_topology_edges = [
                {
                    "source": normalized_stages[idx]["name"],
                    "target": normalized_stages[idx + 1]["name"],
                    "metadata": {"kind": "pipeline"},
                }
                for idx in range(len(normalized_stages) - 1)
            ]
        else:
            resolved_topology_nodes = list(resolved_topology_nodes)
            resolved_topology_edges = list(resolved_topology_edges)

        resolved_model_artifacts = model_artifacts
        if resolved_model_artifacts is None and resolved_model_paths:
            resolved_model_artifacts = []
            for idx, model_path in enumerate(resolved_model_paths):
                resolved_model_artifacts.append(
                    build_partition_artifact(
                        model_partition_id=resolved_partition_ids[idx],
                        file_path=model_path,
                        assigned_device_id=None,
                    )
                )
        elif resolved_model_artifacts is None:
            resolved_model_artifacts = []

        resolved_inference_graph_id = (
            inference_graph_id or f"dnn-graph-{uuid.uuid4().hex[:12]}"
        )

        resolved_model_version_id = model_version_id
        if resolved_model_artifacts and not resolved_model_version_id:
            resolved_model_version_id = f"dnn-model-{uuid.uuid4().hex[:12]}"

        validate_topology(
            stages=normalized_stages,
            topology_nodes=resolved_topology_nodes,
            topology_edges=resolved_topology_edges,
        )

        dnn_config = {
            "inference_graph_id": resolved_inference_graph_id,
            "topology_nodes": resolved_topology_nodes,
            "topology_edges": resolved_topology_edges,
            "model_version_id": resolved_model_version_id,
            "model_artifacts": resolved_model_artifacts,
            "aggregation_strategy": aggregation_strategy,
        }
        return await self.pipeline(
            stages=normalized_stages,
            dependency_map=dependency_map,
            dnn_config=dnn_config,
            pipeline_mode=pipeline_mode,
            **kwargs,
        )

    async def send_intermediate_feature(
        self,
        job_id: str,
        task_id: str,
        target_task_id: str,
        payload: Any,
        source_worker_id: str = "sdk-client",
    ) -> None:
        """Send intermediate feature payload to foreman with tensor-aware encoding."""
        if not self.connected:
            raise RuntimeError("Not connected to foreman. Call connect() first.")

        encoded_payload = encode_feature_payload(payload)
        message = create_intermediate_feature_message(
            job_id=job_id,
            task_id=task_id,
            source_worker_id=source_worker_id,
            target_task_id=target_task_id,
            payload=encoded_payload,
            payload_format="json",
        )
        await self.websocket.send(message.to_json())

    def decode_intermediate_feature_payload(self, payload: Any) -> Any:
        """Public helper for decoding tensor-aware intermediate feature payloads."""
        return decode_feature_payload(payload)
