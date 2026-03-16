"""
FastAPI Worker for CrowdCompute.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import websockets
from fastapi import WebSocketDisconnect

from common.protocol import Message, MessageType
from common.serializer import deserialize_function_for_PC, get_runtime_info
from ..config import WorkerConfig
from ..schema.models import TaskResult
from .app import create_app
from ..staged_results_manager.checkpoint_handler import CheckpointHandler


class FastAPIWorker:
    """FastAPI-based worker for CrowdCompute."""

    websocket_update_interval: int = 5

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_queue = asyncio.Queue()
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "started_at": datetime.now(),
            "task_durations": [],  # Track individual task durations for MCDM
        }
        
        # Initialize checkpoint handler with 5 second intervals
        self.checkpoint_handler = CheckpointHandler(checkpoint_interval=5.0)

        # Build FastAPI application with routes and dashboard
        self.app = create_app(self)

    # ---------- Public serialization helpers ----------
    def get_success_rate(self) -> float:
        """Calculate task success rate (0-1)"""
        total = self.stats["tasks_completed"] + self.stats["tasks_failed"]
        if total == 0:
            return 1.0  # Default to 100% if no tasks yet
        return self.stats["tasks_completed"] / total

    def get_average_task_duration(self) -> float:
        """Calculate average task duration in seconds"""
        if not self.stats["task_durations"]:
            return 0.0
        return sum(self.stats["task_durations"]) / len(self.stats["task_durations"])

    def _stats_for_json(self) -> Dict[str, Any]:
        return {
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "total_execution_time": self.stats["total_execution_time"],
            "started_at": self.stats["started_at"].isoformat(),
            "success_rate": self.get_success_rate(),
            "avg_task_duration_sec": self.get_average_task_duration(),
        }

    def serialize_status(self) -> Dict[str, Any]:
        return {
            "worker_id": self.config.worker_id,
            "status": "online" if self.is_connected else "offline",
            "current_task": self.current_task["task_id"] if self.current_task else None,
            "stats": self._stats_for_json(),
            "config": self.config.dict(),
        }

    def serialize_stats(self) -> Dict[str, Any]:
        return {
            "worker_id": self.config.worker_id,
            "is_connected": self.is_connected,
            "current_task": self.current_task,
            "stats": self._stats_for_json(),
            "uptime": (datetime.now() - self.stats["started_at"]).total_seconds(),
        }

    def serialize_ws_status(self) -> Dict[str, Any]:
        status = self.serialize_stats()
        status["timestamp"] = datetime.now().isoformat()
        return status

    # ---------- Logging helper ----------
    def log(self, message: str) -> None:
        print(message)

    # ---------- Connection management ----------
    async def connect(self) -> bool:
        """Connect to the foreman WebSocket server."""
        try:
            print(f"🔌 Connecting to foreman at {self.config.foreman_url}/worker/ws...")

            # Configure WebSocket connection with proper timeouts
            # ping_interval: Disable built-in ping, we handle heartbeats ourselves
            # ping_timeout: Disable ping timeout
            # close_timeout: Wait up to 30s for close handshake
            # max_size: None disables message size cap so workers can receive
            # large task assignments (e.g., TFLite model blobs).
            self.websocket = await websockets.connect(
                f"{self.config.foreman_url}/worker/ws",
                ping_interval=None,  # Disable built-in ping, we handle heartbeats ourselves
                ping_timeout=None,  # Disable ping timeout
                close_timeout=30,  # Wait up to 30s for close handshake
                max_size=None,
            )
            self.is_connected = True

            print(f"✅ Connected to foreman as {self.config.worker_id}")

            # Collect device specifications
            print("📊 Collecting device specifications...")
            try:
                from common.device_info import (
                    get_device_specs,
                    format_device_specs_summary,
                )

                device_specs = get_device_specs()
                print(f"\n{format_device_specs_summary(device_specs)}\n")
            except Exception as e:
                print(f"⚠️ Could not collect full device specs: {e}")
                try:
                    from common.device_info import get_lightweight_device_specs

                    device_specs = get_lightweight_device_specs()
                except:
                    device_specs = {}
                    print("⚠️ Using minimal device info")

            # Send initial ready message with device specs
            ready_message = Message(
                msg_type=MessageType.WORKER_READY,
                data={"worker_id": self.config.worker_id, "device_specs": device_specs},
            )
            await self.websocket.send(ready_message.to_json())

            return True
        except Exception as e:
            print(f"❌ Failed to connect to foreman: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from the foreman"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_connected = False
        print("🔌 Disconnected from foreman")

    async def restart(self) -> None:
        """Restart the worker connection to the foreman."""
        await self.disconnect()
        await self.connect()

    # ---------- Message handling ----------
    async def handle_message(self, message: Message):
        """Handle incoming messages from foreman"""
        try:
            if message.type == MessageType.ASSIGN_TASK:
                await self._handle_task_assignment(message)
            elif message.type == MessageType.RESUME_TASK:
                await self._handle_task_resume(message)
            elif message.type == MessageType.PING:
                # Respond to ping with performance metrics
                await self._handle_ping()
            elif message.type == MessageType.CHECKPOINT_ACK:
                # Foreman acknowledged checkpoint receipt - no action needed
                checkpoint_id = message.data.get("checkpoint_id", "?")
                task_id = message.data.get("task_id", "?")
                print(f"Checkpoint #{checkpoint_id} acknowledged for task {task_id}")
            else:
                print(f"Unknown message type: {message.type}")

        except Exception as e:
            print(f"❌ Error handling message: {e}")

    async def _handle_ping(self):
        """Handle PING message and respond with PONG + performance metrics"""
        try:
            # Collect current performance metrics
            from common.device_info import get_performance_metrics

            metrics = get_performance_metrics()

            # Send PONG with metrics
            pong_message = Message(
                msg_type=MessageType.PONG,
                data={
                    "worker_id": self.config.worker_id,
                    "performance_metrics": metrics,
                },
            )
            await self.websocket.send(pong_message.to_json())
        except Exception as e:
            # Fallback to simple PONG if metrics collection fails
            print(f"⚠️ Could not collect performance metrics: {e}")
            pong_message = Message(
                msg_type=MessageType.PONG, data={"worker_id": self.config.worker_id}
            )
            await self.websocket.send(pong_message.to_json())

    async def _handle_task_assignment(self, message: Message):
        """Handle a task assignment from foreman with checkpoint metadata support"""
        try:
            task_id = message.data["task_id"]
            job_id = message.job_id
            func_code = message.data["func_code"]
            task_args = message.data["task_args"]
            
            # Extract task metadata for declarative checkpointing
            task_metadata = message.data.get("task_metadata", {})
            checkpoint_enabled = task_metadata.get("checkpoint_enabled", True)
            checkpoint_interval = task_metadata.get("checkpoint_interval", 5.0)
            checkpoint_state_vars = task_metadata.get("checkpoint_state", [])
            
            print(
                f"📋 Received task {task_id} for job {job_id} | worker_runtime={get_runtime_info()}"
            )
            
            if task_metadata:
                print(f"   Checkpoint: {'enabled' if checkpoint_enabled else 'disabled'} | "
                      f"Interval: {checkpoint_interval}s | "
                      f"State vars: {checkpoint_state_vars or 'all'}")

            # Set current task
            self.current_task = {
                "task_id": task_id, 
                "job_id": job_id,
                "task_metadata": task_metadata
            }

            # Execute the task with metadata
            result = await self._execute_task(func_code, task_args, task_id, job_id, task_metadata)

            # Serialize result to JSON string for database storage
            import json
            try:
                result_str = json.dumps(result)
            except (TypeError, ValueError):
                # If result can't be JSON serialized, convert to string
                result_str = str(result)

            # Send result back
            result_message = Message(
                msg_type=MessageType.TASK_RESULT,
                data={"result": result_str, "task_id": task_id},
                job_id=job_id,
            )
            await self.websocket.send(result_message.to_json())
            
            # Small delay to ensure message is transmitted
            await asyncio.sleep(0.1)

            print(f"✅ Completed task {task_id}")

            # Clear current task
            self.current_task = None

        except Exception as e:
            print(f"❌ Error executing task {task_id}: {e}")

            # Send error back
            error_message = Message(
                msg_type=MessageType.TASK_ERROR,
                data={"error": str(e), "task_id": task_id},
                job_id=job_id,
            )
            await self.websocket.send(error_message.to_json())

            # Clear current task
            self.current_task = None

    async def _handle_task_resume(self, message: Message):
        """
        Handle a task resumption from foreman.
        
        This is similar to task assignment but restores the checkpoint state
        so the worker can continue from where the previous worker left off.
        """
        try:
            task_id = message.data["task_id"]
            job_id = message.job_id
            func_code = message.data["func_code"]
            checkpoint_state = message.data.get("checkpoint_state", {})
            progress_percent = message.data.get("progress_percent", 0)
            checkpoint_count = message.data.get("checkpoint_count", 0)
            
            # Extract task metadata for declarative checkpointing
            task_metadata = message.data.get("task_metadata", {})
            recovery_status = message.data.get("recovery_status", "resumed")

            print(
                f"Resuming task {task_id} for job {job_id} from checkpoint "
                f"(progress: {progress_percent:.1f}%, checkpoints: {checkpoint_count}, "
                f"status: {recovery_status}) | worker_runtime={get_runtime_info()}"
            )
            
            # Log checkpoint state contents
            if checkpoint_state:
                print(f"Checkpoint state keys: {list(checkpoint_state.keys())}")
                for key, value in checkpoint_state.items():
                    if key.startswith("_"):
                        continue  # Skip internal keys
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
                    elif isinstance(value, str) and len(value) < 50:
                        print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: <{type(value).__name__}>")

            # Set current task
            self.current_task = {
                "task_id": task_id, 
                "job_id": job_id, 
                "is_resume": True,
                "task_metadata": task_metadata
            }

            # Execute the task with checkpoint state pre-loaded
            result = await self._execute_resumed_task(
                func_code, checkpoint_state, task_id, job_id, task_metadata
            )

            # Serialize result to JSON string for database storage
            import json
            try:
                result_str = json.dumps(result)
            except (TypeError, ValueError):
                # If result can't be JSON serialized, convert to string
                result_str = str(result)

            # Send result back
            result_message = Message(
                msg_type=MessageType.TASK_RESULT,
                data={"result": result_str, "task_id": task_id},
                job_id=job_id,
            )
            await self.websocket.send(result_message.to_json())
            
            # Small delay to ensure message is transmitted
            await asyncio.sleep(0.1)

            print(f"[Success] Completed resumed task {task_id}")

            # Clear current task
            self.current_task = None

        except Exception as e:
            print(f"[Error]Error executing resumed task {task_id}: {e}")
            import traceback
            traceback.print_exc()

            # Send error back
            error_message = Message(
                msg_type=MessageType.TASK_ERROR,
                data={"error": str(e), "task_id": task_id},
                job_id=job_id,
            )
            await self.websocket.send(error_message.to_json())

            # Clear current task
            self.current_task = None

    # ---------- Task execution ----------
    async def _execute_task(
        self, 
        func_code: str, 
        task_args: List[Any], 
        task_id: str, 
        job_id: str,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a task in a safe environment with declarative checkpointing support
        
        Args:
            func_code: Serialized function code
            task_args: Arguments for the task
            task_id: Task identifier
            job_id: Job identifier
            task_metadata: Checkpoint configuration from @task decorator
        """
        start_time = datetime.now()
        
        # Extract checkpoint configuration
        checkpoint_enabled = True
        checkpoint_interval = 5.0
        checkpoint_state_vars = []
        
        if task_metadata:
            checkpoint_enabled = task_metadata.get("checkpoint_enabled", True)
            checkpoint_interval = task_metadata.get("checkpoint_interval", 5.0)
            checkpoint_state_vars = task_metadata.get("checkpoint_state", [])
            
            # Update checkpoint handler interval
            self.checkpoint_handler.checkpoint_interval = checkpoint_interval
        
        # Shared state container for frame introspection
        # This will be updated by the execution wrapper
        _captured_locals = {"_locals": {}, "_frame": None}
        
        # Define state getter that uses frame introspection
        def get_task_state() -> Dict[str, Any]:
            """Get current task state by inspecting captured locals"""
            try:
                import sys
                import ctypes
                
                # Try to get locals from the captured frame
                frame = _captured_locals.get("_frame")
                if frame is not None:
                    try:
                        # Get current local variables from the frame
                        locals_dict = frame.f_locals
                        
                        # Filter to only declared checkpoint variables
                        state = {}
                        for var in checkpoint_state_vars:
                            if var in locals_dict:
                                state[var] = locals_dict[var]
                        
                        # Always include progress_percent if available
                        if "progress_percent" in locals_dict:
                            state["progress_percent"] = locals_dict["progress_percent"]
                        
                        return state if state else {"progress_percent": 0.0}
                    except Exception as e:
                        pass
                
                # Fallback to captured locals snapshot
                locals_snapshot = _captured_locals.get("_locals", {})
                if locals_snapshot:
                    state = {}
                    for var in checkpoint_state_vars:
                        if var in locals_snapshot:
                            state[var] = locals_snapshot[var]
                    if "progress_percent" in locals_snapshot:
                        state["progress_percent"] = locals_snapshot["progress_percent"]
                    return state if state else {"progress_percent": 0.0}
                
                return {"progress_percent": 0.0}
            except Exception as e:
                print(f"[warning] Could not get checkpoint state: {e}")
                return {"progress_percent": 0.0}
        
        # Get reference to the main event loop for cross-thread communication
        main_loop = asyncio.get_event_loop()
        
        # Define checkpoint sender
        def send_checkpoint(checkpoint_msg: Dict[str, Any]) -> None:
            """Send checkpoint to foreman (thread-safe for cross-thread calls)"""
            try:
                from common.protocol import create_task_checkpoint_message
                
                message = create_task_checkpoint_message(
                    task_id=task_id,
                    job_id=job_id,
                    **checkpoint_msg
                )
                
                if self.websocket and not self.websocket.closed:
                    future = asyncio.run_coroutine_threadsafe(
                        self.websocket.send(message.to_json()),
                        main_loop
                    )
                    future.result(timeout=5.0)
                    
                    checkpoint_id = checkpoint_msg.get("checkpoint_id", 0)
                    progress = checkpoint_msg.get("progress_percent", 0)
                    is_base = checkpoint_msg.get("is_base", False)
                    checkpoint_type = "BASE" if is_base else "DELTA"
                    
                    print(f"[Checkpoint] Resumed task {task_id} | {checkpoint_type} #{checkpoint_id} | "
                          f"Progress: {progress:.1f}%")
            except Exception as e:
                print(f"[Error] Error sending checkpoint: {e}")

        try:
            print(f"Executing task... | worker_runtime={get_runtime_info()}")
            if task_metadata:
                vars_info = checkpoint_state_vars if checkpoint_state_vars else "all"
                print(f"   Checkpoint config: enabled={checkpoint_enabled}, interval={checkpoint_interval}s, vars={vars_info}")

            # Deserialize the function
            func = deserialize_function_for_PC(func_code)
            func_name = func.__name__
            
            # Start checkpoint monitoring in background (if enabled)
            if checkpoint_enabled:
                await self.checkpoint_handler.start_checkpoint_monitoring(
                    task_id=task_id,
                    get_state_callback=get_task_state,
                    send_checkpoint_callback=send_checkpoint,
                    task_metadata=task_metadata
                )
                print(f"Checkpoint monitoring started for task {task_id}")
            else:
                print(f"Checkpointing disabled for task {task_id}")

            # Create wrapper that uses trace function to capture locals
            import sys
            
            def execute_with_trace(func, args):
                """Execute function with trace to capture local variables"""
                
                def trace_calls(frame, event, arg):
                    """Trace function that captures locals from the task function"""
                    # Only trace the task function, not all calls
                    if frame.f_code.co_name == func_name:
                        if event == 'line' or event == 'return':
                            # Capture current locals
                            _captured_locals["_locals"] = dict(frame.f_locals)
                    return trace_calls
                
                # Set trace and execute
                old_trace = sys.gettrace()
                try:
                    sys.settrace(trace_calls)
                    
                    # Execute the actual function
                    if isinstance(args, list) and len(args) == 1:
                        return func(args[0])
                    elif isinstance(args, list) and len(args) == 2 and isinstance(args[1], dict):
                        a, kw = args
                        return func(*a, **kw)
                    else:
                        return func(*args)
                finally:
                    sys.settrace(old_trace)
            
            # Execute the function with trace-based capture
            result = await asyncio.to_thread(execute_with_trace, func, task_args)

            execution_time = (datetime.now() - start_time).total_seconds()

            print(f"Task completed in {execution_time:.2f}s")

            # Update stats
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += execution_time
            self.stats["task_durations"].append(execution_time)

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Resumed task execution failed: {e}"

            print(f"[Error] Task failed: {error_msg}")

            self.stats["tasks_failed"] += 1
            self.stats["total_execution_time"] += execution_time

            raise Exception(error_msg)
        
        finally:
            # Stop checkpoint monitoring
            if checkpoint_enabled:
                await self.checkpoint_handler.stop_checkpoint_monitoring()
                print(f"[Info] Checkpoint monitoring stopped for task {task_id}")
            
            # Clean up captured frame reference
            _captured_locals["_frame"] = None
            _captured_locals["_locals"] = {}

    def _transform_code_for_resume(
        self,
        func_code: str,
        checkpoint_state: Dict[str, Any],
        checkpoint_state_vars: List[str]
    ) -> str:
        """
        Transform function code to resume from checkpoint by modifying AST.
        
        This makes checkpoint resume COMPLETELY TRANSPARENT to the user.
        The user writes pure logic, and the framework handles resume automatically.
        
        Transformations applied:
        1. Replace initial variable assignments with checkpoint values
        2. Adjust for-loop ranges to start from checkpointed position
        
        Args:
            func_code: Original function source code
            checkpoint_state: Dictionary with saved checkpoint state
            checkpoint_state_vars: List of variable names being checkpointed
            
        Returns:
            Transformed function code that will resume from checkpoint
        """
        import ast
        import re
        
        try:
            # Parse the function code
            tree = ast.parse(func_code)
            
            # Find the function definition
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break
            
            if not func_def:
                print(f"[Resume] Could not find function definition, using original code")
                return func_code
            
            # Track which variables we've modified and their checkpoint values
            vars_to_inject = {}
            for var in checkpoint_state_vars:
                if var in checkpoint_state:
                    vars_to_inject[var] = checkpoint_state[var]
            
            # Also check for progress_percent
            if "progress_percent" in checkpoint_state:
                vars_to_inject["progress_percent"] = checkpoint_state["progress_percent"]
            
            print(f"[Resume] Injecting checkpoint values: {list(vars_to_inject.keys())}")
            
            # Determine the loop index variable and its start value
            # We use progress_percent to calculate where to resume
            progress = checkpoint_state.get("progress_percent", 0)
            
            # Create a code transformer
            class ResumeTransformer(ast.NodeTransformer):
                def __init__(self):
                    self.in_function = False
                    self.modified_vars = set()
                    self.found_main_loop = False
                    
                def visit_FunctionDef(self, node):
                    if node.name == func_def.name:
                        self.in_function = True
                        self.generic_visit(node)
                        self.in_function = False
                    return node
                
                def visit_Assign(self, node):
                    """Replace initial variable assignments with checkpoint values"""
                    if not self.in_function:
                        return node
                    
                    # Check if this is a simple assignment to a checkpointed variable
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                        if var_name in vars_to_inject and var_name not in self.modified_vars:
                            # Replace with checkpoint value
                            value = vars_to_inject[var_name]
                            new_value = self._create_ast_value(value)
                            if new_value:
                                node.value = new_value
                                self.modified_vars.add(var_name)
                                print(f"[Resume] Injected {var_name} = {repr(value)[:50]}")
                    
                    return node
                
                def visit_For(self, node):
                    """Adjust for-loop range to start from checkpoint position"""
                    if not self.in_function or self.found_main_loop:
                        return node
                    
                    # Check if this is a for loop with range()
                    if isinstance(node.iter, ast.Call):
                        func = node.iter.func
                        if isinstance(func, ast.Name) and func.id == 'range':
                            # Calculate start position from progress
                            # We need to find the total (end of range)
                            args = node.iter.args
                            
                            if len(args) >= 1:
                                # Get the loop variable name
                                if isinstance(node.target, ast.Name):
                                    loop_var = node.target.id
                                    
                                    # Calculate start position based on progress
                                    # progress_percent = (completed / total) * 100
                                    # completed = progress_percent * total / 100
                                    
                                    # We'll inject a calculation at the start
                                    self.found_main_loop = True
                                    
                                    # Modify the range to include a start value
                                    # range(n) -> range(_resume_start, n)
                                    # range(start, end) -> range(max(start, _resume_start), end)
                                    
                                    if len(args) == 1:
                                        # range(n) -> range(_resume_start_, n)
                                        end_arg = args[0]
                                        # Create: int(progress_percent * n / 100)
                                        start_expr = ast.Call(
                                            func=ast.Name(id='int', ctx=ast.Load()),
                                            args=[ast.BinOp(
                                                left=ast.BinOp(
                                                    left=ast.Constant(value=progress),
                                                    op=ast.Mult(),
                                                    right=end_arg
                                                ),
                                                op=ast.Div(),
                                                right=ast.Constant(value=100)
                                            )],
                                            keywords=[]
                                        )
                                        node.iter.args = [start_expr, end_arg]
                                        print(f"[Resume] Modified range() to start from {progress:.1f}%")
                                    
                                    elif len(args) == 2:
                                        # range(start, end) -> range(max(start, calculated), end)
                                        orig_start, end_arg = args
                                        start_expr = ast.Call(
                                            func=ast.Name(id='max', ctx=ast.Load()),
                                            args=[
                                                orig_start,
                                                ast.Call(
                                                    func=ast.Name(id='int', ctx=ast.Load()),
                                                    args=[ast.BinOp(
                                                        left=ast.BinOp(
                                                            left=ast.Constant(value=progress),
                                                            op=ast.Mult(),
                                                            right=end_arg
                                                        ),
                                                        op=ast.Div(),
                                                        right=ast.Constant(value=100)
                                                    )],
                                                    keywords=[]
                                                )
                                            ],
                                            keywords=[]
                                        )
                                        node.iter.args = [start_expr, end_arg]
                                        print(f"[Resume] Modified range(start, end) to resume from {progress:.1f}%")
                    
                    self.generic_visit(node)
                    return node
                
                def _create_ast_value(self, value):
                    """Create an AST node for a Python value"""
                    if isinstance(value, (int, float, str, bool, type(None))):
                        return ast.Constant(value=value)
                    elif isinstance(value, list):
                        return ast.List(
                            elts=[self._create_ast_value(v) for v in value],
                            ctx=ast.Load()
                        )
                    elif isinstance(value, dict):
                        return ast.Dict(
                            keys=[ast.Constant(value=k) for k in value.keys()],
                            values=[self._create_ast_value(v) for v in value.values()]
                        )
                    elif isinstance(value, tuple):
                        return ast.Tuple(
                            elts=[self._create_ast_value(v) for v in value],
                            ctx=ast.Load()
                        )
                    else:
                        # For complex objects, we can't easily inject them
                        # Return None to skip injection
                        return None
            
            # Apply transformations
            transformer = ResumeTransformer()
            transformed_tree = transformer.visit(tree)
            
            # Fix missing line numbers
            ast.fix_missing_locations(transformed_tree)
            
            # Convert back to code using ast.unparse (Python 3.9+)
            try:
                transformed_code = ast.unparse(transformed_tree)
            except AttributeError:
                print(f"[Resume] Could not convert AST back to code (requires Python 3.9+), using original")
                return func_code
            
            print(f"[Resume] Code transformation successful")
            return transformed_code
            
        except Exception as e:
            print(f"[Resume] Code transformation failed: {e}, using original code")
            import traceback
            traceback.print_exc()
            return func_code

    async def _execute_resumed_task(
        self, 
        func_code: str, 
        checkpoint_state: dict, 
        task_id: str, 
        job_id: str,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a resumed task, continuing from checkpoint state.
        
        Uses trace-based capture to automatically capture local variables,
        just like _execute_task, but pre-loads the checkpoint state.
        
        Args:
            func_code: Serialized function code
            checkpoint_state: Dictionary with saved state from last checkpoint
            task_id: Task identifier
            job_id: Job identifier
            task_metadata: Checkpoint configuration from @task decorator
            
        Returns:
            Task result
        """
        start_time = datetime.now()
        
        # Extract checkpoint configuration
        checkpoint_enabled = True
        checkpoint_interval = 5.0
        checkpoint_state_vars = []
        
        if task_metadata:
            checkpoint_enabled = task_metadata.get("checkpoint_enabled", True)
            checkpoint_interval = task_metadata.get("checkpoint_interval", 5.0)
            checkpoint_state_vars = task_metadata.get("checkpoint_state", [])
            self.checkpoint_handler.checkpoint_interval = checkpoint_interval
        
        # Shared state container for frame introspection (same as _execute_task)
        _captured_locals = {"_locals": {}, "_frame": None}
        
        # Define state getter that uses frame introspection (same as _execute_task)
        def get_task_state() -> Dict[str, Any]:
            """Get current task state by inspecting captured locals"""
            try:
                import sys
                import ctypes
                
                # Try to get locals from the captured frame
                frame = _captured_locals.get("_frame")
                if frame is not None:
                    try:
                        # Get current local variables from the frame
                        locals_dict = frame.f_locals
                        
                        # Filter to only declared checkpoint variables
                        state = {}
                        for var in checkpoint_state_vars:
                            if var in locals_dict:
                                state[var] = locals_dict[var]
                        
                        # Always include progress_percent if available
                        if "progress_percent" in locals_dict:
                            state["progress_percent"] = locals_dict["progress_percent"]
                        
                        return state if state else {"progress_percent": 0.0}
                    except Exception as e:
                        pass
                
                # Fallback to captured locals snapshot
                locals_snapshot = _captured_locals.get("_locals", {})
                if locals_snapshot:
                    state = {}
                    for var in checkpoint_state_vars:
                        if var in locals_snapshot:
                            state[var] = locals_snapshot[var]
                    if "progress_percent" in locals_snapshot:
                        state["progress_percent"] = locals_snapshot["progress_percent"]
                    return state if state else {"progress_percent": 0.0}
                
                return {"progress_percent": 0.0}
            except Exception as e:
                print(f"⚠️ Could not get checkpoint state: {e}")
                return {"progress_percent": 0.0}
        
        # Get reference to the main event loop for cross-thread communication
        main_loop = asyncio.get_event_loop()
        
        # Define checkpoint sender
        def send_checkpoint(checkpoint_msg: Dict[str, Any]) -> None:
            """Send checkpoint to foreman (thread-safe for cross-thread calls)"""
            try:
                from common.protocol import create_task_checkpoint_message
                
                message = create_task_checkpoint_message(
                    task_id=task_id,
                    job_id=job_id,
                    **checkpoint_msg
                )
                
                if self.websocket and not self.websocket.closed:
                    future = asyncio.run_coroutine_threadsafe(
                        self.websocket.send(message.to_json()),
                        main_loop
                    )
                    future.result(timeout=5.0)
                    
                    checkpoint_id = checkpoint_msg.get("checkpoint_id", 0)
                    progress = checkpoint_msg.get("progress_percent", 0)
                    is_base = checkpoint_msg.get("is_base", False)
                    checkpoint_type = "BASE" if is_base else "DELTA"
                    
                    print(f"[Checkpoint] Resumed task {task_id} | {checkpoint_type} #{checkpoint_id} | "
                          f"Progress: {progress:.1f}%")
            except Exception as e:
                print(f"[Error] Error sending checkpoint: {e}")

        try:
            print(f"Executing resumed task... | worker_runtime={get_runtime_info()}")
            if task_metadata:
                vars_info = checkpoint_state_vars if checkpoint_state_vars else "all"
                print(f"   Checkpoint config: enabled={checkpoint_enabled}, interval={checkpoint_interval}s, vars={vars_info}")

            # ================================================================
            # TRANSPARENT RESUME: Transform the code to inject checkpoint state
            # The user's code is automatically modified to resume from checkpoint
            # No manual resume logic needed in user code!
            # ================================================================
            
            # Transform the function code to inject checkpoint values
            transformed_code = self._transform_code_for_resume(
                func_code, checkpoint_state, checkpoint_state_vars
            )
            
            # Deserialize the transformed function
            func = deserialize_function_for_PC(transformed_code)
            func_name = func.__name__
            
            # Start checkpoint monitoring in background (if enabled)
            if checkpoint_enabled:
                await self.checkpoint_handler.start_checkpoint_monitoring(
                    task_id=task_id,
                    get_state_callback=get_task_state,
                    send_checkpoint_callback=send_checkpoint,
                    task_metadata=task_metadata
                )
                print(f"Checkpoint monitoring started for resumed task {task_id}")
            else:
                print(f"Checkpointing disabled for resumed task {task_id}")

            # Get original task args from checkpoint or message
            original_task_args = checkpoint_state.get("_original_task_args")
            
            # For backwards compatibility, try to get num_trials or other common arg names
            if original_task_args is None:
                # Try to find the original argument from checkpoint state
                num_trials = checkpoint_state.get("num_trials")
                if num_trials is None:
                    # Calculate from progress_percent if we have total info
                    # Default fallback
                    num_trials = 125000
                original_task_args = [num_trials]
            
            progress = checkpoint_state.get('progress_percent', 0)
            print(f"[Resume] Resuming from {progress:.1f}% progress")
            print(f"[Resume] Original args: {original_task_args}")

            # Create wrapper that uses trace function to capture locals (same as _execute_task)
            import sys
            
            def execute_with_trace(func, args):
                """Execute function with trace to capture local variables"""
                
                def trace_calls(frame, event, arg):
                    """Trace function that captures locals from the task function"""
                    # Only trace the task function, not all calls
                    if frame.f_code.co_name == func_name:
                        if event == 'line' or event == 'return':
                            # Capture current locals
                            _captured_locals["_locals"] = dict(frame.f_locals)
                    return trace_calls
                
                # Set trace and execute
                old_trace = sys.gettrace()
                try:
                    sys.settrace(trace_calls)
                    
                    # Execute the actual function
                    if isinstance(args, list) and len(args) == 1:
                        return func(args[0])
                    elif isinstance(args, list) and len(args) == 2 and isinstance(args[1], dict):
                        a, kw = args
                        return func(*a, **kw)
                    else:
                        return func(*args)
                finally:
                    sys.settrace(old_trace)

            # Execute the TRANSFORMED function - resume is now transparent!
            result = await asyncio.to_thread(execute_with_trace, func, original_task_args)

            execution_time = (datetime.now() - start_time).total_seconds()

            print(f"Resumed task completed in {execution_time:.2f}s")

            # Update stats
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += execution_time
            self.stats["task_durations"].append(execution_time)

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Resumed task execution failed: {e}"

            print(f"[Error] Resumed task failed: {error_msg}")
            import traceback
            traceback.print_exc()

            # Update stats
            self.stats["tasks_failed"] += 1
            self.stats["total_execution_time"] += execution_time

            raise Exception(error_msg)
        
        finally:
            # Stop checkpoint monitoring
            if checkpoint_enabled:
                await self.checkpoint_handler.stop_checkpoint_monitoring()
                print(f"Checkpoint monitoring stopped for resumed task {task_id}")
            
            # Clean up captured frame reference
            _captured_locals["_frame"] = None
            _captured_locals["_locals"] = {}

    # ---------- Background tasks ----------
    async def listen_for_tasks(self):
        """Listen for tasks from foreman"""
        while self.is_connected:
            try:
                if not self.websocket:
                    break

                # Receive message
                message_data = await self.websocket.recv()
                message = Message.from_json(message_data)

                # Handle message
                await self.handle_message(message)

            except websockets.exceptions.ConnectionClosed:
                print("🔌 Connection to foreman closed")
                self.is_connected = False
                break
            except Exception as e:
                print(f"❌ Error in task listener: {e}")
                await asyncio.sleep(1)

    async def heartbeat(self):
        """Send periodic heartbeat to foreman (as PONG with metrics)"""
        while self.is_connected:
            try:
                if self.websocket and not self.websocket.closed:
                    # Collect performance metrics
                    from common.device_info import get_performance_metrics

                    metrics = get_performance_metrics()

                    # Send heartbeat as PONG with performance metrics
                    heartbeat_message = Message(
                        msg_type=MessageType.PONG,
                        data={
                            "worker_id": self.config.worker_id,
                            "status": "online",
                            "current_task": (
                                self.current_task["task_id"]
                                if self.current_task
                                else None
                            ),
                            **metrics,  # Include performance metrics
                        },
                    )
                    await self.websocket.send(heartbeat_message.to_json())

                await asyncio.sleep(self.config.heartbeat_interval)

            except websockets.exceptions.ConnectionClosed:
                print("WebSocket closed during heartbeat, reconnecting...")
                self.is_connected = False
                break
            except Exception as e: 
                print(f"Error sending heartbeat: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
                break

    # ---------- Main worker lifecycle ----------
    async def start(self):
        """Start the worker"""
        print(f"🚀 Starting FastAPI Worker: {self.config.worker_id}")

        # Connect to foreman
        if not await self.connect():
            if self.config.auto_restart:
                print("🔄 Auto-restart enabled, retrying connection...")
                await asyncio.sleep(5)
                await self.start()
            return

        # Start background tasks
        task_listener = asyncio.create_task(self.listen_for_tasks())
        heartbeat_task = asyncio.create_task(self.heartbeat())

        try:
            # Keep worker running
            await asyncio.gather(task_listener, heartbeat_task)
        except KeyboardInterrupt:
            print("\n🛑 Worker stopped by user")
        except Exception as e:
            print(f"❌ Worker error: {e}")
        finally:
            await self.disconnect()

    def run(self):
        """Run the worker with FastAPI server"""
        import uvicorn

        print(f"🚀 Starting FastAPI Worker Server: {self.config.worker_id}")
        print("=" * 60)
        print(f"👤 Worker ID:     {self.config.worker_id}")
        print(f"🔌 Foreman URL:   {self.config.foreman_url}")
        print(f"🌐 Web Interface: http://{self.config.api_host}:{self.config.api_port}")
        print(
            f"📊 API Docs:      http://{self.config.api_host}:{self.config.api_port}/docs"
        )
        print("=" * 60)

        # Run FastAPI server with worker in background
        try:
            uvicorn.run(
                self.app,
                host=self.config.api_host,
                port=self.config.api_port,
                log_level="info",
                loop="asyncio",
            )
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")

    async def run_with_worker(self):
        """Run the worker with FastAPI server (async version)"""
        import uvicorn

        print(f"🚀 Starting FastAPI Worker Server: {self.config.worker_id}")
        print("=" * 60)
        print(f"👤 Worker ID:     {self.config.worker_id}")
        print(f"🔌 Foreman URL:   {self.config.foreman_url}")
        print(f"🌐 Web Interface: http://{self.config.api_host}:{self.config.api_port}")
        print(
            f"📊 API Docs:      http://{self.config.api_host}:{self.config.api_port}/docs"
        )
        print("=" * 60)

        # Start worker in background
        worker_task = asyncio.create_task(self.start())

        # Create server config
        config = uvicorn.Config(
            self.app,
            host=self.config.api_host,
            port=self.config.api_port,
            log_level="info",
        )

        # Create and run server
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        finally:
            worker_task.cancel()
