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

            self.websocket = await websockets.connect(
                f"{self.config.foreman_url}/worker/ws"
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
        """Handle a task assignment from foreman"""
        try:
            task_id = message.data["task_id"]
            job_id = message.job_id
            func_code = message.data["func_code"]
            task_args = message.data["task_args"]

            print(
                f"📋 Received task {task_id} for job {job_id} | worker_runtime={get_runtime_info()}"
            )

            # Set current task
            self.current_task = {"task_id": task_id, "job_id": job_id}

            # Execute the task
            result = await self._execute_task(func_code, task_args, task_id, job_id)

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

            print(
                f"Resuming task {task_id} for job {job_id} from checkpoint "
                f"(progress: {progress_percent:.1f}%, checkpoints: {checkpoint_count}) | "
                f"worker_runtime={get_runtime_info()}"
            )
            
            # Log checkpoint state contents
            if checkpoint_state:
                print(f"Checkpoint state keys: {list(checkpoint_state.keys())}")
                for key, value in checkpoint_state.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
                    elif isinstance(value, str) and len(value) < 50:
                        print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: <{type(value).__name__}>")

            # Set current task
            self.current_task = {"task_id": task_id, "job_id": job_id, "is_resume": True}

            # Execute the task with checkpoint state pre-loaded
            result = await self._execute_resumed_task(
                func_code, checkpoint_state, task_id, job_id
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
    async def _execute_task(self, func_code: str, task_args: List[Any], task_id: str, job_id: str) -> Any:
        """Execute a task in a safe environment with checkpointing support"""
        start_time = datetime.now()
        
        # Create a shared checkpoint state that the function can access
        # This is stored as a module-level variable that we can access
        import builtins
        builtins._checkpoint_state = {
            "progress_percent": 0.0,
            "status": "initializing",
            "task_id": task_id
        }
        
        # Define state getter for checkpoint handler
        def get_task_state() -> Dict[str, Any]:
            """Get current task state for checkpointing"""
            try:
                # Check for checkpoint state in builtins (shared with deserialized function)
                if hasattr(builtins, '_checkpoint_state'):
                    state = builtins._checkpoint_state
                    return dict(state)  # Return a copy
                # Fallback: check sys.modules
                import sys
                for mod_name, mod in sys.modules.items():
                    if hasattr(mod, 'checkpoint_state') and mod.checkpoint_state is not None:
                        state = mod.checkpoint_state
                        if isinstance(state, dict) and 'progress_percent' in state:
                            return dict(state)
                # Default fallback
                return {
                    "progress_percent": 50.0,
                    "status": "running"
                }
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
                
                # Send using thread-safe method to schedule on main event loop
                if self.websocket and not self.websocket.closed:
                    future = asyncio.run_coroutine_threadsafe(
                        self.websocket.send(message.to_json()),
                        main_loop
                    )
                    # Wait for send to complete (with timeout)
                    future.result(timeout=5.0)
                    
                    # Terminal logging for checkpoint progress
                    checkpoint_id = checkpoint_msg.get("checkpoint_id", 0)
                    progress = checkpoint_msg.get("progress_percent", 0)
                    is_base = checkpoint_msg.get("is_base", False)
                    checkpoint_type = "BASE" if is_base else "DELTA"
                    
                    print(f"[Checkpoint] Task {task_id} | {checkpoint_type} #{checkpoint_id} | "
                          f"Progress: {progress:.1f}%")
            except Exception as e:
                print(f"⚠️ Error sending checkpoint: {e}")

        try:
            print(f"🔄 Executing task... | worker_runtime={get_runtime_info()}")

            # Deserialize the function
            func = deserialize_function_for_PC(func_code)
            
            # Start checkpoint monitoring in background
            await self.checkpoint_handler.start_checkpoint_monitoring(
                task_id=task_id,
                get_state_callback=get_task_state,
                send_checkpoint_callback=send_checkpoint
            )
            
            print(f"✅ Checkpoint monitoring started for task {task_id}")

            # Execute the function in a thread to avoid blocking the event loop
            # This allows the checkpoint loop to run concurrently
            if isinstance(task_args, list) and len(task_args) == 1:
                # Single argument
                result = await asyncio.to_thread(func, task_args[0])
            elif (
                isinstance(task_args, list)
                and len(task_args) == 2
                and isinstance(task_args[1], dict)
            ):
                # Function with args and kwargs
                args, kwargs = task_args
                result = await asyncio.to_thread(lambda: func(*args, **kwargs))
            else:
                # Multiple arguments
                result = await asyncio.to_thread(lambda: func(*task_args))

            execution_time = (datetime.now() - start_time).total_seconds()

            print(f"✅ Task completed in {execution_time:.2f}s")

            # Update stats
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += execution_time
            self.stats["task_durations"].append(execution_time)  # Track for average

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Task execution failed: {e}"

            print(f"❌ Task failed: {error_msg}")

            # Update stats
            self.stats["tasks_failed"] += 1
            self.stats["total_execution_time"] += execution_time

            raise Exception(error_msg)
        
        finally:
            # Stop checkpoint monitoring
            await self.checkpoint_handler.stop_checkpoint_monitoring()
            print(f"🛑 Checkpoint monitoring stopped for task {task_id}")
            
            # Clean up checkpoint state from builtins
            if hasattr(builtins, '_checkpoint_state'):
                delattr(builtins, '_checkpoint_state')

    async def _execute_resumed_task(
        self, func_code: str, checkpoint_state: dict, task_id: str, job_id: str
    ) -> Any:
        """
        Execute a resumed task, continuing from checkpoint state.
        
        The checkpoint_state is pre-loaded into builtins._checkpoint_state so the
        deserialized function can access its previous progress and continue.
        
        Args:
            func_code: Serialized function code
            checkpoint_state: Dictionary with saved state from last checkpoint
            task_id: Task identifier
            job_id: Job identifier
            
        Returns:
            Task result
        """
        start_time = datetime.now()
        
        # Pre-load checkpoint state so the function can resume from it
        import builtins
        
        # Restore the checkpoint state for the function to use
        builtins._checkpoint_state = dict(checkpoint_state)
        builtins._checkpoint_state["_is_resumed"] = True
        builtins._checkpoint_state["task_id"] = task_id
        
        print(f"[load] Pre-loaded checkpoint state into builtins._checkpoint_state")
        
        # Define state getter for checkpoint handler
        def get_task_state() -> Dict[str, Any]:
            """Get current task state for checkpointing"""
            try:
                if hasattr(builtins, '_checkpoint_state'):
                    state = builtins._checkpoint_state
                    return dict(state)
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
            print(f"Executing resumed task... | worker_runtime={get_runtime_info()}")

            # Deserialize the function
            func = deserialize_function_for_PC(func_code)
            
            # Start checkpoint monitoring in background
            await self.checkpoint_handler.start_checkpoint_monitoring(
                task_id=task_id,
                get_state_callback=get_task_state,
                send_checkpoint_callback=send_checkpoint
            )
            
            print(f"Checkpoint monitoring started for resumed task {task_id}")

            # For resumed tasks, we need to extract the original args from checkpoint
            # The function should check builtins._checkpoint_state for resume info
            # and continue from where it left off
            
            # Get the original num_trials from checkpoint (specific to monte carlo)
            num_trials = checkpoint_state.get("num_trials", 125000)
            
            print(f"[Info] Resuming with original args: num_trials={num_trials}")
            print(f"Starting from: trials_completed={checkpoint_state.get('trials_completed', 0)}, "
                  f"progress={checkpoint_state.get('progress_percent', 0):.1f}%")

            # Execute the function with original arguments
            # The function should detect _is_resumed in checkpoint_state and continue
            result = await asyncio.to_thread(func, num_trials)

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
            await self.checkpoint_handler.stop_checkpoint_monitoring()
            print(f"Checkpoint monitoring stopped for resumed task {task_id}")
            
            # Clean up checkpoint state from builtins
            if hasattr(builtins, '_checkpoint_state'):
                delattr(builtins, '_checkpoint_state')

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
