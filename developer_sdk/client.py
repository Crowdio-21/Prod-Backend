import asyncio 
import json
import uuid
import websockets
from typing import Any, Callable, List, Optional, Dict
from common.protocol import (
    Message, MessageType, create_submit_job_message, 
    create_ping_message, create_pong_message
)
from common.serializer import serialize_function
from .decorators import get_task_metadata, TaskMetadata


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
                    checkpoint_enabled=kwargs.get('checkpoint', metadata.checkpoint_enabled),
                    checkpoint_interval=kwargs.get('checkpoint_interval', metadata.checkpoint_interval),
                    checkpoint_state=kwargs.get('checkpoint_state', metadata.checkpoint_state),
                    parallel=kwargs.get('parallel', metadata.parallel),
                    retry_on_failure=kwargs.get('retry_on_failure', metadata.retry_on_failure),
                    max_retries=kwargs.get('max_retries', metadata.max_retries),
                    timeout=kwargs.get('timeout', metadata.timeout),
                    _func_name=metadata._func_name
                )
            else:
                # Create default metadata if not decorated
                task_metadata = TaskMetadata(
                    checkpoint_enabled=kwargs.get('checkpoint', False),
                    checkpoint_interval=kwargs.get('checkpoint_interval', 10.0),
                    checkpoint_state=kwargs.get('checkpoint_state', []),
                    parallel=kwargs.get('parallel', True),
                    retry_on_failure=kwargs.get('retry_on_failure', True),
                    max_retries=kwargs.get('max_retries', 3),
                    timeout=kwargs.get('timeout', None),
                    _func_name=func.__name__
                )
            
            # Serialize function (use original if decorated)
            if hasattr(func, '__crowdio_original__'):
                func_code = serialize_function(func.__crowdio_original__)
            else:
                func_code = serialize_function(func)
            
            # Create submission message with metadata
            message = self._create_submit_job_message_with_metadata(
                func_code, iterable, job_id, task_metadata
            )
            
            # Store job metadata for tracking
            self._submitted_jobs[job_id] = {
                'metadata': task_metadata,
                'func_name': task_metadata._func_name,
                'total_tasks': len(iterable),
                'checkpoint_enabled': task_metadata.checkpoint_enabled
            }
            
            # Send to foreman
            await self.websocket.send(message.to_json())
            
            if task_metadata.checkpoint_enabled:
                print(f"[SDK] Job {job_id[:8]}... submitted with checkpointing enabled "
                      f"(interval: {task_metadata.checkpoint_interval}s, "
                      f"state vars: {task_metadata.checkpoint_state or 'all'})")
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
    
    def _create_submit_job_message_with_metadata(
        self, 
        func_code: str, 
        args_list: List[Any], 
        job_id: str,
        metadata: TaskMetadata
    ) -> Message:
        """
        Create a job submission message with checkpoint metadata
        
        Args:
            func_code: Serialized function code
            args_list: List of task arguments
            job_id: Job identifier
            metadata: Task checkpoint metadata
            
        Returns:
            Message object ready to send
        """
        return Message(
            msg_type=MessageType.SUBMIT_JOB,
            data={
                "func_code": func_code,
                "args_list": args_list,
                "total_tasks": len(args_list),
                "task_metadata": metadata.to_dict()
            },
            job_id=job_id
        )
    
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
                checkpoint_enabled=kwargs.get('checkpoint', metadata.checkpoint_enabled),
                checkpoint_interval=kwargs.get('checkpoint_interval', metadata.checkpoint_interval),
                checkpoint_state=kwargs.get('checkpoint_state', metadata.checkpoint_state),
                parallel=metadata.parallel,
                retry_on_failure=metadata.retry_on_failure,
                max_retries=metadata.max_retries,
                timeout=metadata.timeout,
                _func_name=metadata._func_name
            )
        else:
            task_metadata = TaskMetadata(
                checkpoint_enabled=kwargs.get('checkpoint', False),
                checkpoint_interval=kwargs.get('checkpoint_interval', 10.0),
                _func_name=func.__name__
            )
        
        # Serialize function
        if hasattr(func, '__crowdio_original__'):
            func_code = serialize_function(func.__crowdio_original__)
        else:
            func_code = serialize_function(func)
        
        # Create future for this job
        future = asyncio.Future()
        self.pending_jobs[job_id] = future
        
        # Store job metadata
        self._submitted_jobs[job_id] = {
            'metadata': task_metadata,
            'func_name': task_metadata._func_name,
            'total_tasks': len(iterable),
            'checkpoint_enabled': task_metadata.checkpoint_enabled,
            'future': future
        }
        
        # Create and send message
        message = self._create_submit_job_message_with_metadata(
            func_code, iterable, job_id, task_metadata
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
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
        else:
            results = await future
        
        # Clean up
        del self.pending_jobs[job_id]
        if job_id in self._submitted_jobs:
            del self._submitted_jobs[job_id]
        
        return results
    
    async def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run a single function with arguments in one worker"""
        # Like AWS lambda
        pass

