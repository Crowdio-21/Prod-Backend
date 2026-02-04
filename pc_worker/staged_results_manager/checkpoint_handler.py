"""
Worker-side checkpoint handler for incremental state checkpointing

Manages local checkpoint state, computes deltas, and sends checkpoints
to foreman without blocking task execution.

Extended with support for:
- Declarative checkpoint variables (only checkpoint specified variables)
- Task metadata configuration
- Thread-safe state access
"""

import asyncio
import gzip
import pickle
import threading
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime


class CheckpointHandler:
    """
    Manages checkpointing on worker side
    
    Supports declarative checkpointing where only specified variables
    are captured, reducing checkpoint size and overhead.
    """
    
    def __init__(self, checkpoint_interval: float = 10.0):
        """
        Initialize checkpoint handler
        
        Args:
            checkpoint_interval: Seconds between local checkpoints (default 10s)
        """
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_state: Optional[bytes] = None
        self.checkpoint_count = 0
        self.is_base_sent = False
        self.checkpoint_task: Optional[asyncio.Task] = None
        self.current_state: Optional[Dict[str, Any]] = None
        
        # Task metadata for declarative checkpointing
        self._task_metadata: Optional[Dict[str, Any]] = None
        self._checkpoint_state_vars: List[str] = []
        self._checkpoint_enabled: bool = True
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_checkpoints": 0,
            "base_checkpoints": 0,
            "delta_checkpoints": 0,
            "total_bytes_sent": 0,
            "last_checkpoint_time": None
        }
    
    def configure(self, task_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Configure checkpoint handler with task metadata
        
        Args:
            task_metadata: Configuration from @task decorator including:
                - checkpoint_enabled: Whether to checkpoint
                - checkpoint_interval: Override interval
                - checkpoint_state: List of variable names to checkpoint
        """
        if not task_metadata:
            self._task_metadata = None
            self._checkpoint_state_vars = []
            self._checkpoint_enabled = True
            return
        
        self._task_metadata = task_metadata
        self._checkpoint_enabled = task_metadata.get("checkpoint_enabled", True)
        self._checkpoint_state_vars = task_metadata.get("checkpoint_state", [])
        
        # Override interval if specified
        if "checkpoint_interval" in task_metadata:
            self.checkpoint_interval = task_metadata["checkpoint_interval"]
        
        if self._checkpoint_state_vars:
            print(f"[CheckpointHandler] Configured to checkpoint variables: {self._checkpoint_state_vars}")
        else:
            print(f"[CheckpointHandler] Configured to checkpoint all state variables")
    
    def filter_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter state to only include declared checkpoint variables
        
        Args:
            state: Full state dictionary
            
        Returns:
            Filtered state with only declared variables (or all if none declared)
        """
        if not self._checkpoint_state_vars:
            # No specific variables - return all state
            return dict(state)
        
        # Filter to only declared variables
        filtered = {}
        for var in self._checkpoint_state_vars:
            if var in state:
                filtered[var] = state[var]
            else:
                # Variable not yet set - skip silently
                pass
        
        # Always include progress_percent if available
        if "progress_percent" in state and "progress_percent" not in filtered:
            filtered["progress_percent"] = state["progress_percent"]
        
        return filtered
    
    async def start_checkpoint_monitoring(
        self,
        task_id: str,
        get_state_callback: Callable[[], Dict[str, Any]],
        send_checkpoint_callback: Callable[[Dict[str, Any]], None],
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start periodic checkpoint monitoring task
        
        Runs in background, doesn't block execution. Computes deltas and sends
        checkpoint messages to foreman asynchronously.
        
        Args:
            task_id: Task identifier
            get_state_callback: Function to retrieve current state dict
            send_checkpoint_callback: Function to send checkpoint message
            task_metadata: Optional configuration from @task decorator
        """
        # Configure with metadata
        self.configure(task_metadata)
        
        # Check if checkpointing is disabled
        if not self._checkpoint_enabled:
            print(f"[CheckpointHandler] Checkpointing disabled for task {task_id}")
            return
        
        # Reset state for new task
        self.reset()
        
        self.checkpoint_task = asyncio.create_task(
            self._checkpoint_loop(task_id, get_state_callback, send_checkpoint_callback)
        )
    
    async def stop_checkpoint_monitoring(self) -> None:
        """Stop the checkpoint monitoring task"""
        if self.checkpoint_task:
            self.checkpoint_task.cancel()
            try:
                await self.checkpoint_task
            except asyncio.CancelledError:
                pass
    
    async def _checkpoint_loop(
        self,
        task_id: str,
        get_state_callback: Callable[[], Dict[str, Any]],
        send_checkpoint_callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Internal checkpoint loop (runs in background)
        
        Periodically:
        1. Get current state (filtered if variables declared)
        2. Compute delta from last checkpoint
        3. Send to foreman asynchronously
        """
        checkpoint_vars_str = ", ".join(self._checkpoint_state_vars) if self._checkpoint_state_vars else "all"
        print(f"[CheckpointHandler] Starting checkpoint loop for task {task_id} "
              f"(interval: {self.checkpoint_interval}s, variables: {checkpoint_vars_str})")
        
        try:
            # Wait a short initial delay to let task start, then checkpoint
            await asyncio.sleep(0.5)
            
            while True:
                # Process checkpoint first, then wait for interval
                try:
                    # Get current state from callback
                    raw_state = get_state_callback()
                    
                    # Filter to declared variables only
                    current_state = self.filter_state(raw_state)
                    
                    with self._lock:
                        self.current_state = current_state
                    
                    progress = current_state.get("progress_percent", 0)
                    print(f"[CheckpointHandler] Got state for task {task_id}: "
                          f"progress={progress:.1f}%, vars={list(current_state.keys())}")
                    
                    # Serialize state
                    try:
                        state_bytes = pickle.dumps(current_state)
                        state_size = len(state_bytes)
                    except Exception as e:
                        print(f"[CheckpointHandler] Error serializing state: {e}")
                        await asyncio.sleep(self.checkpoint_interval)
                        continue
                    
                    # Compress for transmission
                    compressed = gzip.compress(state_bytes, compresslevel=6)
                    
                    if not self.is_base_sent:
                        # Send base checkpoint first
                        await self._send_base_checkpoint(
                            task_id, compressed, current_state, state_size, send_checkpoint_callback
                        )
                    else:
                        # Compute and send delta
                        await self._send_delta_checkpoint(
                            task_id, compressed, current_state, state_size, send_checkpoint_callback
                        )
                
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[CheckpointHandler] Error in checkpoint loop: {e}")
                
                # Wait for next checkpoint interval
                await asyncio.sleep(self.checkpoint_interval)
        
        except asyncio.CancelledError:
            print(f"[CheckpointHandler] Checkpoint monitoring stopped for {task_id}")
    
    async def _send_base_checkpoint(
        self,
        task_id: str,
        compressed: bytes,
        current_state: Dict[str, Any],
        state_size: int,
        send_checkpoint_callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Send base checkpoint"""
        with self._lock:
            self.last_checkpoint_state = compressed
            self.checkpoint_count = 1
            self.is_base_sent = True
        
        checkpoint_msg = {
            "is_base": True,
            "checkpoint_type": "base",
            "progress_percent": current_state.get("progress_percent", 0),
            "checkpoint_id": self.checkpoint_count,
            "delta_data_hex": compressed.hex(),
            "compression_type": "gzip",
            "checkpoint_state_vars": self._checkpoint_state_vars,
            "state_size_bytes": state_size
        }
        
        progress = current_state.get("progress_percent", 0)
        vars_info = f"vars: {len(current_state)}" if not self._checkpoint_state_vars else f"declared vars: {len(self._checkpoint_state_vars)}"
        print(f"[Checkpoint] Sending BASE #{self.checkpoint_count} for task {task_id} | "
              f"Size: {len(compressed):,} bytes ({vars_info}) | Progress: {progress:.1f}%")
        
        # Send asynchronously without blocking
        await asyncio.to_thread(send_checkpoint_callback, checkpoint_msg)
        
        # Update stats
        self._stats["total_checkpoints"] += 1
        self._stats["base_checkpoints"] += 1
        self._stats["total_bytes_sent"] += len(compressed)
        self._stats["last_checkpoint_time"] = datetime.now()
    
    async def _send_delta_checkpoint(
        self,
        task_id: str,
        compressed: bytes,
        current_state: Dict[str, Any],
        state_size: int,
        send_checkpoint_callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Compute and send delta checkpoint"""
        # Compute delta
        delta_bytes = await self._compute_delta(
            self.last_checkpoint_state,
            compressed
        )
        
        with self._lock:
            self.checkpoint_count += 1
            self.last_checkpoint_state = compressed
        
        checkpoint_msg = {
            "is_base": False,
            "checkpoint_type": "delta",
            "progress_percent": current_state.get("progress_percent", 0),
            "checkpoint_id": self.checkpoint_count,
            "delta_data_hex": delta_bytes.hex(),
            "compression_type": "gzip",
            "checkpoint_state_vars": self._checkpoint_state_vars,
            "state_size_bytes": state_size
        }
        
        progress = current_state.get("progress_percent", 0)
        print(f"[Checkpoint] Sending DELTA #{self.checkpoint_count} for task {task_id} | "
              f"Delta size: {len(delta_bytes):,} bytes | Progress: {progress:.1f}%")
        
        # Send asynchronously
        await asyncio.to_thread(send_checkpoint_callback, checkpoint_msg)
        
        # Update stats
        self._stats["total_checkpoints"] += 1
        self._stats["delta_checkpoints"] += 1
        self._stats["total_bytes_sent"] += len(delta_bytes)
        self._stats["last_checkpoint_time"] = datetime.now()
    
    async def _compute_delta(self, last_checkpoint: bytes, current_checkpoint: bytes) -> bytes:
        """
        Compute delta between last checkpoint and current state
        
        Delta is the difference, transmitted instead of full state.
        For declared variables, delta only includes changed values.
        
        Args:
            last_checkpoint: Last checkpoint bytes
            current_checkpoint: Current state bytes
            
        Returns:
            Compressed delta bytes
        """
        try:
            # Deserialize both states
            last_state = pickle.loads(gzip.decompress(last_checkpoint))
            current_state = pickle.loads(gzip.decompress(current_checkpoint))
            
            # Compute delta (dictionary diff)
            delta = {}
            
            if isinstance(last_state, dict) and isinstance(current_state, dict):
                # Find changed/new keys
                for key in current_state:
                    if key not in last_state:
                        delta[key] = current_state[key]
                    elif self._values_differ(last_state[key], current_state[key]):
                        delta[key] = current_state[key]
            else:
                # Fallback: store entire current state as delta
                delta = current_state
            
            # Serialize and compress delta
            delta_bytes = pickle.dumps(delta)
            compressed_delta = gzip.compress(delta_bytes, compresslevel=6)
            
            return compressed_delta
        
        except Exception as e:
            print(f"[CheckpointHandler] Error computing delta: {e}")
            # Fallback: send full state as delta
            return gzip.compress(current_checkpoint, compresslevel=1)
    
    def _values_differ(self, val1: Any, val2: Any) -> bool:
        """
        Check if two values are different (handling common types)
        
        Args:
            val1: First value
            val2: Second value
            
        Returns:
            True if values differ
        """
        try:
            # Handle numpy arrays
            if hasattr(val1, 'shape') and hasattr(val2, 'shape'):
                import numpy as np
                return not np.array_equal(val1, val2)
            
            # Standard comparison
            return val1 != val2
        except Exception:
            # If comparison fails, assume different
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        return dict(self._stats)
    
    def reset(self) -> None:
        """Reset checkpoint state (call when starting new task)"""
        with self._lock:
            self.last_checkpoint_state = None
            self.checkpoint_count = 0
            self.is_base_sent = False
            self.current_state = None
            self._stats = {
                "total_checkpoints": 0,
                "base_checkpoints": 0,
                "delta_checkpoints": 0,
                "total_bytes_sent": 0,
                "last_checkpoint_time": None
            }
