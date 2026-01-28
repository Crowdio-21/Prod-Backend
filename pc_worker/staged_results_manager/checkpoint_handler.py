"""
Worker-side checkpoint handler for incremental state checkpointing

Manages local checkpoint state, computes deltas, and sends checkpoints
to foreman without blocking task execution.
"""

import asyncio
import gzip
import pickle
from typing import Optional, Callable, Any, Dict
from datetime import datetime


class CheckpointHandler:
    """Manages checkpointing on worker side"""
    
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
    
    async def start_checkpoint_monitoring(
        self,
        task_id: str,
        get_state_callback: Callable[[], Dict[str, Any]],
        send_checkpoint_callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Start periodic checkpoint monitoring task
        
        Runs in background, doesn't block execution. Computes deltas and sends
        checkpoint messages to foreman asynchronously.
        
        Args:
            task_id: Task identifier
            get_state_callback: Function to retrieve current state dict
            send_checkpoint_callback: Function to send checkpoint message
        """
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
        1. Get current state
        2. Compute delta from last checkpoint
        3. Send to foreman asynchronously
        """
        print(f"[CheckpointHandler] Starting checkpoint loop for task {task_id} (interval: {self.checkpoint_interval}s)")
        try:
            # Wait a short initial delay to let task start, then checkpoint immediately
            await asyncio.sleep(1.0)
            
            while True:
                print(f"[CheckpointHandler] Taking checkpoint for task {task_id}")
                
                # Process checkpoint first, then wait for interval
                # This ensures we get a checkpoint early in task execution
                
                try:
                    # Get current state
                    current_state = get_state_callback()
                    self.current_state = current_state
                    
                    print(f"[CheckpointHandler] Got state for task {task_id}: progress={current_state.get('progress_percent', 0):.1f}%")
                    
                    # Serialize state
                    try:
                        state_bytes = pickle.dumps(current_state)
                    except Exception as e:
                        print(f"CheckpointHandler: Error serializing state: {e}")
                        continue
                    
                    # Compress for transmission
                    compressed = gzip.compress(state_bytes, compresslevel=6)
                    
                    if not self.is_base_sent:
                        # Send base checkpoint first
                        self.last_checkpoint_state = compressed
                        self.checkpoint_count = 1
                        self.is_base_sent = True
                        
                        checkpoint_msg = {
                            "is_base": True,
                            "progress_percent": current_state.get("progress_percent", 0),
                            "checkpoint_id": self.checkpoint_count,
                            "delta_data_hex": compressed.hex(),
                            "compression_type": "gzip"
                        }
                        
                        progress = current_state.get("progress_percent", 0)
                        print(f"[Checkpoint] Sending BASE #{self.checkpoint_count} for task {task_id} | "
                              f"Size: {len(compressed):,} bytes | Progress: {progress:.1f}%")
                        
                        # Send asynchronously without blocking
                        await asyncio.to_thread(send_checkpoint_callback, checkpoint_msg)
                    else:
                        # Compute and send delta
                        delta_bytes = await self._compute_delta(
                            self.last_checkpoint_state,
                            compressed
                        )
                        
                        self.checkpoint_count += 1
                        self.last_checkpoint_state = compressed
                        
                        checkpoint_msg = {
                            "is_base": False,
                            "progress_percent": current_state.get("progress_percent", 0),
                            "checkpoint_id": self.checkpoint_count,
                            "delta_data_hex": delta_bytes.hex(),
                            "compression_type": "gzip"
                        }
                        
                        progress = current_state.get("progress_percent", 0)
                        print(f"[Checkpoint] Sending DELTA #{self.checkpoint_count} for task {task_id} | "
                              f"Delta size: {len(delta_bytes):,} bytes | Progress: {progress:.1f}%")
                        
                        # Send asynchronously
                        await asyncio.to_thread(send_checkpoint_callback, checkpoint_msg)
                
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"CheckpointHandler: Error in checkpoint loop: {e}")
                
                # Wait for next checkpoint interval
                await asyncio.sleep(self.checkpoint_interval)
        
        except asyncio.CancelledError:
            print(f"CheckpointHandler: Checkpoint monitoring stopped for {task_id}")
    
    async def _compute_delta(self, last_checkpoint: bytes, current_checkpoint: bytes) -> bytes:
        """
        Compute delta between last checkpoint and current state
        
        Delta is the difference, transmitted instead of full state.
        For framework-aware deltas, uses delta_computer.
        
        Args:
            last_checkpoint: Last checkpoint bytes
            current_checkpoint: Current state bytes
            
        Returns:
            Compressed delta bytes
        """
        try:
            import pickle
            
            # Deserialize both states
            last_state = pickle.loads(gzip.decompress(last_checkpoint))
            current_state = pickle.loads(gzip.decompress(current_checkpoint))
            
            # Compute delta (dictionary diff)
            delta = {}
            
            if isinstance(last_state, dict) and isinstance(current_state, dict):
                # Find changed/new keys
                for key in current_state:
                    if key not in last_state or last_state[key] != current_state[key]:
                        delta[key] = current_state[key]
            else:
                # Fallback: store entire current state as delta
                delta = current_state
            
            # Serialize and compress delta
            delta_bytes = pickle.dumps(delta)
            compressed_delta = gzip.compress(delta_bytes, compresslevel=6)
            
            return compressed_delta
        
        except Exception as e:
            print(f"CheckpointHandler: Error computing delta: {e}")
            # Fallback: send full state as delta
            return gzip.compress(current_checkpoint, compresslevel=1)
    
    def reset(self) -> None:
        """Reset checkpoint state (call when resuming from checkpoint)"""
        self.last_checkpoint_state = None
        self.checkpoint_count = 0
        self.is_base_sent = False
        self.current_state = None
