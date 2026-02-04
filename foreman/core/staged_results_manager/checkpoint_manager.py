"""
Foreman-side checkpoint management for incremental task state recovery

Handles storing, validating, compacting, and reconstructing checkpoint data.
Uses hybrid storage: SQLite for deltas <1MB, filesystem for larger checkpoints.
Supports declarative checkpointing with configurable state variables.
"""

import json
import gzip
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from foreman.db.models import TaskModel
from .storage_handler import StorageHandler
from common.protocol import CheckpointType


class CheckpointManager:
    """Manages task checkpoints on foreman side"""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory for large checkpoint storage
        """
        self.checkpoint_dir = checkpoint_dir
        self.storage_handler = StorageHandler(checkpoint_dir)
        
        # Create checkpoint directory if needed
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    async def store_checkpoint(
        self,
        session: AsyncSession,
        task_id: str,
        job_id: str,
        is_base: bool,
        delta_data_bytes: bytes,
        progress_percent: float,
        checkpoint_id: int,
        compression_type: str = "gzip",
        checkpoint_type: str = None,
        checkpoint_state_vars: List[str] = None,
        state_size_bytes: int = None
    ) -> bool:
        """
        Store a checkpoint from worker
        
        Args:
            session: Database session
            task_id: Task identifier
            job_id: Job identifier
            is_base: True if base checkpoint, False if delta
            delta_data_bytes: Raw checkpoint data (already decompressed)
            progress_percent: Task progress (0-100)
            checkpoint_id: Sequential checkpoint number
            compression_type: Type of compression applied
            checkpoint_type: Type of checkpoint (BASE, DELTA, COMPACTED) - from declarative API
            checkpoint_state_vars: List of state variables being checkpointed
            state_size_bytes: Uncompressed size of checkpoint state
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Get or create task model
            task = await session.get(TaskModel, task_id)
            if not task:
                print(f"CheckpointManager: Task {task_id} not found")
                return False
            
            # Update progress
            task.progress_percent = progress_percent
            task.last_checkpoint_at = datetime.now()
            task.checkpoint_count = checkpoint_id
            
            # Determine checkpoint type - use explicit type if provided, else infer from is_base
            effective_checkpoint_type = checkpoint_type
            if effective_checkpoint_type is None:
                effective_checkpoint_type = CheckpointType.BASE.value if is_base else CheckpointType.DELTA.value
            
            # Log declarative checkpointing details if state vars provided
            state_vars_str = ""
            if checkpoint_state_vars:
                state_vars_str = f" | State vars: {checkpoint_state_vars}"
            
            if is_base or effective_checkpoint_type == CheckpointType.BASE.value:
                # Store base checkpoint - returns (storage_ref, blob_data_or_None)
                storage_ref, blob_data = await self.storage_handler.store_checkpoint(
                    task_id, delta_data_bytes, is_base=True
                )
                task.base_checkpoint_data = storage_ref
                task.base_checkpoint_size = len(delta_data_bytes)
                task.delta_checkpoints = json.dumps([])
                
                # Store blob in database if small enough, otherwise set filesystem path
                if blob_data is not None:
                    task.base_checkpoint_blob = blob_data
                    task.checkpoint_storage_path = "db"
                    storage_location = "database (blob)"
                else:
                    task.base_checkpoint_blob = None
                    task.checkpoint_storage_path = os.path.join(self.checkpoint_dir, task_id)
                    storage_location = task.checkpoint_storage_path
                
                # Clear delta blobs when new base is set
                task.delta_checkpoint_blobs = json.dumps({})
                
                print(f"[Checkpoint DB] Task {task_id} | BASE #{checkpoint_id} | "
                      f"Size: {len(delta_data_bytes):,} bytes | Progress: {progress_percent:.1f}% | "
                      f"Storage: {storage_location}{state_vars_str}")
            else:
                # Append delta checkpoint
                deltas = json.loads(task.delta_checkpoints or "[]")
                delta_blobs = json.loads(task.delta_checkpoint_blobs or "{}")
                
                delta_info = {
                    "checkpoint_id": checkpoint_id,
                    "size": len(delta_data_bytes),
                    "stored_at": datetime.now().isoformat(),
                    "compression": compression_type,
                    "checkpoint_type": effective_checkpoint_type,
                }
                
                # Add state vars if provided (declarative checkpointing)
                if checkpoint_state_vars:
                    delta_info["state_vars"] = checkpoint_state_vars
                if state_size_bytes is not None:
                    delta_info["uncompressed_size"] = state_size_bytes
                
                # Store delta and get reference + optional blob
                storage_ref, blob_data = await self.storage_handler.store_checkpoint(
                    task_id, delta_data_bytes, is_base=False, checkpoint_id=checkpoint_id
                )
                delta_info["storage_ref"] = storage_ref
                
                # Store blob in database if small enough
                if blob_data is not None:
                    import base64
                    delta_blobs[str(checkpoint_id)] = base64.b64encode(blob_data).decode('ascii')
                    delta_info["storage_type"] = "db"
                else:
                    delta_info["storage_type"] = "fs"
                
                deltas.append(delta_info)
                task.delta_checkpoints = json.dumps(deltas)
                task.delta_checkpoint_blobs = json.dumps(delta_blobs)
                
                storage_type = "database" if blob_data else "filesystem"
                print(f"[Checkpoint DB] Task {task_id} | DELTA #{checkpoint_id} | "
                      f"Size: {len(delta_data_bytes):,} bytes | Progress: {progress_percent:.1f}% | "
                      f"Total deltas: {len(deltas)} | Storage: {storage_type}{state_vars_str}")
                
                # Compact if too many deltas (merge into new base every 50)
                if len(deltas) >= 50:
                    print(f"[Checkpoint DB] Compacting {len(deltas)} deltas for task {task_id}...")
                    await self._compact_checkpoints(session, task_id)
            
            await session.commit()
            return True
            
        except Exception as e:
            print(f"CheckpointManager: Error storing checkpoint for {task_id}: {e}")
            await session.rollback()
            return False
    
    async def get_latest_checkpoint(
        self,
        session: AsyncSession,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve latest checkpoint for a task
        
        Args:
            session: Database session
            task_id: Task identifier
            
        Returns:
            Dictionary with checkpoint data or None if not found
        """
        try:
            task = await session.get(TaskModel, task_id)
            if not task or not task.base_checkpoint_data:
                return None
            
            return {
                "task_id": task_id,
                "progress_percent": task.progress_percent,
                "checkpoint_count": task.checkpoint_count,
                "last_checkpoint_at": task.last_checkpoint_at,
                "base_checkpoint_size": task.base_checkpoint_size,
                "delta_count": len(json.loads(task.delta_checkpoints or "[]"))
            }
            
        except Exception as e:
            print(f"CheckpointManager: Error retrieving checkpoint for {task_id}: {e}")
            return None
    
    async def reconstruct_state(
        self,
        session: AsyncSession,
        task_id: str
    ) -> Optional[bytes]:
        """
        Reconstruct full state from base + deltas
        
        For recovery when worker fails mid-execution. Merges all deltas sequentially.
        
        Args:
            session: Database session
            task_id: Task identifier
            
        Returns:
            Reconstructed state bytes or None if not available
        """
        try:
            task = await session.get(TaskModel, task_id)
            if not task or not task.base_checkpoint_data:
                return None
            
            # Load base checkpoint
            base_data = await self.storage_handler.retrieve_checkpoint(
                task_id, is_base=True
            )
            if not base_data:
                print(f"CheckpointManager: Could not load base checkpoint for {task_id}")
                return None
            
            # If no deltas, return base
            deltas = json.loads(task.delta_checkpoints or "[]")
            if not deltas:
                return base_data
            
            # Load and merge deltas sequentially
            reconstructed = base_data
            for delta_info in deltas:
                checkpoint_id = delta_info["checkpoint_id"]
                delta_data = await self.storage_handler.retrieve_checkpoint(
                    task_id, is_base=False, checkpoint_id=checkpoint_id
                )
                if delta_data:
                    # Merge delta into reconstructed state
                    reconstructed = await self._merge_delta(reconstructed, delta_data)
                else:
                    print(f"CheckpointManager: Warning - could not load delta {checkpoint_id} "
                          f"for {task_id}, skipping")
            
            print(f"CheckpointManager: Successfully reconstructed state for {task_id} "
                  f"({len(deltas)} deltas merged)")
            return reconstructed
            
        except Exception as e:
            print(f"CheckpointManager: Error reconstructing state for {task_id}: {e}")
            return None
    
    async def cleanup_checkpoint(
        self,
        session: AsyncSession,
        task_id: str
    ) -> bool:
        """
        Delete checkpoints for a task (called when task completes - Option A)
        
        Args:
            session: Database session
            task_id: Task identifier
            
        Returns:
            True if cleanup successful
        """
        try:
            task = await session.get(TaskModel, task_id)
            if not task:
                return False
            
            # Delete storage files
            await self.storage_handler.delete_checkpoints(task_id)
            
            # Clear checkpoint fields from database
            task.base_checkpoint_data = None
            task.base_checkpoint_size = 0
            task.delta_checkpoints = None
            task.checkpoint_count = 0
            task.last_checkpoint_at = None
            
            await session.commit()
            print(f"CheckpointManager: Cleaned up checkpoints for task {task_id}")
            return True
            
        except Exception as e:
            print(f"CheckpointManager: Error cleaning up checkpoints for {task_id}: {e}")
            await session.rollback()
            return False
    
    async def _compact_checkpoints(
        self,
        session: AsyncSession,
        task_id: str
    ) -> None:
        """
        Compact checkpoints by merging deltas into new base (internal)
        
        Reduces number of delta files and improves recovery performance.
        Triggered when delta count reaches threshold (e.g., 50).
        """
        try:
            task = await session.get(TaskModel, task_id)
            if not task:
                return
            
            # Reconstruct full state
            full_state = await self.reconstruct_state(session, task_id)
            if not full_state:
                return
            
            # Delete old checkpoints
            await self.storage_handler.delete_checkpoints(task_id)
            
            # Store as new base
            new_checkpoint_id = task.checkpoint_count + 1
            await self.storage_handler.store_checkpoint(
                task_id, full_state, is_base=True, checkpoint_id=new_checkpoint_id
            )
            
            # Update metadata
            task.base_checkpoint_data = f"stored_{new_checkpoint_id}"
            task.base_checkpoint_size = len(full_state)
            task.delta_checkpoints = json.dumps([])
            task.checkpoint_count = new_checkpoint_id
            
            await session.commit()
            print(f"CheckpointManager: Compacted checkpoints for task {task_id}")
            
        except Exception as e:
            print(f"CheckpointManager: Error compacting checkpoints for {task_id}: {e}")
    
    async def _merge_delta(self, base: bytes, delta: bytes) -> bytes:
        """
        Merge delta into base state (internal, framework-aware)
        
        Uses framework detection to apply deltas correctly:
        - PyTorch: merge weight updates
        - NumPy: array operations
        - Generic: pickle-based dict merging
        """
        try:
            # Try PyTorch first
            try:
                import torch
                import pickle
                
                base_state = pickle.loads(base)
                delta_state = pickle.loads(delta)
                
                if isinstance(base_state, dict) and isinstance(delta_state, dict):
                    # Update weights/parameters
                    merged = base_state.copy()
                    merged.update(delta_state)
                    return pickle.dumps(merged)
            except (ImportError, Exception):
                pass
            
            # Try NumPy array handling
            try:
                import numpy as np
                import pickle
                
                base_state = pickle.loads(base)
                delta_state = pickle.loads(delta)
                
                if isinstance(base_state, np.ndarray) and isinstance(delta_state, np.ndarray):
                    merged = base_state + delta_state
                    return pickle.dumps(merged)
            except (ImportError, Exception):
                pass
            
            # Fallback: generic dict merge
            import pickle
            base_state = pickle.loads(base)
            delta_state = pickle.loads(delta)
            
            if isinstance(base_state, dict) and isinstance(delta_state, dict):
                merged = base_state.copy()
                merged.update(delta_state)
                return pickle.dumps(merged)
            
            # If all else fails, return base unchanged
            print(f"CheckpointManager: Could not merge delta, returning base")
            return base
            
        except Exception as e:
            print(f"CheckpointManager: Error merging delta: {e}")
            return base
