"""
Base MCDM Scheduler - Adapter between MCDM strategies and TaskScheduler interface

This module bridges the MCDM allocation strategies (ARAS, EDAS, MABAC, WRR)
with the CrowdCompute TaskScheduler interface.
"""

import json
import numpy as np
from typing import List, Optional, Set, Dict
from abc import ABC

from ..scheduler_interface import TaskScheduler, Task, Worker
from .base_strategy import AllocationStrategy


class BaseMCDMScheduler(TaskScheduler, ABC):
    """
    Base class for MCDM schedulers

    Bridges MCDM AllocationStrategy with TaskScheduler interface.
    Handles decision matrix construction and worker/task selection.
    """

    def __init__(
        self,
        strategy: AllocationStrategy,
        criteria_weights: List[float],
        criteria_names: List[str],
        criteria_types: List[int],
    ):
        """
        Initialize MCDM scheduler

        Args:
            strategy: MCDM strategy instance (ARAS, EDAS, MABAC, WRR)
            criteria_weights: List of weights for each criterion (must sum to ~1.0)
            criteria_names: List of worker attribute names to use as criteria
            criteria_types: List of +1 (benefit) or -1 (cost) for each criterion
        """
        self.strategy = strategy
        self.criteria_weights = np.array(criteria_weights)
        self.criteria_names = criteria_names
        self.criteria_types = criteria_types

        # Set weights in strategy
        self.strategy.set_weights(self.criteria_weights)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate scheduler configuration"""
        # Check weights sum to approximately 1.0
        weight_sum = np.sum(self.criteria_weights)
        if not (0.95 <= weight_sum <= 1.05):
            print(f"Warning: Criteria weights sum to {weight_sum}, not 1.0")

        # Check all lists have same length
        if not (
            len(self.criteria_weights)
            == len(self.criteria_names)
            == len(self.criteria_types)
        ):
            raise ValueError(
                f"Criteria configuration mismatch: "
                f"weights={len(self.criteria_weights)}, "
                f"names={len(self.criteria_names)}, "
                f"types={len(self.criteria_types)}"
            )

    def _build_decision_matrix(
        self, workers: Dict[str, Worker]
    ) -> tuple[np.ndarray, List[str]]:
        """
        Build decision matrix from worker objects

        Args:
            workers: Dictionary of Worker objects {worker_id: Worker}

        Returns:
            Tuple of (decision_matrix, worker_ids)
            - decision_matrix: (n_workers × m_criteria) numpy array
            - worker_ids: List of worker IDs corresponding to matrix rows
        """
        if not workers:
            return np.array([]), []

        worker_ids = list(workers.keys())
        n_workers = len(worker_ids)
        n_criteria = len(self.criteria_names)

        # Initialize matrix
        matrix = np.zeros((n_workers, n_criteria))

        # Fill matrix with worker data
        for i, worker_id in enumerate(worker_ids):
            worker = workers[worker_id]

            for j, criterion in enumerate(self.criteria_names):
                # Get value from worker object
                value = getattr(worker, criterion, 0.0)

                # Handle None values
                if value is None:
                    value = 0.0

                # Handle boolean values
                if isinstance(value, bool):
                    value = 1.0 if value else 0.0

                # Convert to float
                try:
                    matrix[i, j] = float(value)
                except (TypeError, ValueError):
                    matrix[i, j] = 0.0
                    print(
                        f"Warning: Could not convert {criterion}={value} to float for worker {worker_id}"
                    )

        return matrix, worker_ids

    async def select_worker(
        self, task: Task, available_workers: Set[str], all_workers: Dict[str, Worker]
    ) -> Optional[str]:
        """
        Select best worker for a task using MCDM algorithm

        Args:
            task: Task to be assigned
            available_workers: Set of available worker IDs
            all_workers: Dictionary of all workers

        Returns:
            Selected worker ID or None if no suitable worker
        """
        if not available_workers or not all_workers:
            return None

        try:
            # Build decision matrix from all workers
            matrix, worker_ids = self._build_decision_matrix(all_workers)

            if matrix.size == 0:
                return None

            # Rank workers using MCDM algorithm
            ranked_indices = self.strategy.rank_devices(matrix, self.criteria_types)

            # Return first available worker from ranked list
            for idx in ranked_indices:
                worker_id = worker_ids[idx]
                if worker_id in available_workers:
                    # Log decision (optional)
                    # print(f"MCDM: Selected worker {worker_id} (rank {ranked_indices.index(idx) + 1}/{len(ranked_indices)})")
                    return worker_id

            # No available workers in ranked list
            return None

        except Exception as e:
            print(f"Error in MCDM select_worker: {e}")
            import traceback

            traceback.print_exc()
            # Fallback to first available worker
            return list(available_workers)[0] if available_workers else None

    async def select_task(
        self, pending_tasks: List[Task], worker_id: str
    ) -> Optional[Task]:
        """
        Select best task for a worker

        For MCDM schedulers, we use simple FIFO for task selection
        since the intelligence is in worker selection.

        Args:
            pending_tasks: List of pending tasks
            worker_id: Worker requesting a task

        Returns:
            Selected task or None
        """
        if not pending_tasks:
            return None

        # Simple FIFO for now - can be enhanced later
        # Could consider task priority, job affinity, etc.
        return pending_tasks[0]

    def get_config(self) -> Dict:
        """
        Get current scheduler configuration

        Returns:
            Dictionary with scheduler config
        """
        return {
            "algorithm": self.strategy.__class__.__name__,
            "criteria_names": self.criteria_names,
            "criteria_weights": self.criteria_weights.tolist(),
            "criteria_types": self.criteria_types,
        }

    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(strategy={self.strategy.__class__.__name__})"
