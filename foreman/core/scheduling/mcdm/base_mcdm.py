"""
Base MCDM Scheduler - Adapter between MCDM strategies and TaskScheduler interface

This module bridges the MCDM allocation strategies (ARAS, EDAS, MABAC, WRR)
with the CrowdCompute TaskScheduler interface.
"""

import json
import logging
import numpy as np
from typing import List, Optional, Set, Dict
from abc import ABC
from pathlib import Path

from ..scheduler_interface import TaskScheduler, Task, Worker
from .base_strategy import AllocationStrategy

# Configure MCDM-specific logger
logger = logging.getLogger("mcdm_scheduler")
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# File handler for MCDM decisions
file_handler = logging.FileHandler(log_dir / "mcdm_decisions.log")
file_handler.setLevel(logging.DEBUG)

# Console handler for important info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Detailed formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers if not already added
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


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
            logger.warning(f"Criteria weights sum to {weight_sum}, not 1.0")

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

        # Log configuration on initialization
        logger.info(f"\n{'='*80}")
        logger.info(f"MCDM Scheduler Initialized: {self.strategy.__class__.__name__}")
        logger.info(f"{'='*80}")
        logger.info(f"Criteria Configuration:")
        for i, (name, weight, ctype) in enumerate(
            zip(self.criteria_names, self.criteria_weights, self.criteria_types)
        ):
            criterion_type = "BENEFIT" if ctype == 1 else "COST"
            logger.info(f"  [{i+1}] {name}: weight={weight:.3f}, type={criterion_type}")
        logger.info(f"{'='*80}\n")

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
                    logger.warning(
                        f"Could not convert {criterion}={value} to float for worker {worker_id}"
                    )

        # Log the decision matrix
        logger.debug(
            f"\nDecision Matrix ({n_workers} workers × {n_criteria} criteria):"
        )
        logger.debug(f"Workers: {worker_ids}")
        logger.debug(f"Criteria: {self.criteria_names}")
        logger.debug(f"\nMatrix values:")
        for i, worker_id in enumerate(worker_ids):
            logger.debug(f"  {worker_id}:")
            for j, criterion in enumerate(self.criteria_names):
                criterion_type = "BENEFIT" if self.criteria_types[j] == 1 else "COST"
                logger.debug(f"    {criterion} = {matrix[i, j]:.4f} ({criterion_type})")

        return matrix, worker_ids

    async def select_worker(
        self, task: Task, available_workers: Set[str], all_workers: Dict[str, Worker]
    ) -> Optional[str]:
        logger.info(f"\n{'='*80}")
        logger.info(f"WORKER SELECTION - Task ID: {task.id} (Job: {task.job_id})")
        logger.info(f"{'='*80}")
        logger.info(f"Available workers: {sorted(list(available_workers))}")
        logger.info(f"Total workers in system: {len(all_workers)}")

        if not available_workers or not all_workers:
            logger.warning("No available workers or no workers in system")
            return None

        try:
            # Build decision matrix from all workers
            matrix, worker_ids = self._build_decision_matrix(all_workers)

            if matrix.size == 0:
                logger.warning("Decision matrix is empty")
                return None

            # Rank workers using MCDM algorithm
            logger.info(f"\nRunning {self.strategy.__class__.__name__} algorithm...")
            ranked_indices = self.strategy.rank_devices(matrix, self.criteria_types)

            # Get scores if available from strategy
            scores = None
            if hasattr(self.strategy, "_last_scores"):
                scores = self.strategy._last_scores

            # Log complete ranking with scores
            logger.info(f"\n{'─'*80}")
            logger.info(f"WORKER RANKINGS (Best to Worst):")
            logger.info(f"{'─'*80}")
            for rank, idx in enumerate(ranked_indices, 1):
                worker_id = worker_ids[idx]
                is_available = (
                    "✓ AVAILABLE" if worker_id in available_workers else "✗ BUSY"
                )
                score_str = f", Score: {scores[idx]:.6f}" if scores is not None else ""
                logger.info(f"  Rank {rank}: {worker_id} [{is_available}]{score_str}")

                # Log worker details for top 3
                if rank <= 3:
                    worker = all_workers[worker_id]
                    logger.debug(
                        f"    Details: status={worker.status}, completed={worker.tasks_completed}, failed={worker.tasks_failed}"
                    )

            logger.info(f"{'─'*80}")

            # Return first available worker from ranked list
            selected_worker = None
            selected_rank = None
            for rank, idx in enumerate(ranked_indices, 1):
                worker_id = worker_ids[idx]
                if worker_id in available_workers:
                    selected_worker = worker_id
                    selected_rank = rank
                    break

            if selected_worker:
                score_str = (
                    f" with score {scores[ranked_indices[selected_rank-1]]:.6f}"
                    if scores is not None
                    else ""
                )
                logger.info(
                    f"\n✓ SELECTED: {selected_worker} (Rank {selected_rank}/{len(ranked_indices)}){score_str}"
                )
                logger.info(f"{'='*80}\n")
                return selected_worker
            else:
                logger.warning("No available workers in ranked list")
                logger.info(f"{'='*80}\n")
                return None

        except Exception as e:
            logger.error(f"Error in MCDM select_worker: {e}", exc_info=True)
            # Fallback to first available worker
            fallback = list(available_workers)[0] if available_workers else None
            if fallback:
                logger.warning(f"Using fallback selection: {fallback}")
            return fallback

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

    async def batch_select_workers(
        self,
        tasks: List[Task],
        available_workers: Set[str],
        all_workers: Dict[str, Worker],
    ) -> List[tuple]:
        """
        Optimized batch assignment: select workers for multiple tasks at once.

        This method ranks workers ONCE and assigns them to tasks, avoiding
        redundant MCDM computations.

        Optimization strategy:
        - If tasks >= available_workers: Skip ranking entirely, assign all workers
        - If tasks < available_workers: Rank once, pick top N workers

        Args:
            tasks: List of tasks to assign
            available_workers: Set of available worker IDs
            all_workers: Dictionary of all workers (id -> Worker)

        Returns:
            List of (task, worker_id) tuples for successful assignments
        """
        logger.info(f"\n{'='*80}")
        logger.info(
            f"BATCH WORKER SELECTION - {len(tasks)} tasks, {len(available_workers)} available workers"
        )
        logger.info(f"{'='*80}")

        if not tasks or not available_workers:
            logger.warning("No tasks or no available workers for batch assignment")
            return []

        available_list = list(available_workers)
        n_tasks = len(tasks)
        n_workers = len(available_list)

        # OPTIMIZATION 1: If tasks >= workers, skip ranking - assign all workers
        if n_tasks >= n_workers:
            logger.info(f"⚡ FAST PATH: {n_tasks} tasks >= {n_workers} workers")
            logger.info(
                f"   Skipping MCDM ranking - assigning all available workers directly"
            )

            assignments = []
            for i, worker_id in enumerate(available_list):
                if i < len(tasks):
                    assignments.append((tasks[i], worker_id))
                    logger.info(f"   Assigned task {tasks[i].id} → {worker_id}")

            logger.info(f"✓ Fast-path assigned {len(assignments)} tasks")
            logger.info(f"{'='*80}\n")
            return assignments

        # OPTIMIZATION 2: tasks < workers - rank ONCE and pick top N
        logger.info(f"🎯 RANKED PATH: {n_tasks} tasks < {n_workers} workers")
        logger.info(f"   Running MCDM ranking ONCE to select best {n_tasks} workers")

        try:
            # Build decision matrix ONCE for all available workers only
            available_workers_dict = {
                wid: all_workers[wid] for wid in available_list if wid in all_workers
            }
            matrix, worker_ids = self._build_decision_matrix(available_workers_dict)

            if matrix.size == 0:
                logger.warning("Decision matrix is empty")
                return []

            # Rank workers using MCDM algorithm - SINGLE computation
            logger.info(f"   Running {self.strategy.__class__.__name__} algorithm...")
            ranked_indices = self.strategy.rank_devices(matrix, self.criteria_types)

            # Get scores if available
            scores = getattr(self.strategy, "_last_scores", None)

            # Log complete ranking with clear separation of selected vs waitlisted
            logger.info(f"\n{'─'*80}")
            logger.info(f"WORKER RANKINGS (computed once for batch):")
            logger.info(f"{'─'*80}")

            ranking_info = []
            for rank, idx in enumerate(ranked_indices, 1):
                worker_id = worker_ids[idx]
                score_value = None
                if scores is not None:
                    score_value = float(scores[idx])
                ranking_info.append((rank, worker_id, score_value))

            selected_count = min(n_tasks, len(ranking_info))
            if selected_count:
                logger.info(f"SELECTED WORKERS (top {selected_count}):")
                for rank, worker_id, score_value in ranking_info[:selected_count]:
                    score_str = (
                        f", Score: {score_value:.6f}" if score_value is not None else ""
                    )
                    logger.info(f"  Rank {rank}: {worker_id}{score_str} [SELECTED]")

            if selected_count < len(ranking_info):
                logger.info("WAITLISTED WORKERS:")
                for rank, worker_id, score_value in ranking_info[selected_count:]:
                    score_str = (
                        f", Score: {score_value:.6f}" if score_value is not None else ""
                    )
                    logger.info(f"  Rank {rank}: {worker_id}{score_str}")

            logger.info(f"{'─'*80}")

            # Assign tasks to top-ranked workers
            assignments = []
            for i, task in enumerate(tasks):
                if i < len(ranked_indices):
                    best_worker_idx = ranked_indices[i]
                    worker_id = worker_ids[best_worker_idx]
                    assignments.append((task, worker_id))
                    logger.info(f"   Task {task.id} → {worker_id} (Rank {i+1})")

            logger.info(
                f"\n✓ Ranked-path assigned {len(assignments)} tasks using single MCDM computation"
            )
            logger.info(f"{'='*80}\n")
            return assignments

        except Exception as e:
            logger.error(f"Error in batch_select_workers: {e}", exc_info=True)
            # Fallback: simple assignment without ranking
            assignments = []
            for i, worker_id in enumerate(available_list):
                if i < len(tasks):
                    assignments.append((tasks[i], worker_id))
            logger.warning(
                f"Fallback: assigned {len(assignments)} tasks without ranking"
            )
            return assignments

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
