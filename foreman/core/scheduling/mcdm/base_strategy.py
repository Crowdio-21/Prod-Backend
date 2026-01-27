"""
Base classes for MCDM allocation strategies

Ported from TEMP_ALGO_FOLDER with adaptations for CrowdCompute
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class AllocationStrategy(ABC):
    """
    Abstract Base Class for Task Allocation Algorithms.
    Enables Strategy Pattern for hot-swapping algorithms at runtime.
    """

    def __init__(self, criteria_weights: np.ndarray = None):
        """
        Initialize allocation strategy

        Args:
            criteria_weights: Numpy array of weights for each criterion
        """
        self.weights = criteria_weights

    @abstractmethod
    def rank_devices(
        self, decision_matrix: np.ndarray, criteria_types: List[int]
    ) -> List[int]:
        """
        Rank devices based on the implemented algorithm.

        Args:
            decision_matrix: Numpy array (n_alternatives x m_criteria)
            criteria_types: List of +1 (Benefit) or -1 (Cost)

        Returns:
            List of device indices sorted by rank (best to worst)
        """
        raise NotImplementedError

    def set_weights(self, weights: np.ndarray):
        """
        Update criteria weights

        Args:
            weights: New weights array
        """
        self.weights = weights
