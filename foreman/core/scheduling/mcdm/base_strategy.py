"""
Base classes for MCDM allocation strategies

Ported from TEMP_ALGO_FOLDER with adaptations for CrowdCompute
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

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
        self._last_scores = None

    def _calculate_entropy_weights(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates Objective Weights using Shannon Entropy method.
        Returns None if calculation is impossible (e.g., insufficient data).
        
        Args:
            matrix: The decision matrix (n_alternatives x m_criteria)
            
        Returns:
            np.ndarray or None: Array of calculated weights (sum = 1.0) or None on failure
        """
        m_alternatives, n_criteria = matrix.shape

        # Need at least 2 devices to compare variance
        if m_alternatives < 2:
            return None

        try:
            # Step 1: Normalization (p_ij)
            col_sums = matrix.sum(axis=0)
            
            # Handle columns with sum = 0 (avoid div by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                P = matrix / col_sums
                P = np.nan_to_num(P)  # Replace NaNs with 0

            # Step 2: Entropy Calculation (E_j)
            # k = 1 / ln(m)
            k = 1.0 / np.log(m_alternatives)
            
            # Calculate P * ln(P) only where P > 0
            p_ln_p = np.zeros_like(P)
            non_zero_mask = P > 0
            p_ln_p[non_zero_mask] = P[non_zero_mask] * np.log(P[non_zero_mask])
            
            E = -k * np.sum(p_ln_p, axis=0)

            # Step 3: Degree of Divergence (d_j)
            d = 1.0 - E

            # Step 4: Weight Determination (w_j)
            d_sum = d.sum()

            # If all data is identical (d_sum = 0), entropy fails
            if d_sum == 0:
                return None
                
            w = d / d_sum
            return w

        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}. Reverting to static weights.")
            return None

    def _get_active_weights(self, decision_matrix: np.ndarray) -> np.ndarray:
        """
        Helper to determine whether to use Dynamic (Entropy) or Static (Config) weights.
        """
        cols = decision_matrix.shape[1]
        
        # 1. Try Dynamic Weights
        dynamic_weights = self._calculate_entropy_weights(decision_matrix)
        
        if dynamic_weights is not None:
            logger.debug(f"Using DYNAMIC (Shannon Entropy) weights: {np.round(dynamic_weights, 3)}")
            return dynamic_weights
            
        # 2. Fallback to Static Config Weights
        if self.weights is not None and len(self.weights) == cols:
            logger.debug(f"Using STATIC (Config) weights: {self.weights}")
            return self.weights
            
        # 3. Emergency Fallback (Equal Weights)
        logger.warning("No valid weights found. Using EQUAL weights.")
        return np.ones(cols) / cols

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