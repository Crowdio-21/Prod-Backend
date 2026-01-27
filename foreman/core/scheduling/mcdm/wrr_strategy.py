"""
WRR (Weighted Round Robin) Strategy

Ported from TEMP_ALGO_FOLDER/foreman_server/strategies/wrr.py
"""

import numpy as np
from .base_strategy import AllocationStrategy


class WRRStrategy(AllocationStrategy):
    """
    WRR - Weighted Round Robin

    Simplistic baseline: rank simply based on weighted sum of benefit criteria.
    This is a lightweight alternative to more complex MCDM algorithms.
    """

    def rank_devices(self, decision_matrix, criteria_types):
        """
        Rank devices using simple weighted sum

        Args:
            decision_matrix: (n_devices x m_criteria) numpy array
            criteria_types: List of +1 (benefit) or -1 (cost)

        Returns:
            List of device indices sorted by rank (best to worst)
        """
        # Simple approach: weighted sum considering benefit vs cost
        weighted_scores = np.zeros(decision_matrix.shape[0])

        for j in range(decision_matrix.shape[1]):
            if criteria_types[j] == 1:  # Benefit - higher is better
                weighted_scores += decision_matrix[:, j] * self.weights[j]
            else:  # Cost - lower is better, so invert
                max_val = np.max(decision_matrix[:, j])
                if max_val > 0:
                    inverted = max_val - decision_matrix[:, j]
                    weighted_scores += inverted * self.weights[j]
        
        print(f"[WRR STRATEGY]: Weighted Scores: {weighted_scores}")

        # Return indices sorted by weighted scores descending
        return np.argsort(weighted_scores)[::-1].tolist()
