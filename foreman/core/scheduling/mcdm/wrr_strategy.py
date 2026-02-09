"""
WRR (Weighted Round Robin) Strategy
"""

import numpy as np
from .base_strategy import AllocationStrategy


class WRRStrategy(AllocationStrategy):
    """
    WRR - Weighted Round Robin
    Simplistic baseline: rank simply based on weighted sum of benefit criteria.
    """

    def rank_devices(self, decision_matrix, criteria_types):
        """
        Rank devices using simple weighted sum with Dynamic Weighting
        """
        rows, cols = decision_matrix.shape
        if rows == 0: return []

        # --- GET WEIGHTS (Dynamic or Static) ---
        active_weights = self._get_active_weights(decision_matrix)

        # Simple approach: weighted sum considering benefit vs cost
        weighted_scores = np.zeros(rows)

        for j in range(cols):
            if criteria_types[j] == 1:  # Benefit - higher is better
                weighted_scores += decision_matrix[:, j] * active_weights[j]
            else:  # Cost - lower is better, so invert
                max_val = np.max(decision_matrix[:, j])
                if max_val > 0:
                    inverted = max_val - decision_matrix[:, j]
                    weighted_scores += inverted * active_weights[j]
                # If max is 0 (all zeros), cost contribution is 0

        self._last_scores = weighted_scores

        return np.argsort(weighted_scores)[::-1].tolist()