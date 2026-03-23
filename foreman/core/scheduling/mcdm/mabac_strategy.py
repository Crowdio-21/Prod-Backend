"""
MABAC (Multi-Attributive Border Approximation area Comparison) Strategy
"""

import numpy as np
from .base_strategy import AllocationStrategy


class MABACStrategy(AllocationStrategy):
    """
    MABAC - Multi-Attributive Border Approximation area Comparison
    Ranks devices based on their distance from the Border Approximation Area.
    """

    def rank_devices(self, decision_matrix, criteria_types):
        # Impute missing values for fairness
        decision_matrix = self._impute_missing(decision_matrix, criteria_types)
        """
        Rank devices using MABAC algorithm with Dynamic Weighting
        """
        rows, cols = decision_matrix.shape
        if rows == 0: return []

        # --- GET WEIGHTS (Dynamic or Static) ---
        active_weights = self._get_active_weights(decision_matrix)

        # 1. Max-Min Normalization
        norm_matrix = np.zeros((rows, cols))
        for j in range(cols):
            max_val = np.max(decision_matrix[:, j])
            min_val = np.min(decision_matrix[:, j])
            dist = max_val - min_val if max_val != min_val else 1.0

            if criteria_types[j] == 1:  # Benefit
                norm_matrix[:, j] = (decision_matrix[:, j] - min_val) / dist
            else:  # Cost
                norm_matrix[:, j] = (decision_matrix[:, j] - max_val) / (
                    min_val - max_val or -1.0
                )

         # Log normalized values for each worker and criterion
        import logging
        logger = logging.getLogger("mcdm_scheduler")
        logger.debug("Normalized values (MABAC):")
        for i in range(rows):
            logger.debug(f"  Worker {i}:")
            for j in range(cols):
                logger.debug(f"    Criterion {j}: {norm_matrix[i, j]:.4f}")

        # 2. Weighted Matrix (V)
        # MABAC specific: V = w * (n + 1)
        # USES ACTIVE WEIGHTS HERE
        weighted_matrix = active_weights * (norm_matrix + 1.0)

        # 3. Border Approximation Area (BAA) - Geometric Mean
        baa = np.prod(weighted_matrix, axis=0) ** (1.0 / rows)

        # 4. Distance Calculation (Q)
        q_matrix = weighted_matrix - baa

        # 5. Final Score (Sum of Q)
        final_scores = np.sum(q_matrix, axis=1)

        self._last_scores = final_scores

        return np.argsort(final_scores)[::-1].tolist()