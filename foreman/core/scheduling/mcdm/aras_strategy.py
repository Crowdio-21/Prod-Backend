"""
ARAS (Additive Ratio Assessment) Strategy
"""

import numpy as np
from .base_strategy import AllocationStrategy


class ARASStrategy(AllocationStrategy):
    """
    ARAS - Additive Ratio Assessment
    Ranks devices by comparing them to an optimal alternative (A0).
    """

    def rank_devices(self, decision_matrix, criteria_types):
        """
        Rank devices using ARAS algorithm with Dynamic Weighting
        """
        rows, cols = decision_matrix.shape
        if rows == 0: return []

        # --- GET WEIGHTS (Dynamic or Static) ---
        active_weights = self._get_active_weights(decision_matrix)

        # 1. Determine Optimal Alternative (A0)
        a0 = np.zeros(cols)
        for j in range(cols):
            if criteria_types[j] == 1:  # Benefit criterion
                a0[j] = np.max(decision_matrix[:, j])
            else:  # Cost criterion
                a0[j] = np.min(decision_matrix[:, j])

        # Append A0 to matrix for normalization
        extended_matrix = np.vstack([a0, decision_matrix])

        # 2. Two-stage Normalization
        norm_matrix = np.zeros_like(extended_matrix, dtype=float)
        for j in range(cols):
            if criteria_types[j] == 1:  # Benefit
                norm_matrix[:, j] = extended_matrix[:, j]
            else:  # Cost -> Inverse
                # Handle zero values for cost
                safe_col = np.where(
                    extended_matrix[:, j] == 0, 1e-9, extended_matrix[:, j]
                )
                norm_matrix[:, j] = 1.0 / safe_col

        # Standardize to sum to 1 column-wise
        col_sums = np.sum(norm_matrix, axis=0)
        final_norm = norm_matrix / (col_sums + 1e-9)

            # Log normalized values for each worker and criterion
            import logging
            logger = logging.getLogger("mcdm_scheduler")
            logger.debug("Normalized values (ARAS):")
            for i in range(final_norm.shape[0]):
                logger.debug(f"  Worker {i}:")
                for j in range(final_norm.shape[1]):
                    logger.debug(f"    Criterion {j}: {final_norm[i, j]:.4f}")

        # 3. Weighted Matrix & Optimality Function (Si)
        # USES ACTIVE WEIGHTS HERE
        weighted_matrix = final_norm * active_weights
        s_values = np.sum(weighted_matrix, axis=1)

        # 4. Utility Degree (Ki)
        s0 = s_values[0] if s_values[0] != 0 else 1.0
        k_values = s_values[1:] / s0

        self._last_scores = k_values

        return np.argsort(k_values)[::-1].tolist()