"""
ARAS (Additive Ratio Assessment) Strategy

Ported from TEMP_ALGO_FOLDER/foreman_server/strategies/aras.py
"""

import numpy as np
from .base_strategy import AllocationStrategy


class ARASStrategy(AllocationStrategy):
    """
    ARAS - Additive Ratio Assessment

    Ranks devices by comparing them to an optimal alternative (A0).
    A0 is constructed from the best values of each criterion.
    """

    def rank_devices(self, decision_matrix, criteria_types):
        """
        Rank devices using ARAS algorithm

        Args:
            decision_matrix: (n_devices x m_criteria) numpy array
            criteria_types: List of +1 (benefit) or -1 (cost)

        Returns:
            List of device indices sorted by rank (best to worst)
        """
        rows, cols = decision_matrix.shape

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

        # 3. Weighted Matrix & Optimality Function (Si)
        weighted_matrix = final_norm * self.weights
        s_values = np.sum(weighted_matrix, axis=1)

        # 4. Utility Degree (Ki)
        s0 = s_values[0] if s_values[0] != 0 else 1.0
        k_values = s_values[1:] / s0

        # Return indices sorted by utility degree (descending)
        return np.argsort(k_values)[::-1].tolist()
