"""
MABAC (Multi-Attributive Border Approximation area Comparison) Strategy

Ported from TEMP_ALGO_FOLDER/foreman_server/strategies/mabac.py
"""

import numpy as np
from .base_strategy import AllocationStrategy


class MABACStrategy(AllocationStrategy):
    """
    MABAC - Multi-Attributive Border Approximation area Comparison

    Ranks devices based on their distance from the Border Approximation Area (BAA).
    The BAA represents the geometric mean of weighted normalized values.
    """

    def rank_devices(self, decision_matrix, criteria_types):
        """
        Rank devices using MABAC algorithm

        Args:
            decision_matrix: (n_devices x m_criteria) numpy array
            criteria_types: List of +1 (benefit) or -1 (cost)

        Returns:
            List of device indices sorted by rank (best to worst)
        """
        rows, cols = decision_matrix.shape

        # 1. Max-Min Normalization
        norm_matrix = np.zeros((rows, cols))
        for j in range(cols):
            max_val = np.max(decision_matrix[:, j])
            min_val = np.min(decision_matrix[:, j])
            dist = max_val - min_val if max_val != min_val else 1.0

            if criteria_types[j] == 1:  # Benefit criterion
                norm_matrix[:, j] = (decision_matrix[:, j] - min_val) / dist
            else:  # Cost criterion
                norm_matrix[:, j] = (decision_matrix[:, j] - max_val) / (
                    min_val - max_val or -1.0
                )

        # 2. Weighted Matrix (V)
        # MABAC specific: V = w * (n + 1)
        weighted_matrix = self.weights * (norm_matrix + 1.0)

        # 3. Border Approximation Area (BAA) - Geometric Mean
        # g_j = product(v_ij)^(1/m)
        baa = np.prod(weighted_matrix, axis=0) ** (1.0 / rows)

        # 4. Distance Calculation (Q)
        q_matrix = weighted_matrix - baa

        # 5. Final Score (Sum of Q)
        final_scores = np.sum(q_matrix, axis=1)
        print(f"[MABAC STRATEGY]: Final Scores: {final_scores}")

        # Return indices sorted by final scores descending
        return np.argsort(final_scores)[::-1].tolist()
