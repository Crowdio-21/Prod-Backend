"""
EDAS (Evaluation based on Distance from Average Solution) Strategy

Ported from TEMP_ALGO_FOLDER/foreman_server/strategies/edas.py
"""

import numpy as np
from .base_strategy import AllocationStrategy


class EDASStrategy(AllocationStrategy):
    """
    EDAS - Evaluation based on Distance from Average Solution

    Ranks devices based on their positive and negative distances
    from the average solution for each criterion.
    """

    def rank_devices(self, decision_matrix, criteria_types):
        """
        Rank devices using EDAS algorithm

        Args:
            decision_matrix: (n_devices x m_criteria) numpy array
            criteria_types: List of +1 (benefit) or -1 (cost)

        Returns:
            List of device indices sorted by rank (best to worst)
        """
        # 1. Calculate Average Solution (AV)
        av = np.mean(decision_matrix, axis=0)

        rows, cols = decision_matrix.shape
        pda = np.zeros((rows, cols))
        nda = np.zeros((rows, cols))

        # 2. Calculate PDA (Positive Distance from Average) and NDA (Negative Distance from Average)
        for j in range(cols):
            if criteria_types[j] == 1:  # Benefit criterion
                # Avoid division by zero if AV is 0
                denom = av[j] if av[j] != 0 else 1e-9
                pda[:, j] = np.maximum(0, decision_matrix[:, j] - av[j]) / denom
                nda[:, j] = np.maximum(0, av[j] - decision_matrix[:, j]) / denom
            else:  # Cost criterion
                denom = av[j] if av[j] != 0 else 1e-9
                pda[:, j] = np.maximum(0, av[j] - decision_matrix[:, j]) / denom
                nda[:, j] = np.maximum(0, decision_matrix[:, j] - av[j]) / denom

        # 3. Calculate SP and SN (Weighted Sums)
        sp = np.sum(pda * self.weights, axis=1)
        sn = np.sum(nda * self.weights, axis=1)

        # 4. Normalize (NSP, NSN)
        max_sp = np.max(sp) if np.max(sp) != 0 else 1
        max_sn = np.max(sn) if np.max(sn) != 0 else 1

        nsp = sp / max_sp
        nsn = 1 - (sn / max_sn)

        # 5. Appraisal Score (AS)
        as_score = 0.5 * (nsp + nsn)

        # Return indices sorted by AS descending
        return np.argsort(as_score)[::-1].tolist()
