"""
EDAS (Evaluation based on Distance from Average Solution) Strategy
"""

import numpy as np
from .base_strategy import AllocationStrategy


class EDASStrategy(AllocationStrategy):
    """
    EDAS - Evaluation based on Distance from Average Solution
    Ranks devices based on their positive and negative distances from average.
    """

    def rank_devices(self, decision_matrix, criteria_types):
        """
        Rank devices using EDAS algorithm with Dynamic Weighting
        """
        rows, cols = decision_matrix.shape
        if rows == 0: return []

        # --- GET WEIGHTS (Dynamic or Static) ---
        active_weights = self._get_active_weights(decision_matrix)

        # 1. Calculate Average Solution (AV)
        av = np.mean(decision_matrix, axis=0)

        pda = np.zeros((rows, cols))
        nda = np.zeros((rows, cols))

        # 2. Calculate PDA and NDA
        for j in range(cols):
            denom = av[j] if av[j] != 0 else 1e-9
            
            if criteria_types[j] == 1:  # Benefit criterion
                pda[:, j] = np.maximum(0, decision_matrix[:, j] - av[j]) / denom
                nda[:, j] = np.maximum(0, av[j] - decision_matrix[:, j]) / denom
            else:  # Cost criterion
                pda[:, j] = np.maximum(0, av[j] - decision_matrix[:, j]) / denom
                nda[:, j] = np.maximum(0, decision_matrix[:, j] - av[j]) / denom

        # 3. Calculate SP and SN (Weighted Sums)
        # USES ACTIVE WEIGHTS HERE
        sp = np.sum(pda * active_weights, axis=1)
        sn = np.sum(nda * active_weights, axis=1)

        # 4. Normalize (NSP, NSN)
        max_sp = np.max(sp) if np.max(sp) != 0 else 1
        max_sn = np.max(sn) if np.max(sn) != 0 else 1

        nsp = sp / max_sp
        nsn = 1 - (sn / max_sn)

        # 5. Appraisal Score (AS)
        as_score = 0.5 * (nsp + nsn)

        self._last_scores = as_score

        return np.argsort(as_score)[::-1].tolist()