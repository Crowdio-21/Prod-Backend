"""
ARAS Scheduler - TaskScheduler implementation using ARAS strategy
"""

from typing import List
from .base_mcdm import BaseMCDMScheduler
from .aras_strategy import ARASStrategy


class ARASScheduler(BaseMCDMScheduler):
    """
    ARAS (Additive Ratio Assessment) Scheduler

    Ranks workers by comparing them to an optimal alternative constructed
    from the best values of each criterion.

    Best for: Balanced workload distribution considering multiple factors
    """

    def __init__(
        self,
        criteria_weights: List[float],
        criteria_names: List[str],
        criteria_types: List[int],
    ):
        """
        Initialize ARAS scheduler

        Args:
            criteria_weights: Weights for each criterion (should sum to 1.0)
            criteria_names: Worker attribute names to use
            criteria_types: +1 for benefit, -1 for cost
        """
        strategy = ARASStrategy(criteria_weights=None)  # Will be set by parent
        super().__init__(strategy, criteria_weights, criteria_names, criteria_types)
