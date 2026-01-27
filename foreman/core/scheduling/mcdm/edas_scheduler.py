"""
EDAS Scheduler - TaskScheduler implementation using EDAS strategy
"""

from typing import List
from .base_mcdm import BaseMCDMScheduler
from .edas_strategy import EDASStrategy


class EDASScheduler(BaseMCDMScheduler):
    """
    EDAS (Evaluation based on Distance from Average Solution) Scheduler

    Ranks workers based on their positive and negative distances from
    the average solution for each criterion.

    Best for: Identifying workers that deviate positively from average performance
    """

    def __init__(
        self,
        criteria_weights: List[float],
        criteria_names: List[str],
        criteria_types: List[int],
    ):
        """
        Initialize EDAS scheduler

        Args:
            criteria_weights: Weights for each criterion (should sum to 1.0)
            criteria_names: Worker attribute names to use
            criteria_types: +1 for benefit, -1 for cost
        """
        strategy = EDASStrategy(criteria_weights=None)  # Will be set by parent
        super().__init__(strategy, criteria_weights, criteria_names, criteria_types)
