"""
WRR Scheduler - TaskScheduler implementation using WRR strategy
"""

from typing import List
from .base_mcdm import BaseMCDMScheduler
from .wrr_strategy import WRRStrategy


class WRRScheduler(BaseMCDMScheduler):
    """
    WRR (Weighted Round Robin) Scheduler

    Simple weighted sum approach for worker ranking.
    Lightweight alternative to complex MCDM algorithms.

    Best for: Low-latency scheduling with basic multi-criteria support
    """

    def __init__(
        self,
        criteria_weights: List[float],
        criteria_names: List[str],
        criteria_types: List[int],
    ):
        """
        Initialize WRR scheduler

        Args:
            criteria_weights: Weights for each criterion (should sum to 1.0)
            criteria_names: Worker attribute names to use
            criteria_types: +1 for benefit, -1 for cost
        """
        strategy = WRRStrategy(criteria_weights=None)  # Will be set by parent
        super().__init__(strategy, criteria_weights, criteria_names, criteria_types)
