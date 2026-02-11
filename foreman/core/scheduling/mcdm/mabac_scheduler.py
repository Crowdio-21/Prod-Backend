"""
MABAC Scheduler - TaskScheduler implementation using MABAC strategy
"""

from typing import List
from .base_mcdm import BaseMCDMScheduler
from .mabac_strategy import MABACStrategy


class MABACScheduler(BaseMCDMScheduler):
    """
    MABAC (Multi-Attributive Border Approximation area Comparison) Scheduler

    Ranks workers based on their distance from the Border Approximation Area (BAA),
    which represents the geometric mean of weighted normalized values.

    Best for: High-performance computing workloads prioritizing top performers
    """

    def __init__(
        self,
        criteria_weights: List[float],
        criteria_names: List[str],
        criteria_types: List[int],
        use_dynamic_weighting: bool = True,
    ):
        """
        Initialize MABAC scheduler

        Args:
            criteria_weights: Weights for each criterion (should sum to 1.0)
            criteria_names: Worker attribute names to use
            criteria_types: +1 for benefit, -1 for cost
            use_dynamic_weighting: Whether to use Shannon Entropy for dynamic weighting (default: True)
        """
        strategy = MABACStrategy(criteria_weights=None, use_dynamic_weighting=use_dynamic_weighting)  # Will be set by parent
        super().__init__(strategy, criteria_weights, criteria_names, criteria_types)
