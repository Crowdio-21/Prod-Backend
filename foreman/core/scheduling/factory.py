import os

from .scheduler_interface import TaskScheduler
from .fifo_scheduler import FIFOScheduler
from .round_robin_scheduler import RoundRobinScheduler
from .performance_scheduler import PerformanceBasedScheduler
from .least_loaded_scheduler import LeastLoadedScheduler
from .priority_scheduler import PriorityScheduler
from .bandwidth_scheduler import BandwidthAwareScheduler

SCHEDULERS = {
    "fifo": FIFOScheduler,
    "round_robin": RoundRobinScheduler,
    "performance": PerformanceBasedScheduler,
    "least_loaded": LeastLoadedScheduler,
    "priority": PriorityScheduler,
    "bandwidth_aware": BandwidthAwareScheduler,
}


def create_scheduler(scheduler_type: str = "fifo", **options) -> TaskScheduler:
    """
    Create a scheduler by name.

    Args:
        scheduler_type: "fifo", "round_robin", "performance", "least_loaded", "priority"
                        or "bandwidth_aware"
        options: passed to the scheduler's constructor (only "bandwidth_aware" takes any:
                 estimator, shared_capacity_bps, result_ratio, default_compute_s, ...).
                 For "bandwidth_aware", CROWDIO_SHARED_CAPACITY_MBPS sets the capacity of a
                 shared medium that concurrent transfers split (unset: links are independent).

    Returns:
        TaskScheduler instance

    Raises:
        ValueError: if the name is not a known scheduler
    """
    scheduler_class = SCHEDULERS.get(str(scheduler_type).lower())
    if scheduler_class is None:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type!r}. Available: {', '.join(SCHEDULERS)}"
        )
    if scheduler_class is BandwidthAwareScheduler:
        capacity_mbps = os.getenv("CROWDIO_SHARED_CAPACITY_MBPS")
        if capacity_mbps and "shared_capacity_bps" not in options:
            options["shared_capacity_bps"] = float(capacity_mbps) * 1e6 / 8
    elif options:
        raise ValueError(f"The {scheduler_type!r} scheduler takes no options (got {sorted(options)})")
    return scheduler_class(**options)
