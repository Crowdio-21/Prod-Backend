from .scheduler_interface import TaskScheduler
from .fifo_scheduler import FIFOScheduler
from .round_robin_scheduler import RoundRobinScheduler
from .performance_scheduler import PerformanceBasedScheduler
from .least_loaded_scheduler import LeastLoadedScheduler
from .priority_scheduler import PriorityScheduler

# MCDM schedulers
from .mcdm import ARASScheduler, EDASScheduler, MABACScheduler, WRRScheduler


async def _load_mcdm_config(algorithm_name: str) -> dict:
    """
    Load MCDM scheduler configuration from database

    Args:
        algorithm_name: Name of MCDM algorithm (aras, edas, mabac, wrr)

    Returns:
        Configuration dictionary with criteria_weights, names, types
    """
    import json
    from foreman.db.base import async_session
    from foreman.db.models import SchedulerConfigModel
    from sqlalchemy import select

    async with async_session() as session:
        result = await session.execute(
            select(SchedulerConfigModel).filter_by(algorithm_name=algorithm_name)
        )
        config = result.scalar_one_or_none()

        if not config:
            # Return default configuration if not found
            print(f"Warning: No config found for {algorithm_name}, using defaults")
            return _get_default_mcdm_config(algorithm_name)

        return {
            "criteria_weights": json.loads(config.criteria_weights),
            "criteria_names": json.loads(config.criteria_names),
            "criteria_types": json.loads(config.criteria_types),
        }


def _get_default_mcdm_config(algorithm_name: str) -> dict:
    """Get default MCDM configuration"""
    defaults = {
        "aras": {
            "criteria_weights": [0.28, 0.23, 0.18, 0.18, 0.13],
            "criteria_names": [
                "cpu_cores",
                "ram_available_mb",
                "battery_level",
                "success_rate",
                "avg_task_duration_sec",
            ],
            "criteria_types": [1, 1, 1, 1, -1],
        },
        "edas": {
            "criteria_weights": [0.34, 0.29, 0.24, 0.13],
            "criteria_names": [
                "cpu_cores",
                "ram_available_mb",
                "success_rate",
                "avg_task_duration_sec",
            ],
            "criteria_types": [1, 1, 1, -1],
        },
        "mabac": {
            "criteria_weights": [0.35, 0.30, 0.20, 0.15],
            "criteria_names": [
                "cpu_cores",
                "cpu_frequency_mhz",
                "ram_available_mb",
                "success_rate",
            ],
            "criteria_types": [1, 1, 1, 1],
        },
        "wrr": {
            "criteria_weights": [ 0, 0.8, 0.2],
            "criteria_names": [ 
                "ram_total_mb",
                "cpu_frequency_mhz",
                "success_rate",
            ],
            "criteria_types": [1, 1, 1],
        },
    }
    return defaults.get(algorithm_name, defaults["aras"])


# Scheduler factory
async def create_scheduler_async(
    scheduler_type: str = "fifo", use_dynamic_weighting: bool = False
) -> TaskScheduler:
    """
    Factory function to create scheduler instances (async version for MCDM)

    Args:
        scheduler_type: Type of scheduler
                       Simple: "fifo", "round_robin", "performance", "least_loaded", "priority"
                       MCDM: "aras", "edas", "mabac", "wrr"
        use_dynamic_weighting: For MCDM schedulers, whether to use Shannon Entropy for dynamic weighting (default: False)

    Returns:
        TaskScheduler instance
    """
    scheduler_type = scheduler_type.lower()

    # Simple schedulers (no config needed)
    simple_schedulers = {
        "fifo": FIFOScheduler,
        "round_robin": RoundRobinScheduler,
        "performance": PerformanceBasedScheduler,
        "least_loaded": LeastLoadedScheduler,
        "priority": PriorityScheduler,
    }

    if scheduler_type in simple_schedulers:
        return simple_schedulers[scheduler_type]()

    # MCDM schedulers (need config)
    mcdm_schedulers = {
        "aras": ARASScheduler,
        "edas": EDASScheduler,
        "mabac": MABACScheduler,
        "wrr": WRRScheduler,
    }

    if scheduler_type in mcdm_schedulers:
        # Load configuration
        config = await _load_mcdm_config(scheduler_type)
        return mcdm_schedulers[scheduler_type](
            criteria_weights=config["criteria_weights"],
            criteria_names=config["criteria_names"],
            criteria_types=config["criteria_types"],
            use_dynamic_weighting=use_dynamic_weighting,
        )

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_scheduler(
    scheduler_type: str = "fifo", use_dynamic_weighting: bool = False
) -> TaskScheduler:
    """
    Synchronous factory function for backward compatibility

    Note: For MCDM schedulers, use create_scheduler_async() or provide config manually

    Args:
        scheduler_type: Type of scheduler (simple schedulers only)
        use_dynamic_weighting: For MCDM schedulers, whether to use Shannon Entropy for dynamic weighting (default: False)

    Returns:
        TaskScheduler instance
    """
    schedulers = {
        "fifo": FIFOScheduler,
        "round_robin": RoundRobinScheduler,
        "performance": PerformanceBasedScheduler,
        "least_loaded": LeastLoadedScheduler,
        "priority": PriorityScheduler,
    }

    scheduler_class = schedulers.get(scheduler_type.lower())
    if not scheduler_class:
        # Check if MCDM scheduler requested
        if scheduler_type.lower() in ["aras", "edas", "mabac", "wrr"]:
            # Use default config
            config = _get_default_mcdm_config(scheduler_type.lower())
            mcdm_schedulers = {
                "aras": ARASScheduler,
                "edas": EDASScheduler,
                "mabac": MABACScheduler,
                "wrr": WRRScheduler,
            }
            return mcdm_schedulers[scheduler_type.lower()](
                criteria_weights=config["criteria_weights"],
                criteria_names=config["criteria_names"],
                criteria_types=config["criteria_types"],
                use_dynamic_weighting=use_dynamic_weighting,
            )
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler_class()
