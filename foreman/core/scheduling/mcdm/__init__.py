"""
MCDM (Multi-Criteria Decision Making) Schedulers

This package contains advanced task allocation algorithms based on MCDM methods:
- ARAS: Additive Ratio Assessment
- EDAS: Evaluation based on Distance from Average Solution
- MABAC: Multi-Attributive Border Approximation area Comparison
- WRR: Weighted Round Robin

These schedulers use multiple criteria (CPU, RAM, battery, network, etc.)
to make intelligent worker selection decisions.
"""

from .aras_scheduler import ARASScheduler
from .edas_scheduler import EDASScheduler
from .mabac_scheduler import MABACScheduler
from .wrr_scheduler import WRRScheduler
from .config_manager import SchedulerConfigManager

__all__ = [
    "ARASScheduler",
    "EDASScheduler",
    "MABACScheduler",
    "WRRScheduler",
    "SchedulerConfigManager",
]
