"""Reconfiguration-aware placement and partitioning for the Foreman.

Pure layers (no I/O, clocks or globals; reusable by the trace-driven simulator):
``types``, ``cost_model``, ``planner``, ``policies``, ``controller``,
``routes`` (selection logic), ``prefetch``, ``registry.plan_evictions``.

Stateful helpers used by the live Foreman: ``telemetry``, ``profiles``,
``registry.ShardRegistry``, ``problems``, ``eventlog``.
"""

from .config import RunConfig
from .flags import FeatureFlags

__all__ = ["FeatureFlags", "RunConfig"]
