"""
CROWDio Evaluation Framework

Comprehensive evaluation metrics and experimental scenarios for
distributed edge computing performance analysis.

Metrics:
- Task completion time
- Throughput (tasks/second)
- Load balancing efficiency
- Energy consumption
- Communication overhead
- Failure recovery time

Experimental Scenarios:
- Scalability with increasing devices
- Device heterogeneity
- Mobile failures and disconnections
- Energy-constrained devices
"""

from .energy_tracker import EnergyTracker, EnergyProfile, DeviceEnergyState
from .load_balancing import LoadBalancingAnalyzer, LoadDistribution
from .failure_recovery import FailureRecoveryTracker, RecoveryMetrics
from .communication_tracker import CommunicationTracker, CommunicationMetrics
from .runner import EvaluationRunner, ExperimentConfig
from .visualization import EvaluationVisualizer, ChartConfig, quick_generate_report

# The real-metrics modules import foreman.db.base, which reads CROWDIO_DATABASE_URL once at import.
# Load them on first use so the device simulator can point the database elsewhere first.
_LAZY = {
    **dict.fromkeys(
        ("RealMetricsCollector", "RealTaskMetrics", "RealJobMetrics",
         "RealWorkerMetrics", "RealSystemMetrics", "collect_real_evaluation"),
        ".real_metrics_collector"),
    **dict.fromkeys(
        ("RealEvaluationRunner", "EvaluationConfig", "WorkloadConfig", "WorkloadResult",
         "run_quick_real_evaluation", "run_full_real_evaluation"),
        ".real_runner"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Real Metrics Collection
    "RealMetricsCollector",
    "RealTaskMetrics",
    "RealJobMetrics",
    "RealWorkerMetrics",
    "RealSystemMetrics",
    "collect_real_evaluation",
    # Energy Tracking
    "EnergyTracker",
    "EnergyProfile",
    "DeviceEnergyState",
    # Load Balancing
    "LoadBalancingAnalyzer",
    "LoadDistribution",
    # Failure Recovery
    "FailureRecoveryTracker",
    "RecoveryMetrics",
    # Communication
    "CommunicationTracker",
    "CommunicationMetrics",
    # Simulated Runner
    "EvaluationRunner",
    # Real Runner
    "RealEvaluationRunner",
    "EvaluationConfig",
    "WorkloadConfig",
    "WorkloadResult",
    "run_quick_real_evaluation",
    "run_full_real_evaluation",
    "ExperimentConfig",
    # Visualization
    "EvaluationVisualizer",
    "ChartConfig",
    "quick_generate_report",
]
