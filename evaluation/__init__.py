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

from .metrics_collector import MetricsCollector, TaskMetrics, JobMetrics, SystemMetrics
from .energy_tracker import EnergyTracker, EnergyProfile, DeviceEnergyState
from .load_balancing import LoadBalancingAnalyzer, LoadDistribution
from .failure_recovery import FailureRecoveryTracker, RecoveryMetrics
from .communication_tracker import CommunicationTracker, CommunicationMetrics
from .runner import EvaluationRunner, ExperimentConfig
from .visualization import EvaluationVisualizer, ChartConfig, quick_generate_report
from .real_metrics_collector import (
    RealMetricsCollector, RealTaskMetrics, RealJobMetrics, 
    RealWorkerMetrics, RealSystemMetrics, collect_real_evaluation
)
from .real_runner import (
    RealEvaluationRunner, EvaluationConfig, WorkloadConfig, WorkloadResult,
    run_quick_real_evaluation, run_full_real_evaluation
)

__all__ = [
    # Metrics Collection (Simulated)
    "MetricsCollector",
    "TaskMetrics",
    "JobMetrics", 
    "SystemMetrics",
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
