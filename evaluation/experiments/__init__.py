"""
CROWDio Evaluation Experiments Package

Contains experimental scenarios for evaluating the distributed
edge computing system:

- Scalability: Performance with increasing devices
- Heterogeneity: Mixed device capabilities
- Failure Simulation: Worker failures and disconnections
- Energy Constraints: Battery-limited scenarios
"""

from .base import (
    Experiment,
    ExperimentResult,
    ExperimentConfig,
    WorkloadConfig,
    WorkerSimulationConfig,
)
from .scalability import ScalabilityExperiment
from .heterogeneity import HeterogeneityExperiment
from .failure_simulation import FailureSimulationExperiment
from .energy_constraints import EnergyConstraintExperiment

__all__ = [
    "Experiment",
    "ExperimentResult",
    "ExperimentConfig",
    "WorkloadConfig",
    "WorkerSimulationConfig",
    "ScalabilityExperiment",
    "HeterogeneityExperiment",
    "FailureSimulationExperiment",
    "EnergyConstraintExperiment",
]
