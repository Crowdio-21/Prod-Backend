"""
Evaluation Runner Module

Main orchestrator for running CROWDio evaluation experiments and
generating comprehensive reports.

Provides:
- Experiment orchestration
- Multi-experiment comparisons
- Report generation
- Result aggregation
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
import shutil

from .experiments.base import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    WorkloadConfig,
    WorkloadType,
)
from .experiments.scalability import ScalabilityExperiment, ScalabilityConfig
from .experiments.heterogeneity import HeterogeneityExperiment, HeterogeneityConfig
from .experiments.failure_simulation import FailureSimulationExperiment, FailureSimulationConfig
from .experiments.energy_constraints import EnergyConstraintExperiment, EnergyConstraintConfig


@dataclass
class EvaluationPlan:
    """Plan for a complete evaluation run"""
    name: str
    description: str = ""
    
    # Experiments to run
    run_scalability: bool = True
    run_heterogeneity: bool = True
    run_failure_simulation: bool = True
    run_energy_constraints: bool = True
    
    # Custom experiment configurations
    scalability_config: Optional[ScalabilityConfig] = None
    heterogeneity_config: Optional[HeterogeneityConfig] = None
    failure_config: Optional[FailureSimulationConfig] = None
    energy_config: Optional[EnergyConstraintConfig] = None
    
    # Global settings
    num_iterations: int = 3
    output_dir: str = "./evaluation_results"
    
    # Random seed for reproducibility
    random_seed: Optional[int] = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "experiments": {
                "scalability": self.run_scalability,
                "heterogeneity": self.run_heterogeneity,
                "failure_simulation": self.run_failure_simulation,
                "energy_constraints": self.run_energy_constraints,
            },
            "num_iterations": self.num_iterations,
            "random_seed": self.random_seed,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    plan: EvaluationPlan
    
    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    # Results
    experiment_results: Dict[str, ExperimentResult] = field(default_factory=dict)
    
    # Analysis
    summary: Dict[str, Any] = field(default_factory=dict)
    comparisons: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def save(self, filepath: str) -> None:
        """Save report to JSON file"""
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "plan": self.plan.to_dict(),
                "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
                "completed_at": datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
                "duration_seconds": self.completed_at - self.started_at if self.completed_at else None,
            },
            "experiment_results": {
                name: result.to_dict()
                for name, result in self.experiment_results.items()
            },
            "summary": self.summary,
            "comparisons": self.comparisons,
            "recommendations": self.recommendations,
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"EvaluationReport: Saved to {filepath}")


class EvaluationRunner:
    """
    Main class for running evaluation experiments.
    
    Orchestrates multiple experiments, collects results, and generates
    comprehensive reports.
    """
    
    def __init__(self, plan: EvaluationPlan):
        """
        Initialize evaluation runner.
        
        Args:
            plan: Evaluation plan specifying experiments to run
        """
        self.plan = plan
        self.report: Optional[EvaluationReport] = None
        
        # Experiments to run
        self.experiments: Dict[str, Experiment] = {}
        
        # Create output directory
        os.makedirs(plan.output_dir, exist_ok=True)
    
    def _create_default_scalability_config(self) -> ScalabilityConfig:
        """Create default scalability experiment config"""
        return ScalabilityConfig(
            name="Scalability Experiment",
            description="Test system performance with increasing workers",
            num_iterations=self.plan.num_iterations,
            random_seed=self.plan.random_seed,
            output_dir=self.plan.output_dir,
            worker_counts=[1, 2, 4, 8, 16],
            num_workers=8,
            workload=WorkloadConfig(
                workload_type=WorkloadType.CPU_LIGHT,
                num_tasks=100,
                task_duration_mean=1.0,
            ),
        )
    
    def _create_default_heterogeneity_config(self) -> HeterogeneityConfig:
        """Create default heterogeneity experiment config"""
        return HeterogeneityConfig(
            name="Heterogeneity Experiment",
            description="Test system with diverse device capabilities",
            num_iterations=self.plan.num_iterations,
            random_seed=self.plan.random_seed,
            output_dir=self.plan.output_dir,
            num_workers=10,
            workload=WorkloadConfig(
                workload_type=WorkloadType.MIXED,
                num_tasks=100,
                task_duration_mean=1.0,
            ),
        )
    
    def _create_default_failure_config(self) -> FailureSimulationConfig:
        """Create default failure simulation config"""
        return FailureSimulationConfig(
            name="Failure Simulation Experiment",
            description="Test system resilience to failures",
            num_iterations=self.plan.num_iterations,
            random_seed=self.plan.random_seed,
            output_dir=self.plan.output_dir,
            num_workers=8,
            workload=WorkloadConfig(
                workload_type=WorkloadType.CPU_LIGHT,
                num_tasks=100,
                task_duration_mean=1.0,
                enable_checkpointing=True,
            ),
        )
    
    def _create_default_energy_config(self) -> EnergyConstraintConfig:
        """Create default energy constraint config"""
        return EnergyConstraintConfig(
            name="Energy Constraint Experiment",
            description="Test system with energy-constrained devices",
            num_iterations=self.plan.num_iterations,
            random_seed=self.plan.random_seed,
            output_dir=self.plan.output_dir,
            num_workers=10,
            workload=WorkloadConfig(
                workload_type=WorkloadType.CPU_LIGHT,
                num_tasks=100,
                task_duration_mean=1.0,
            ),
        )
    
    async def setup(self) -> None:
        """Set up experiments based on plan"""
        print(f"EvaluationRunner: Setting up evaluation '{self.plan.name}'")
        
        if self.plan.run_scalability:
            config = self.plan.scalability_config or self._create_default_scalability_config()
            self.experiments["scalability"] = ScalabilityExperiment(config)
        
        if self.plan.run_heterogeneity:
            config = self.plan.heterogeneity_config or self._create_default_heterogeneity_config()
            self.experiments["heterogeneity"] = HeterogeneityExperiment(config)
        
        if self.plan.run_failure_simulation:
            config = self.plan.failure_config or self._create_default_failure_config()
            self.experiments["failure_simulation"] = FailureSimulationExperiment(config)
        
        if self.plan.run_energy_constraints:
            config = self.plan.energy_config or self._create_default_energy_config()
            self.experiments["energy_constraints"] = EnergyConstraintExperiment(config)
        
        print(f"EvaluationRunner: Will run {len(self.experiments)} experiments")
    
    async def run(self) -> EvaluationReport:
        """
        Run all planned experiments.
        
        Returns:
            Complete evaluation report
        """
        self.report = EvaluationReport(plan=self.plan)
        
        await self.setup()
        
        total_experiments = len(self.experiments)
        completed = 0
        
        for name, experiment in self.experiments.items():
            print(f"\n{'='*60}")
            print(f"Running experiment: {name} ({completed + 1}/{total_experiments})")
            print(f"{'='*60}")
            
            try:
                result = await experiment.run()
                self.report.experiment_results[name] = result
                
                # Save individual result
                result_path = os.path.join(
                    self.plan.output_dir,
                    f"{name}_result.json"
                )
                result.save(result_path)
                
                completed += 1
                print(f"Experiment '{name}' completed successfully")
                
            except Exception as e:
                print(f"Experiment '{name}' failed: {e}")
                raise
        
        # Generate analysis
        await self._generate_analysis()
        
        self.report.completed_at = time.time()
        
        # Save complete report
        report_path = os.path.join(
            self.plan.output_dir,
            f"evaluation_report_{int(time.time())}.json"
        )
        self.report.save(report_path)
        
        # Generate summary
        self._print_summary()
        
        return self.report
    
    async def _generate_analysis(self) -> None:
        """Generate analysis and comparisons from experiment results"""
        if not self.report:
            return
        
        summary = {}
        comparisons = {}
        recommendations = []
        
        # Scalability analysis
        if "scalability" in self.report.experiment_results:
            exp = self.experiments.get("scalability")
            if isinstance(exp, ScalabilityExperiment):
                scalability_analysis = exp.get_scalability_analysis()
                summary["scalability"] = {
                    "scaling_quality": scalability_analysis.get("scaling_quality", "Unknown"),
                    "scaling_factor": scalability_analysis.get("scaling_factor", 0),
                    "peak_throughput": max(scalability_analysis.get("throughput_curve", [0])),
                }
                
                if scalability_analysis.get("scaling_factor", 1) < 0.7:
                    recommendations.append(
                        "Scalability bottleneck detected. Consider optimizing task distribution "
                        "or reducing communication overhead."
                    )
        
        # Heterogeneity analysis
        if "heterogeneity" in self.report.experiment_results:
            exp = self.experiments.get("heterogeneity")
            if isinstance(exp, HeterogeneityExperiment):
                hetero_analysis = exp.get_heterogeneity_analysis()
                summary["heterogeneity"] = {
                    "scenarios_tested": len(hetero_analysis.get("scenarios", {})),
                    "recommendations": hetero_analysis.get("recommendations", []),
                }
                recommendations.extend(hetero_analysis.get("recommendations", []))
        
        # Failure analysis
        if "failure_simulation" in self.report.experiment_results:
            exp = self.experiments.get("failure_simulation")
            if isinstance(exp, FailureSimulationExperiment):
                failure_analysis = exp.get_failure_analysis()
                summary["failure_recovery"] = {
                    "checkpoint_benefit": failure_analysis.get("checkpoint_comparison", {}),
                    "recovery_stats": failure_analysis.get("recovery_statistics", {}),
                }
                recommendations.extend(failure_analysis.get("recommendations", []))
        
        # Energy analysis
        if "energy_constraints" in self.report.experiment_results:
            exp = self.experiments.get("energy_constraints")
            if isinstance(exp, EnergyConstraintExperiment):
                energy_analysis = exp.get_energy_analysis()
                summary["energy"] = {
                    "scheduling_comparison": energy_analysis.get("scheduling_comparison", {}),
                    "efficiency": energy_analysis.get("energy_efficiency", {}),
                }
                recommendations.extend(energy_analysis.get("recommendations", []))
        
        # Cross-experiment comparisons
        comparisons = self._generate_cross_comparisons()
        
        self.report.summary = summary
        self.report.comparisons = comparisons
        self.report.recommendations = recommendations
    
    def _generate_cross_comparisons(self) -> Dict[str, Any]:
        """Generate comparisons across different experiments"""
        comparisons = {}
        
        results = self.report.experiment_results if self.report else {}
        
        # Compare throughput across experiments
        throughputs = {}
        for name, result in results.items():
            if result.throughput > 0:
                throughputs[name] = result.throughput
        
        if throughputs:
            comparisons["throughput"] = {
                "by_experiment": throughputs,
                "best": max(throughputs.items(), key=lambda x: x[1])[0],
                "worst": min(throughputs.items(), key=lambda x: x[1])[0],
            }
        
        # Compare success rates
        success_rates = {}
        for name, result in results.items():
            if result.total_tasks > 0:
                rate = result.completed_tasks / result.total_tasks * 100
                success_rates[name] = rate
        
        if success_rates:
            comparisons["success_rate"] = {
                "by_experiment": success_rates,
                "best": max(success_rates.items(), key=lambda x: x[1])[0],
            }
        
        # Compare fairness
        fairness = {}
        for name, result in results.items():
            if result.jains_fairness_index > 0:
                fairness[name] = result.jains_fairness_index
        
        if fairness:
            comparisons["fairness"] = {
                "by_experiment": fairness,
                "best": max(fairness.items(), key=lambda x: x[1])[0],
            }
        
        return comparisons
    
    def _print_summary(self) -> None:
        """Print a summary of the evaluation results"""
        if not self.report:
            return
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE: {self.plan.name}")
        print(f"{'='*60}")
        
        duration = self.report.completed_at - self.report.started_at if self.report.completed_at else 0
        print(f"Duration: {duration:.2f} seconds")
        print(f"Experiments run: {len(self.report.experiment_results)}")
        
        print("\nResults Summary:")
        print("-" * 40)
        
        for name, result in self.report.experiment_results.items():
            print(f"\n{name.upper()}:")
            print(f"  Total tasks: {result.total_tasks}")
            print(f"  Completed: {result.completed_tasks}")
            print(f"  Throughput: {result.throughput:.2f} tasks/sec")
            if result.avg_task_time:
                print(f"  Avg task time: {result.avg_task_time:.3f}s")
            print(f"  Fairness (JFI): {result.jains_fairness_index:.3f}")
        
        if self.report.recommendations:
            print(f"\nRecommendations:")
            print("-" * 40)
            for i, rec in enumerate(self.report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\nFull report saved to: {self.plan.output_dir}")


async def run_quick_evaluation(
    output_dir: str = "./evaluation_results",
    num_workers: int = 5,
    num_tasks: int = 50
) -> EvaluationReport:
    """
    Run a quick evaluation with reduced parameters for testing.
    
    Args:
        output_dir: Directory for output files
        num_workers: Number of simulated workers
        num_tasks: Number of tasks per experiment
    
    Returns:
        Evaluation report
    """
    # Create quick evaluation plan
    plan = EvaluationPlan(
        name="Quick Evaluation",
        description="Rapid evaluation with reduced parameters",
        num_iterations=1,
        output_dir=output_dir,
        random_seed=42,
    )
    
    # Customize configs for quick run
    workload = WorkloadConfig(
        num_tasks=num_tasks,
        task_duration_mean=0.5,
        task_duration_std=0.1,
    )
    
    plan.scalability_config = ScalabilityConfig(
        name="Quick Scalability",
        num_iterations=1,
        num_workers=num_workers,
        worker_counts=[2, 4, 8],
        workload=workload,
        output_dir=output_dir,
    )
    
    plan.heterogeneity_config = HeterogeneityConfig(
        name="Quick Heterogeneity",
        num_iterations=1,
        num_workers=num_workers,
        workload=workload,
        output_dir=output_dir,
    )
    
    plan.failure_config = FailureSimulationConfig(
        name="Quick Failure Simulation",
        num_iterations=1,
        num_workers=num_workers,
        workload=workload,
        output_dir=output_dir,
    )
    
    plan.energy_config = EnergyConstraintConfig(
        name="Quick Energy Constraints",
        num_iterations=1,
        num_workers=num_workers,
        workload=workload,
        output_dir=output_dir,
    )
    
    runner = EvaluationRunner(plan)
    return await runner.run()


async def run_full_evaluation(
    output_dir: str = "./evaluation_results",
    num_iterations: int = 5
) -> EvaluationReport:
    """
    Run a comprehensive evaluation with full parameters.
    
    Args:
        output_dir: Directory for output files
        num_iterations: Number of iterations per experiment
    
    Returns:
        Evaluation report
    """
    plan = EvaluationPlan(
        name="Full CROWDio Evaluation",
        description="Comprehensive evaluation of all system aspects",
        num_iterations=num_iterations,
        output_dir=output_dir,
        random_seed=42,
    )
    
    runner = EvaluationRunner(plan)
    return await runner.run()


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CROWDio Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation")
    parser.add_argument("--output", default="./evaluation_results", help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(run_quick_evaluation(output_dir=args.output))
    else:
        asyncio.run(run_full_evaluation(output_dir=args.output, num_iterations=args.iterations))
