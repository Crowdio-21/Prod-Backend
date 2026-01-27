"""
Visualization Module for CROWDio Evaluation Framework.

Generates charts, graphs, and visual reports for evaluation results.
Supports matplotlib for static charts and optional plotly for interactive visualizations.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

# Type hints for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Figure = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    width: float = 10.0
    height: float = 6.0
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: List[str] = field(default_factory=lambda: [
        "#2ecc71",  # Green
        "#3498db",  # Blue
        "#e74c3c",  # Red
        "#9b59b6",  # Purple
        "#f39c12",  # Orange
        "#1abc9c",  # Teal
        "#e91e63",  # Pink
        "#795548",  # Brown
    ])
    font_size: int = 12
    title_size: int = 14
    label_size: int = 11
    legend_size: int = 10
    grid_alpha: float = 0.3
    save_format: str = "png"


class EvaluationVisualizer:
    """
    Generates visualizations for CROWDio evaluation results.
    
    Features:
    - Scalability charts (speedup, efficiency curves)
    - Load distribution heatmaps
    - Energy consumption graphs
    - Failure recovery timelines
    - Communication overhead analysis
    - Comparison charts across experiments
    """
    
    def __init__(
        self,
        output_dir: Path | str = "evaluation_charts",
        config: Optional[ChartConfig] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ChartConfig()
        
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed. Visualization features limited.")
    
    def _setup_style(self) -> None:
        """Apply chart styling."""
        if not HAS_MATPLOTLIB:
            return
        try:
            plt.style.use(self.config.style)
        except OSError:
            plt.style.use("ggplot")
        
        plt.rcParams.update({
            "font.size": self.config.font_size,
            "axes.titlesize": self.config.title_size,
            "axes.labelsize": self.config.label_size,
            "legend.fontsize": self.config.legend_size,
            "figure.figsize": (self.config.width, self.config.height),
            "figure.dpi": self.config.dpi,
        })
    
    def _save_figure(self, fig: Figure, name: str) -> Path:
        """Save figure to file."""
        filepath = self.output_dir / f"{name}.{self.config.save_format}"
        fig.savefig(filepath, bbox_inches="tight", dpi=self.config.dpi)
        plt.close(fig)
        return filepath
    
    # ==================== Scalability Charts ====================
    
    def plot_scalability_speedup(
        self,
        worker_counts: List[int],
        execution_times: List[float],
        title: str = "Scalability: Speedup Analysis",
        save_name: str = "scalability_speedup"
    ) -> Optional[Path]:
        """
        Plot speedup curve for scalability experiment.
        
        Args:
            worker_counts: List of worker counts tested
            execution_times: Corresponding execution times
            title: Chart title
            save_name: Filename for saved chart
        
        Returns:
            Path to saved chart or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = self.config.color_palette
        
        # Left plot: Execution time
        ax1.plot(worker_counts, execution_times, 'o-', 
                color=colors[0], linewidth=2, markersize=8, label="Actual")
        ax1.set_xlabel("Number of Workers")
        ax1.set_ylabel("Execution Time (seconds)")
        ax1.set_title("Execution Time vs Workers")
        ax1.grid(True, alpha=self.config.grid_alpha)
        ax1.set_xticks(worker_counts)
        
        # Calculate speedup
        baseline_time = execution_times[0]
        speedups = [baseline_time / t for t in execution_times]
        ideal_speedups = worker_counts
        
        # Right plot: Speedup
        ax2.plot(worker_counts, speedups, 'o-', 
                color=colors[1], linewidth=2, markersize=8, label="Actual Speedup")
        ax2.plot(worker_counts, ideal_speedups, '--', 
                color=colors[2], linewidth=2, label="Ideal Linear Speedup")
        ax2.set_xlabel("Number of Workers")
        ax2.set_ylabel("Speedup")
        ax2.set_title("Speedup Analysis")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=self.config.grid_alpha)
        ax2.set_xticks(worker_counts)
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_scalability_efficiency(
        self,
        worker_counts: List[int],
        execution_times: List[float],
        title: str = "Scalability: Parallel Efficiency",
        save_name: str = "scalability_efficiency"
    ) -> Optional[Path]:
        """Plot parallel efficiency curve."""
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate efficiency
        baseline_time = execution_times[0]
        efficiencies = [(baseline_time / (t * n)) * 100 
                        for t, n in zip(execution_times, worker_counts)]
        
        colors = self.config.color_palette
        bars = ax.bar(range(len(worker_counts)), efficiencies, 
                     color=colors[0], edgecolor=colors[1], linewidth=1.5)
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f"{eff:.1f}%", ha="center", va="bottom", fontsize=10)
        
        # Add ideal line
        ax.axhline(y=100, color=colors[2], linestyle="--", 
                  linewidth=2, label="Ideal (100%)")
        
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Parallel Efficiency (%)")
        ax.set_title(title)
        ax.set_xticks(range(len(worker_counts)))
        ax.set_xticklabels(worker_counts)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha, axis="y")
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_throughput_scaling(
        self,
        worker_counts: List[int],
        throughputs: List[float],
        title: str = "Throughput Scaling",
        save_name: str = "throughput_scaling"
    ) -> Optional[Path]:
        """Plot throughput as workers increase."""
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = self.config.color_palette
        
        ax.plot(worker_counts, throughputs, 'o-', 
               color=colors[0], linewidth=2, markersize=10, label="Actual Throughput")
        
        # Calculate and plot ideal linear throughput
        base_throughput = throughputs[0]
        ideal_throughputs = [base_throughput * n for n in worker_counts]
        ax.plot(worker_counts, ideal_throughputs, '--', 
               color=colors[2], linewidth=2, label="Ideal Linear Scaling")
        
        ax.fill_between(worker_counts, throughputs, alpha=0.3, color=colors[0])
        
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Throughput (tasks/second)")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.set_xticks(worker_counts)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    # ==================== Load Balancing Charts ====================
    
    def plot_load_distribution(
        self,
        worker_loads: Dict[str, List[int]],
        title: str = "Load Distribution Across Workers",
        save_name: str = "load_distribution"
    ) -> Optional[Path]:
        """
        Plot load distribution across workers over time.
        
        Args:
            worker_loads: Dict mapping worker_id to list of task counts over time
        """
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = self.config.color_palette
        workers = list(worker_loads.keys())
        
        # Left plot: Stacked area chart over time
        if HAS_NUMPY:
            time_points = range(len(list(worker_loads.values())[0]))
            ax1.stackplot(time_points, 
                         [worker_loads[w] for w in workers],
                         labels=workers,
                         colors=colors[:len(workers)],
                         alpha=0.8)
        else:
            for i, (worker, loads) in enumerate(worker_loads.items()):
                ax1.plot(loads, label=worker, color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Active Tasks")
        ax1.set_title("Task Distribution Over Time")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Right plot: Final distribution bar chart
        final_totals = [sum(loads) for loads in worker_loads.values()]
        bars = ax2.bar(workers, final_totals, color=colors[:len(workers)], 
                      edgecolor="black", linewidth=0.5)
        
        mean_load = statistics.mean(final_totals) if final_totals else 0
        ax2.axhline(y=mean_load, color="red", linestyle="--", 
                   linewidth=2, label=f"Mean: {mean_load:.1f}")
        
        for bar, val in zip(bars, final_totals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=10)
        
        ax2.set_xlabel("Worker")
        ax2.set_ylabel("Total Tasks Processed")
        ax2.set_title("Total Tasks Per Worker")
        ax2.legend()
        ax2.grid(True, alpha=self.config.grid_alpha, axis="y")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_load_fairness(
        self,
        fairness_history: List[float],
        title: str = "Load Balancing Fairness Over Time",
        save_name: str = "load_fairness"
    ) -> Optional[Path]:
        """Plot Jain's fairness index over time."""
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = self.config.color_palette
        
        ax.plot(fairness_history, color=colors[0], linewidth=2)
        ax.fill_between(range(len(fairness_history)), fairness_history, 
                       alpha=0.3, color=colors[0])
        
        ax.axhline(y=1.0, color=colors[2], linestyle="--", 
                  linewidth=2, label="Perfect Fairness (1.0)")
        ax.axhline(y=0.8, color=colors[4], linestyle=":", 
                  linewidth=2, label="Good Threshold (0.8)")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Jain's Fairness Index")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    # ==================== Energy Consumption Charts ====================
    
    def plot_energy_consumption(
        self,
        worker_energy: Dict[str, Dict[str, float]],
        title: str = "Energy Consumption by Worker",
        save_name: str = "energy_consumption"
    ) -> Optional[Path]:
        """
        Plot energy consumption breakdown per worker.
        
        Args:
            worker_energy: Dict mapping worker_id to energy breakdown
                          (e.g., {"compute": 10.5, "communication": 2.3, "idle": 0.5})
        """
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = self.config.color_palette
        workers = list(worker_energy.keys())
        
        # Collect all energy types
        energy_types = set()
        for breakdown in worker_energy.values():
            energy_types.update(breakdown.keys())
        energy_types = sorted(energy_types)
        
        # Left plot: Stacked bar chart
        bottom = [0] * len(workers)
        for i, energy_type in enumerate(energy_types):
            values = [worker_energy[w].get(energy_type, 0) for w in workers]
            ax1.bar(workers, values, bottom=bottom, 
                   label=energy_type.capitalize(),
                   color=colors[i % len(colors)])
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax1.set_xlabel("Worker")
        ax1.set_ylabel("Energy Consumption (Joules)")
        ax1.set_title("Energy Breakdown by Worker")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=self.config.grid_alpha, axis="y")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        
        # Right plot: Pie chart of total breakdown
        total_by_type = {et: sum(worker_energy[w].get(et, 0) for w in workers) 
                        for et in energy_types}
        
        sizes = list(total_by_type.values())
        labels = [f"{et.capitalize()}\n({v:.1f}J)" for et, v in total_by_type.items()]
        
        ax2.pie(sizes, labels=labels, colors=colors[:len(energy_types)],
               autopct="%1.1f%%", startangle=90)
        ax2.set_title("Total Energy Distribution")
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_battery_levels(
        self,
        battery_history: Dict[str, List[Tuple[float, float]]],
        title: str = "Battery Levels Over Time",
        save_name: str = "battery_levels"
    ) -> Optional[Path]:
        """
        Plot battery level changes over time for each device.
        
        Args:
            battery_history: Dict mapping device_id to list of (timestamp, level) tuples
        """
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = self.config.color_palette
        
        for i, (device, history) in enumerate(battery_history.items()):
            times = [h[0] for h in history]
            levels = [h[1] for h in history]
            ax.plot(times, levels, 'o-', label=device, 
                   color=colors[i % len(colors)], linewidth=2, markersize=4)
        
        # Add warning thresholds
        ax.axhline(y=20, color="red", linestyle="--", alpha=0.7, label="Critical (20%)")
        ax.axhline(y=50, color="orange", linestyle="--", alpha=0.7, label="Low (50%)")
        
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Battery Level (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    # ==================== Failure Recovery Charts ====================
    
    def plot_failure_timeline(
        self,
        failure_events: List[Dict[str, Any]],
        title: str = "Failure and Recovery Timeline",
        save_name: str = "failure_timeline"
    ) -> Optional[Path]:
        """
        Plot failure and recovery events on a timeline.
        
        Args:
            failure_events: List of dicts with keys:
                - worker_id: str
                - failure_time: float
                - recovery_time: float
                - recovery_success: bool
        """
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(14, 6))
        
        workers = list(set(e["worker_id"] for e in failure_events))
        worker_positions = {w: i for i, w in enumerate(workers)}
        
        for event in failure_events:
            y = worker_positions[event["worker_id"]]
            failure_time = event["failure_time"]
            recovery_time = event.get("recovery_time", failure_time + 5)
            recovery_success = event.get("recovery_success", True)
            
            # Draw failure marker
            ax.scatter(failure_time, y, color="red", s=200, marker="X", zorder=5)
            
            # Draw recovery line
            line_color = "green" if recovery_success else "orange"
            ax.hlines(y=y, xmin=failure_time, xmax=recovery_time,
                     colors=line_color, linewidth=4, alpha=0.7)
            
            # Draw recovery marker
            recovery_marker = "o" if recovery_success else "s"
            ax.scatter(recovery_time, y, color=line_color, s=200, 
                      marker=recovery_marker, zorder=5)
        
        ax.set_yticks(range(len(workers)))
        ax.set_yticklabels(workers)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Worker")
        ax.set_title(title)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor="red", label="Failure"),
            mpatches.Patch(facecolor="green", label="Successful Recovery"),
            mpatches.Patch(facecolor="orange", label="Partial/Failed Recovery"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        ax.grid(True, alpha=self.config.grid_alpha, axis="x")
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_recovery_metrics(
        self,
        mtbf_values: List[float],
        mttr_values: List[float],
        labels: List[str],
        title: str = "Failure Recovery Metrics",
        save_name: str = "recovery_metrics"
    ) -> Optional[Path]:
        """Plot MTBF and MTTR comparison."""
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = self.config.color_palette
        x = range(len(labels))
        width = 0.35
        
        # MTBF
        bars1 = ax1.bar(x, mtbf_values, color=colors[0], edgecolor="black")
        ax1.set_xlabel("Configuration")
        ax1.set_ylabel("Mean Time Between Failures (seconds)")
        ax1.set_title("MTBF Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        for bar, val in zip(bars1, mtbf_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
        ax1.grid(True, alpha=self.config.grid_alpha, axis="y")
        
        # MTTR
        bars2 = ax2.bar(x, mttr_values, color=colors[1], edgecolor="black")
        ax2.set_xlabel("Configuration")
        ax2.set_ylabel("Mean Time To Recover (seconds)")
        ax2.set_title("MTTR Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        for bar, val in zip(bars2, mttr_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
        ax2.grid(True, alpha=self.config.grid_alpha, axis="y")
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_checkpoint_effectiveness(
        self,
        scenarios: List[str],
        without_checkpoint: List[float],
        with_checkpoint: List[float],
        title: str = "Checkpoint Effectiveness",
        save_name: str = "checkpoint_effectiveness"
    ) -> Optional[Path]:
        """Compare recovery times with and without checkpointing."""
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = self.config.color_palette
        x = range(len(scenarios))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], without_checkpoint, width,
                      label="Without Checkpoint", color=colors[2])
        bars2 = ax.bar([i + width/2 for i in x], with_checkpoint, width,
                      label="With Checkpoint", color=colors[0])
        
        ax.set_xlabel("Failure Scenario")
        ax.set_ylabel("Recovery Time (seconds)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha, axis="y")
        
        # Add improvement percentages
        for i, (wo, w) in enumerate(zip(without_checkpoint, with_checkpoint)):
            if wo > 0:
                improvement = ((wo - w) / wo) * 100
                ax.annotate(f"{improvement:.0f}% ↓",
                           xy=(i, max(wo, w)),
                           xytext=(0, 10),
                           textcoords="offset points",
                           ha="center", fontsize=9, color="green")
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    # ==================== Communication Charts ====================
    
    def plot_communication_overhead(
        self,
        message_types: List[str],
        message_counts: List[int],
        message_sizes: List[float],
        title: str = "Communication Overhead Analysis",
        save_name: str = "communication_overhead"
    ) -> Optional[Path]:
        """Plot communication overhead by message type."""
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = self.config.color_palette[:len(message_types)]
        
        # Message count pie
        ax1.pie(message_counts, labels=message_types, colors=colors,
               autopct="%1.1f%%", startangle=90)
        ax1.set_title("Message Count Distribution")
        
        # Total bytes bar chart
        bars = ax2.bar(message_types, message_sizes, color=colors, edgecolor="black")
        ax2.set_xlabel("Message Type")
        ax2.set_ylabel("Total Size (KB)")
        ax2.set_title("Data Transfer by Message Type")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.grid(True, alpha=self.config.grid_alpha, axis="y")
        
        for bar, val in zip(bars, message_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}KB", ha="center", va="bottom", fontsize=9)
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_latency_distribution(
        self,
        latencies: List[float],
        title: str = "Message Latency Distribution",
        save_name: str = "latency_distribution"
    ) -> Optional[Path]:
        """Plot histogram of message latencies."""
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = self.config.color_palette
        
        # Histogram
        ax1.hist(latencies, bins=30, color=colors[0], edgecolor="black", alpha=0.7)
        ax1.axvline(statistics.mean(latencies), color=colors[2], linestyle="--",
                   linewidth=2, label=f"Mean: {statistics.mean(latencies):.2f}ms")
        ax1.axvline(statistics.median(latencies), color=colors[4], linestyle=":",
                   linewidth=2, label=f"Median: {statistics.median(latencies):.2f}ms")
        ax1.set_xlabel("Latency (ms)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Latency Histogram")
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Box plot
        ax2.boxplot(latencies, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=colors[0], alpha=0.7))
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency Box Plot")
        ax2.grid(True, alpha=self.config.grid_alpha, axis="y")
        
        # Add percentile annotations
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        
        stats_text = f"P50: {p50:.2f}ms\nP95: {p95:.2f}ms\nP99: {p99:.2f}ms"
        ax2.text(1.3, statistics.mean(latencies), stats_text, 
                fontsize=10, verticalalignment="center")
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    # ==================== Heterogeneity Charts ====================
    
    def plot_device_performance(
        self,
        device_metrics: Dict[str, Dict[str, float]],
        title: str = "Device Performance Comparison",
        save_name: str = "device_performance"
    ) -> Optional[Path]:
        """
        Plot performance comparison across different device types.
        
        Args:
            device_metrics: Dict mapping device_type to metrics dict
                           (e.g., {"laptop": {"throughput": 10, "efficiency": 0.8}})
        """
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        
        devices = list(device_metrics.keys())
        metrics = list(device_metrics[devices[0]].keys()) if devices else []
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        colors = self.config.color_palette
        
        for idx, metric in enumerate(metrics):
            values = [device_metrics[d].get(metric, 0) for d in devices]
            bars = axes[idx].bar(devices, values, 
                                color=colors[idx % len(colors)], 
                                edgecolor="black")
            axes[idx].set_xlabel("Device Type")
            axes[idx].set_ylabel(metric.replace("_", " ").title())
            axes[idx].set_title(metric.replace("_", " ").title())
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha="right")
            axes[idx].grid(True, alpha=self.config.grid_alpha, axis="y")
            
            for bar, val in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                              f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    # ==================== Experiment Comparison Charts ====================
    
    def plot_experiment_comparison(
        self,
        experiment_results: Dict[str, Dict[str, float]],
        metric_name: str,
        title: str = "Experiment Comparison",
        save_name: str = "experiment_comparison"
    ) -> Optional[Path]:
        """
        Compare a specific metric across different experiments.
        
        Args:
            experiment_results: Dict mapping experiment_name to metrics dict
            metric_name: The metric to compare
        """
        if not HAS_MATPLOTLIB:
            return None
        
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = self.config.color_palette
        experiments = list(experiment_results.keys())
        values = [experiment_results[e].get(metric_name, 0) for e in experiments]
        
        bars = ax.bar(experiments, values, color=colors[:len(experiments)], 
                     edgecolor="black")
        
        ax.set_xlabel("Experiment")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=self.config.grid_alpha, axis="y")
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_multi_metric_radar(
        self,
        experiment_results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Multi-Metric Comparison",
        save_name: str = "multi_metric_radar"
    ) -> Optional[Path]:
        """Create radar/spider chart comparing multiple metrics."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return None
        
        self._setup_style()
        
        experiments = list(experiment_results.keys())
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = self.config.color_palette
        
        for i, exp in enumerate(experiments):
            values = [experiment_results[exp].get(m, 0) for m in metrics]
            # Normalize values to 0-1 scale for each metric
            max_vals = [max(experiment_results[e].get(m, 0) for e in experiments) 
                       for m in metrics]
            norm_values = [v / max_v if max_v > 0 else 0 
                          for v, max_v in zip(values, max_vals)]
            norm_values += norm_values[:1]
            
            ax.plot(angles, norm_values, 'o-', linewidth=2, 
                   label=exp, color=colors[i % len(colors)])
            ax.fill(angles, norm_values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.set_title(title, size=self.config.title_size + 2, fontweight="bold", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    # ==================== Report Generation ====================
    
    def generate_full_report(
        self,
        evaluation_data: Dict[str, Any],
        report_name: str = "evaluation_report"
    ) -> Dict[str, Path]:
        """
        Generate a complete set of visualizations from evaluation data.
        
        Args:
            evaluation_data: Complete evaluation results dictionary
            report_name: Base name for the report
        
        Returns:
            Dict mapping chart names to their file paths
        """
        generated_charts = {}
        
        # Scalability charts
        if "scalability" in evaluation_data:
            data = evaluation_data["scalability"]
            if "worker_counts" in data and "execution_times" in data:
                path = self.plot_scalability_speedup(
                    data["worker_counts"],
                    data["execution_times"],
                    save_name=f"{report_name}_scalability_speedup"
                )
                if path:
                    generated_charts["scalability_speedup"] = path
                
                path = self.plot_scalability_efficiency(
                    data["worker_counts"],
                    data["execution_times"],
                    save_name=f"{report_name}_scalability_efficiency"
                )
                if path:
                    generated_charts["scalability_efficiency"] = path
        
        # Load balancing charts
        if "load_balancing" in evaluation_data:
            data = evaluation_data["load_balancing"]
            if "worker_loads" in data:
                path = self.plot_load_distribution(
                    data["worker_loads"],
                    save_name=f"{report_name}_load_distribution"
                )
                if path:
                    generated_charts["load_distribution"] = path
            
            if "fairness_history" in data:
                path = self.plot_load_fairness(
                    data["fairness_history"],
                    save_name=f"{report_name}_load_fairness"
                )
                if path:
                    generated_charts["load_fairness"] = path
        
        # Energy charts
        if "energy" in evaluation_data:
            data = evaluation_data["energy"]
            if "worker_energy" in data:
                path = self.plot_energy_consumption(
                    data["worker_energy"],
                    save_name=f"{report_name}_energy_consumption"
                )
                if path:
                    generated_charts["energy_consumption"] = path
            
            if "battery_history" in data:
                path = self.plot_battery_levels(
                    data["battery_history"],
                    save_name=f"{report_name}_battery_levels"
                )
                if path:
                    generated_charts["battery_levels"] = path
        
        # Failure recovery charts
        if "failures" in evaluation_data:
            data = evaluation_data["failures"]
            if "events" in data:
                path = self.plot_failure_timeline(
                    data["events"],
                    save_name=f"{report_name}_failure_timeline"
                )
                if path:
                    generated_charts["failure_timeline"] = path
            
            if "checkpoint_comparison" in data:
                cp_data = data["checkpoint_comparison"]
                path = self.plot_checkpoint_effectiveness(
                    cp_data.get("scenarios", []),
                    cp_data.get("without_checkpoint", []),
                    cp_data.get("with_checkpoint", []),
                    save_name=f"{report_name}_checkpoint_effectiveness"
                )
                if path:
                    generated_charts["checkpoint_effectiveness"] = path
        
        # Communication charts
        if "communication" in evaluation_data:
            data = evaluation_data["communication"]
            if "latencies" in data:
                path = self.plot_latency_distribution(
                    data["latencies"],
                    save_name=f"{report_name}_latency_distribution"
                )
                if path:
                    generated_charts["latency_distribution"] = path
        
        # Generate summary
        self._generate_html_report(evaluation_data, generated_charts, report_name)
        
        return generated_charts
    
    def _generate_html_report(
        self,
        evaluation_data: Dict[str, Any],
        charts: Dict[str, Path],
        report_name: str
    ) -> Path:
        """Generate an HTML report with embedded charts."""
        html_path = self.output_dir / f"{report_name}.html"
        
        chart_html = ""
        for name, path in charts.items():
            chart_html += f"""
            <div class="chart-container">
                <h3>{name.replace('_', ' ').title()}</h3>
                <img src="{path.name}" alt="{name}" style="max-width: 100%;">
            </div>
            """
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CROWDio Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart-container h3 {{
            color: #2980b9;
            margin-top: 0;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🚀 CROWDio Evaluation Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <p>This report contains the evaluation results for the CROWDio distributed computing system.</p>
    </div>
    
    <h2>📈 Visualizations</h2>
    {chart_html}
    
    <div class="summary">
        <h2>📝 Raw Data</h2>
        <pre>{json.dumps(evaluation_data, indent=2, default=str)[:5000]}...</pre>
    </div>
    
    <footer>
        <p style="text-align: center; color: #7f8c8d;">
            CROWDio Evaluation Framework | {datetime.now().year}
        </p>
    </footer>
</body>
</html>
        """
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return html_path


# ==================== Quick Visualization Functions ====================

def quick_plot_scalability(
    worker_counts: List[int],
    execution_times: List[float],
    output_dir: str = "charts"
) -> Optional[Path]:
    """Quick helper to plot scalability results."""
    viz = EvaluationVisualizer(output_dir)
    return viz.plot_scalability_speedup(worker_counts, execution_times)


def quick_plot_load_balance(
    worker_loads: Dict[str, List[int]],
    output_dir: str = "charts"
) -> Optional[Path]:
    """Quick helper to plot load distribution."""
    viz = EvaluationVisualizer(output_dir)
    return viz.plot_load_distribution(worker_loads)


def quick_generate_report(
    evaluation_data: Dict[str, Any],
    output_dir: str = "charts"
) -> Dict[str, Path]:
    """Quick helper to generate full report."""
    viz = EvaluationVisualizer(output_dir)
    return viz.generate_full_report(evaluation_data)


if __name__ == "__main__":
    # Example usage and testing
    print("CROWDio Evaluation Visualization Module")
    print("=" * 50)
    
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib not installed. Install with: pip install matplotlib")
        print("   Visualization features are limited without matplotlib.")
    else:
        print("✅ matplotlib available")
        
        # Create sample data and generate test charts
        viz = EvaluationVisualizer("test_charts")
        
        # Test scalability chart
        worker_counts = [1, 2, 4, 8, 16]
        execution_times = [100, 52, 28, 16, 10]
        path = viz.plot_scalability_speedup(worker_counts, execution_times)
        print(f"Generated: {path}")
        
        # Test load distribution
        worker_loads = {
            "worker_1": [3, 4, 2, 5, 3, 4],
            "worker_2": [2, 3, 4, 3, 4, 3],
            "worker_3": [4, 2, 3, 4, 3, 4],
        }
        path = viz.plot_load_distribution(worker_loads)
        print(f"Generated: {path}")
        
        print("\n✅ Visualization module ready!")
