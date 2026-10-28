"""YAML run configuration. Every threshold here is a tunable, not a known value."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .flags import FeatureFlags

POLICIES = ("B0", "B1", "B2", "B3", "B4", "B5a", "B5b", "S-aware", "S-aware-placement")
OBJECTIVES = ("min_total", "min_steady_deadline")


@dataclass(frozen=True)
class EstimatorConfig:
    ewma_alpha: float = 0.3
    max_age_s: float = 30.0  # estimates older than this are stale
    min_samples: int = 2  # estimates with fewer samples are stale
    probe_after_s: float = 20.0  # active probe only when older than this ...
    probe_min_interval_s: float = 60.0  # ... and at most this often per worker
    probe_bytes: int = 256 * 1024
    age_std_growth: float = 0.05  # relative std added per second of age
    uncertainty_k: float = 2.0  # bounds are mean +/- k * std
    small_sample_rel_std: float = 0.25  # std floor |mean| * this / sqrt(n): few samples, wide bounds


@dataclass(frozen=True)
class CostConfig:
    compression_ratio: float = 0.9  # measured zlib ratio on fp32 activations; prior
    seq_len: int = 64
    batch: int = 1
    runtime_overhead_bytes: int = 64 * 1024 * 1024
    min_bandwidth_bps: float = 1_000.0
    include_io_legs: bool = True  # Foreman->stage1 input and stageK->Foreman output
    allow_time_multiplexing: bool = False
    donor_min_battery: float = 30.0
    donor_max_thermal: int = 2
    donor_max_uploads: int = 1
    temp_slowdown_per_c: float = 0.02  # above temp_ref_c
    temp_ref_c: float = 38.0


@dataclass(frozen=True)
class ControllerConfig:
    bw_drift: float = 0.3  # relative
    rtt_drift: float = 0.5  # relative
    cpu_drift: float = 0.25  # absolute (0..1)
    temp_drift: float = 5.0  # Celsius
    queue_growth: int = 8  # requests queued at a stage
    problem_threshold: int = 3  # aggregated non-urgent problems per window
    persist_k: int = 2  # trigger must persist for k samples
    cooldown_s: float = 30.0
    max_switches: int = 3
    switch_window_s: float = 300.0
    margin_abs_s: float = 1.0
    margin_frac: float = 0.05


@dataclass(frozen=True)
class RouteConfig:
    margin_frac: float = 0.15
    min_dwell_s: float = 10.0
    capability_retest_s: float = 600.0


@dataclass(frozen=True)
class PrefetchConfig:
    max_link_utilisation: float = 0.3
    min_battery: float = 40.0
    max_thermal: int = 1
    top_candidates: int = 3


@dataclass(frozen=True)
class RunConfig:
    policy: str = "B0"
    objective: str = "min_total"
    deadline_s: Optional[float] = None
    flags: FeatureFlags = field(default_factory=FeatureFlags)
    seed: int = 0
    n_requests: int = 100
    run_id: Optional[str] = None
    log_path: Optional[str] = None
    manifest_path: Optional[str] = None
    scenario: List[Dict[str, Any]] = field(default_factory=list)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    routes: RouteConfig = field(default_factory=RouteConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    planner_mode: str = "exhaustive"  # exhaustive | pareto_dp | scalar_dp
    dnn_workers: str = "mobile"  # mobile | pc (validate partitions on PCs) | any

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; expected one of {POLICIES}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}")
        if self.objective == "min_steady_deadline" and self.deadline_s is None:
            raise ValueError("min_steady_deadline needs deadline_s")
        from .scope import check_scope

        check_scope(self.dnn_workers)
        if self.planner_mode not in ("exhaustive", "pareto_dp", "scalar_dp"):
            raise ValueError(f"unknown planner_mode {self.planner_mode!r}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunConfig":
        data = dict(data)
        sections = {
            "estimator": EstimatorConfig,
            "cost": CostConfig,
            "controller": ControllerConfig,
            "routes": RouteConfig,
            "prefetch": PrefetchConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in sections:
                kwargs[key] = sections[key](**(value or {}))
            elif key == "flags":
                kwargs[key] = FeatureFlags.from_mapping(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        import yaml

        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def with_(self, **changes: Any) -> "RunConfig":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
