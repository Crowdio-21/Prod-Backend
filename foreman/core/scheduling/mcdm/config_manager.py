"""
MCDM Scheduler Configuration Manager

Manages MCDM scheduler configurations and provides default configurations.
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy import select

from foreman.db.base import async_session
from foreman.db.models import SchedulerConfigModel


class SchedulerConfigManager:
    """Manages MCDM scheduler configurations"""

    # Default configurations for each MCDM algorithm
    DEFAULT_CONFIGS = {
        "aras": {
            "criteria_weights": [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05],
            "criteria_names": [
                "cpu_cores",
                "ram_available_mb",
                "success_rate",
                "battery_level",
                "network_speed_mbps",
                "storage_available_gb",
                "avg_task_duration_sec",
            ],
            "criteria_types": [1, 1, 1, 1, 1, 1, -1],  # 1=benefit, -1=cost
            "description": "ARAS scheduler - balanced for general-purpose computing",
        },
        "edas": {
            "criteria_weights": [0.20, 0.25, 0.20, 0.15, 0.10, 0.10],
            "criteria_names": [
                "cpu_cores",
                "ram_available_mb",
                "success_rate",
                "battery_level",
                "network_speed_mbps",
                "gpu_available",
            ],
            "criteria_types": [1, 1, 1, 1, 1, 1],
            "description": "EDAS scheduler - optimized for resource-intensive tasks",
        },
        "mabac": {
            "criteria_weights": [0.30, 0.25, 0.20, 0.15, 0.10],
            "criteria_names": [
                "success_rate",
                "cpu_cores",
                "ram_available_mb",
                "battery_level",
                "avg_task_duration_sec",
            ],
            "criteria_types": [1, 1, 1, 1, -1],
            "description": "MABAC scheduler - reliability-focused with performance balance",
        },
        "wrr": {
            "criteria_weights": [0.35, 0.25, 0.20, 0.20],
            "criteria_names": [
                "success_rate",
                "cpu_cores",
                "ram_available_mb",
                "battery_level",
            ],
            "criteria_types": [1, 1, 1, 1],
            "description": "WRR scheduler - weighted round robin with reliability emphasis",
        },
    }

    @classmethod
    async def initialize_default_configs(cls, force: bool = False) -> int:
        """
        Initialize default configurations for all MCDM algorithms.
        Only creates configs that don't already exist unless force=True.

        Args:
            force: If True, recreate all configs even if they exist

        Returns:
            Number of configurations created
        """
        created_count = 0
        updated_count = 0

        async with async_session() as session:
            print("\n🔧 Initializing MCDM Scheduler Configurations...")

            for algo_name, config in cls.DEFAULT_CONFIGS.items():
                # Check if config already exists
                result = await session.execute(
                    select(SchedulerConfigModel).filter_by(algorithm_name=algo_name)
                )
                existing = result.scalar_one_or_none()

                if existing and not force:
                    print(
                        f"  ℹ️  {algo_name.upper():8s} - Already configured (skipping)"
                    )
                    continue

                if existing and force:
                    # Update existing configuration
                    existing.criteria_weights = json.dumps(config["criteria_weights"])
                    existing.criteria_names = json.dumps(config["criteria_names"])
                    existing.criteria_types = json.dumps(config["criteria_types"])
                    existing.description = config["description"]
                    existing.updated_at = datetime.now()
                    print(
                        f"  ✓ {algo_name.upper():8s} - Reset to default configuration"
                    )
                    updated_count += 1
                else:
                    # Create new configuration
                    new_config = SchedulerConfigModel(
                        algorithm_name=algo_name,
                        is_active=(algo_name == "aras"),  # ARAS is default
                        criteria_weights=json.dumps(config["criteria_weights"]),
                        criteria_names=json.dumps(config["criteria_names"]),
                        criteria_types=json.dumps(config["criteria_types"]),
                        description=config["description"],
                    )
                    session.add(new_config)
                    active_marker = " (ACTIVE)" if algo_name == "aras" else ""
                    print(f"  ✓ {algo_name.upper():8s} - Initialized{active_marker}")
                    created_count += 1

            await session.commit()

            if created_count > 0 or updated_count > 0:
                print(f"\n✅ Scheduler initialization complete:")
                if created_count > 0:
                    print(f"   • Created {created_count} new configuration(s)")
                if updated_count > 0:
                    print(f"   • Updated {updated_count} configuration(s)")
                print(f"   • Default: ARAS (Additive Ratio Assessment)")
            else:
                print("✅ All scheduler configurations already exist\n")

            return created_count

    @classmethod
    async def get_active_config(cls) -> Optional[Dict]:
        """
        Get the currently active MCDM scheduler configuration.

        Returns:
            Dictionary with config details or None if no active config
        """
        async with async_session() as session:
            result = await session.execute(
                select(SchedulerConfigModel).filter_by(is_active=True)
            )
            config = result.scalar_one_or_none()

            if not config:
                return None

            return {
                "algorithm_name": config.algorithm_name,
                "criteria_weights": json.loads(config.criteria_weights),
                "criteria_names": json.loads(config.criteria_names),
                "criteria_types": json.loads(config.criteria_types),
                "description": config.description,
            }

    @classmethod
    async def activate_scheduler(cls, algorithm_name: str) -> bool:
        """
        Activate a specific MCDM scheduler.

        Args:
            algorithm_name: Name of the algorithm to activate

        Returns:
            True if successful, False if algorithm not found
        """
        async with async_session() as session:
            # Deactivate all configs
            all_configs = await session.execute(select(SchedulerConfigModel))
            for config in all_configs.scalars().all():
                config.is_active = False

            # Activate the specified config
            result = await session.execute(
                select(SchedulerConfigModel).filter_by(algorithm_name=algorithm_name)
            )
            target_config = result.scalar_one_or_none()

            if not target_config:
                return False

            target_config.is_active = True
            target_config.updated_at = datetime.now()
            await session.commit()
            return True

    @classmethod
    async def update_config(
        cls,
        algorithm_name: str,
        criteria_weights: Optional[List[float]] = None,
        criteria_names: Optional[List[str]] = None,
        criteria_types: Optional[List[int]] = None,
        description: Optional[str] = None,
    ) -> bool:
        """
        Update an existing MCDM scheduler configuration.

        Args:
            algorithm_name: Name of the algorithm to update
            criteria_weights: New weights (optional)
            criteria_names: New criteria names (optional)
            criteria_types: New criteria types (optional)
            description: New description (optional)

        Returns:
            True if successful, False if algorithm not found
        """
        async with async_session() as session:
            result = await session.execute(
                select(SchedulerConfigModel).filter_by(algorithm_name=algorithm_name)
            )
            config = result.scalar_one_or_none()

            if not config:
                return False

            if criteria_weights is not None:
                config.criteria_weights = json.dumps(criteria_weights)
            if criteria_names is not None:
                config.criteria_names = json.dumps(criteria_names)
            if criteria_types is not None:
                config.criteria_types = json.dumps(criteria_types)
            if description is not None:
                config.description = description

            config.updated_at = datetime.now()
            await session.commit()
            return True

    @classmethod
    def get_default_config(cls, algorithm_name: str) -> Optional[Dict]:
        """
        Get the default configuration for a specific algorithm.

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            Default configuration dictionary or None if not found
        """
        return cls.DEFAULT_CONFIGS.get(algorithm_name)

    @classmethod
    def validate_config(
        cls,
        criteria_weights: List[float],
        criteria_names: List[str],
        criteria_types: List[int],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a scheduler configuration.

        Args:
            criteria_weights: List of weights
            criteria_names: List of criteria names
            criteria_types: List of criteria types

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check lengths match
        if len(criteria_weights) != len(criteria_names) != len(criteria_types):
            return False, "All lists must have the same length"

        # Check weights sum to ~1.0
        weight_sum = sum(criteria_weights)
        if not (0.95 <= weight_sum <= 1.05):
            return False, f"Weights must sum to ~1.0 (current: {weight_sum})"

        # Check weight range
        for weight in criteria_weights:
            if weight < 0 or weight > 1:
                return False, "All weights must be between 0 and 1"

        # Check criteria types
        for ctype in criteria_types:
            if ctype not in [1, -1]:
                return False, "Criteria types must be 1 (benefit) or -1 (cost)"

        return True, None
