"""
Scheduler Configuration API Routes

Provides REST API endpoints for managing MCDM scheduler configurations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from foreman.db.base import async_session
from foreman.db.models import SchedulerConfigModel
from sqlalchemy import select, update
from datetime import datetime

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


# Pydantic models for request/response
class SchedulerConfigCreate(BaseModel):
    """Model for creating/updating scheduler configuration"""

    algorithm_name: str = Field(
        ..., description="Algorithm name: aras, edas, mabac, wrr"
    )
    criteria_weights: List[float] = Field(
        ..., description="List of weights (must sum to ~1.0)"
    )
    criteria_names: List[str] = Field(..., description="List of worker attribute names")
    criteria_types: List[int] = Field(
        ..., description="List of 1 (benefit) or -1 (cost)"
    )
    description: Optional[str] = Field(None, description="Human-readable description")

    class Config:
        json_schema_extra = {
            "example": {
                "algorithm_name": "aras",
                "criteria_weights": [0.3, 0.3, 0.2, 0.2],
                "criteria_names": [
                    "cpu_cores",
                    "ram_available_mb",
                    "battery_level",
                    "success_rate",
                ],
                "criteria_types": [1, 1, 1, 1],
                "description": "Custom ARAS config for high-performance tasks",
            }
        }


class SchedulerConfigResponse(BaseModel):
    """Model for scheduler configuration response"""

    id: int
    algorithm_name: str
    is_active: bool
    criteria_weights: List[float]
    criteria_names: List[str]
    criteria_types: List[int]
    description: Optional[str]
    created_at: str
    updated_at: str


class SchedulerListResponse(BaseModel):
    """Model for listing available schedulers"""

    name: str
    type: str  # "simple" or "mcdm"
    is_active: bool
    description: Optional[str] = None


@router.get("/algorithms", response_model=List[SchedulerListResponse])
async def list_available_schedulers():
    """
    List all available scheduler algorithms (simple + MCDM).

    Returns:
        List of available schedulers with their status
    """
    simple_schedulers = [
        {
            "name": "fifo",
            "type": "simple",
            "is_active": False,
            "description": "First-In-First-Out",
        },
        {
            "name": "round_robin",
            "type": "simple",
            "is_active": False,
            "description": "Round Robin distribution",
        },
        {
            "name": "performance",
            "type": "simple",
            "is_active": False,
            "description": "Based on success rate",
        },
        {
            "name": "least_loaded",
            "type": "simple",
            "is_active": False,
            "description": "Fewest completed tasks",
        },
        {
            "name": "priority",
            "type": "simple",
            "is_active": False,
            "description": "Task priority based",
        },
    ]

    # Get MCDM schedulers from database
    mcdm_schedulers = []
    async with async_session() as session:
        result = await session.execute(select(SchedulerConfigModel))
        configs = result.scalars().all()

        for config in configs:
            mcdm_schedulers.append(
                {
                    "name": config.algorithm_name,
                    "type": "mcdm",
                    "is_active": config.is_active,
                    "description": config.description,
                }
            )

    return simple_schedulers + mcdm_schedulers


@router.get("/config", response_model=SchedulerConfigResponse)
async def get_active_scheduler_config():
    """
    Get the currently active scheduler configuration.

    Returns:
        Active scheduler configuration
    """
    async with async_session() as session:
        result = await session.execute(
            select(SchedulerConfigModel).filter_by(is_active=True)
        )
        config = result.scalar_one_or_none()

        if not config:
            raise HTTPException(
                status_code=404, detail="No active scheduler configuration found"
            )

        return SchedulerConfigResponse(
            id=config.id,
            algorithm_name=config.algorithm_name,
            is_active=config.is_active,
            criteria_weights=json.loads(config.criteria_weights),
            criteria_names=json.loads(config.criteria_names),
            criteria_types=json.loads(config.criteria_types),
            description=config.description,
            created_at=config.created_at.isoformat(),
            updated_at=config.updated_at.isoformat(),
        )


@router.get("/config/{algorithm_name}", response_model=SchedulerConfigResponse)
async def get_scheduler_config(algorithm_name: str):
    """
    Get configuration for a specific MCDM algorithm.

    Args:
        algorithm_name: Algorithm name (aras, edas, mabac, wrr)

    Returns:
        Scheduler configuration
    """
    async with async_session() as session:
        result = await session.execute(
            select(SchedulerConfigModel).filter_by(algorithm_name=algorithm_name)
        )
        config = result.scalar_one_or_none()

        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration for '{algorithm_name}' not found",
            )

        return SchedulerConfigResponse(
            id=config.id,
            algorithm_name=config.algorithm_name,
            is_active=config.is_active,
            criteria_weights=json.loads(config.criteria_weights),
            criteria_names=json.loads(config.criteria_names),
            criteria_types=json.loads(config.criteria_types),
            description=config.description,
            created_at=config.created_at.isoformat(),
            updated_at=config.updated_at.isoformat(),
        )


@router.post("/config", response_model=SchedulerConfigResponse)
async def create_or_update_scheduler_config(config: SchedulerConfigCreate):
    """
    Create or update a scheduler configuration.

    Args:
        config: Scheduler configuration to create/update

    Returns:
        Created/updated configuration
    """
    # Validate configuration
    if (
        len(config.criteria_weights)
        != len(config.criteria_names)
        != len(config.criteria_types)
    ):
        raise HTTPException(
            status_code=400,
            detail="criteria_weights, criteria_names, and criteria_types must have the same length",
        )

    weight_sum = sum(config.criteria_weights)
    if not (0.95 <= weight_sum <= 1.05):
        raise HTTPException(
            status_code=400,
            detail=f"criteria_weights must sum to ~1.0 (current sum: {weight_sum})",
        )

    for weight in config.criteria_weights:
        if weight < 0 or weight > 1:
            raise HTTPException(
                status_code=400, detail="All weights must be between 0 and 1"
            )

    for ctype in config.criteria_types:
        if ctype not in [1, -1]:
            raise HTTPException(
                status_code=400,
                detail="criteria_types must be 1 (benefit) or -1 (cost)",
            )

    async with async_session() as session:
        # Check if config already exists
        result = await session.execute(
            select(SchedulerConfigModel).filter_by(algorithm_name=config.algorithm_name)
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing configuration
            existing.criteria_weights = json.dumps(config.criteria_weights)
            existing.criteria_names = json.dumps(config.criteria_names)
            existing.criteria_types = json.dumps(config.criteria_types)
            existing.description = config.description
            existing.updated_at = datetime.now()
            await session.commit()
            await session.refresh(existing)

            return SchedulerConfigResponse(
                id=existing.id,
                algorithm_name=existing.algorithm_name,
                is_active=existing.is_active,
                criteria_weights=json.loads(existing.criteria_weights),
                criteria_names=json.loads(existing.criteria_names),
                criteria_types=json.loads(existing.criteria_types),
                description=existing.description,
                created_at=existing.created_at.isoformat(),
                updated_at=existing.updated_at.isoformat(),
            )
        else:
            # Create new configuration
            new_config = SchedulerConfigModel(
                algorithm_name=config.algorithm_name,
                is_active=False,
                criteria_weights=json.dumps(config.criteria_weights),
                criteria_names=json.dumps(config.criteria_names),
                criteria_types=json.dumps(config.criteria_types),
                description=config.description,
            )
            session.add(new_config)
            await session.commit()
            await session.refresh(new_config)

            return SchedulerConfigResponse(
                id=new_config.id,
                algorithm_name=new_config.algorithm_name,
                is_active=new_config.is_active,
                criteria_weights=json.loads(new_config.criteria_weights),
                criteria_names=json.loads(new_config.criteria_names),
                criteria_types=json.loads(new_config.criteria_types),
                description=new_config.description,
                created_at=new_config.created_at.isoformat(),
                updated_at=new_config.updated_at.isoformat(),
            )


@router.post("/activate/{algorithm_name}")
async def activate_scheduler(algorithm_name: str):
    """
    Activate a specific scheduler algorithm.

    Args:
        algorithm_name: Algorithm to activate (aras, edas, mabac, wrr)

    Returns:
        Success message
    """
    async with async_session() as session:
        # Deactivate all schedulers
        await session.execute(update(SchedulerConfigModel).values(is_active=False))

        # Activate the specified scheduler
        result = await session.execute(
            update(SchedulerConfigModel)
            .where(SchedulerConfigModel.algorithm_name == algorithm_name)
            .values(is_active=True, updated_at=datetime.now())
        )

        if result.rowcount == 0:
            raise HTTPException(
                status_code=404, detail=f"Scheduler '{algorithm_name}' not found"
            )

        await session.commit()

    # TODO: Trigger WebSocketManager to reload scheduler
    # This would require access to the WebSocketManager instance

    return {
        "status": "success",
        "message": f"Activated scheduler: {algorithm_name}",
        "note": "Restart foreman to apply changes",
    }


@router.get("/stats")
async def get_scheduler_stats():
    """
    Get statistics about the scheduler system.

    Returns:
        Scheduler statistics
    """
    # TODO: Implement stats collection
    # - Current scheduler name
    # - Last 10 task assignments with scores
    # - Worker rankings

    return {
        "status": "not_implemented",
        "message": "Scheduler stats endpoint is not yet implemented",
    }
