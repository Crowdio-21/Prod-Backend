"""Wiring for the plan coordinator. With ``adaptive_planning`` off nothing is built
and the Foreman behaves exactly as before (policy B0 on the legacy path).

Environment:
  CROWDIO_FLAGS       comma list or JSON of feature flags (overrides the run config)
  CROWDIO_RUN_CONFIG  path to a YAML run config
  CROWDIO_MANIFEST    shard manifest (manifest.json or a legacy export directory)
  CROWDIO_PUBLIC_URL  base URL workers use for /api/probe and /api/shards
  CROWDIO_DNN_WORKERS mobile (default) | pc (validate partitions on PCs) | any
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from foreman.planning.config import RunConfig
from foreman.planning.flags import FeatureFlags

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "tools" / "onnx_partitions_sentiment"


def load_run_config() -> RunConfig:
    path = os.getenv("CROWDIO_RUN_CONFIG")
    config = RunConfig.load(path) if path else RunConfig()
    if os.getenv("CROWDIO_FLAGS"):
        config = replace(config, flags=FeatureFlags.from_env())
    if os.getenv("CROWDIO_DNN_WORKERS"):
        config = replace(config, dnn_workers=os.environ["CROWDIO_DNN_WORKERS"].strip().lower())
    return config


def manifest_path(config: RunConfig) -> Path:
    return Path(config.manifest_path or os.getenv("CROWDIO_MANIFEST") or DEFAULT_MANIFEST)


def build_plan_coordinator(connection_manager):
    config = load_run_config()
    flags = config.flags
    logger.info("Feature flags at run start: %s", flags.to_dict())
    print(f"[FLAGS] {flags.to_dict()}")
    if not flags.adaptive_planning:
        if flags.any_on:
            logger.warning("Feature flags %s need adaptive_planning; ignoring them", flags.to_dict())
        return None
    from foreman.planning.registry import ShardRegistry

    from .plan_coordinator import PlanCoordinator
    from .worker_protocol import WebSocketTransport

    registry = ShardRegistry.load(manifest_path(config))
    base = os.getenv("CROWDIO_PUBLIC_URL", "http://localhost:8000").rstrip("/")
    return PlanCoordinator(config, registry, WebSocketTransport(connection_manager),
                           probe_url=f"{base}/api/probe")
