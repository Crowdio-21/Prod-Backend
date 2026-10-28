"""Centralised feature flags. All default off; all off reproduces policy B0."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional

FLAG_NAMES = ("adaptive_planning", "direct_path", "peer_fetch", "prefetch", "problem_reports")


@dataclass(frozen=True)
class FeatureFlags:
    adaptive_planning: bool = False
    direct_path: bool = False
    peer_fetch: bool = False
    prefetch: bool = False
    problem_reports: bool = False

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "FeatureFlags":
        values = dict(values or {})
        unknown = set(values) - set(FLAG_NAMES)
        if unknown:
            raise ValueError(f"unknown feature flag(s): {sorted(unknown)}")
        return cls(**{name: bool(values.get(name, False)) for name in FLAG_NAMES})

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "FeatureFlags":
        """``CROWDIO_FLAGS`` holds a comma list ("adaptive_planning,prefetch") or JSON."""
        environ = os.environ if environ is None else environ
        raw = (environ.get("CROWDIO_FLAGS") or "").strip()
        if not raw:
            return cls()
        if raw.startswith("{"):
            return cls.from_mapping(json.loads(raw))
        return cls.from_mapping({name.strip(): True for name in raw.split(",") if name.strip()})

    @classmethod
    def all_on(cls) -> "FeatureFlags":
        return cls(**{name: True for name in FLAG_NAMES})

    @property
    def any_on(self) -> bool:
        return any(getattr(self, f.name) for f in fields(self))

    def to_dict(self) -> dict:
        return asdict(self)
