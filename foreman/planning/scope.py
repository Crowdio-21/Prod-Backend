"""DNN worker scope: which kinds of worker may run shards.

``mobile`` (default): Android workers only. ``pc``: PC workers only, for
validating partitions on a desktop. ``any``: both.
"""

from __future__ import annotations

from typing import Optional

SCOPES = ("mobile", "pc", "any")


def check_scope(scope: str) -> str:
    scope = (scope or "mobile").strip().lower()
    if scope not in SCOPES:
        raise ValueError(f"DNN worker scope must be one of {SCOPES}, got {scope!r}")
    return scope


def infer_kind(device_class: Optional[str]) -> str:
    """Fallback when the worker type is unknown: "pc*" classes are PCs, the rest mobile."""
    return "pc" if (device_class or "").lower().startswith("pc") else "mobile"


def kind_allowed(kind: str, scope: str) -> bool:
    if scope == "any":
        return kind in ("mobile", "pc")
    return kind == scope
