"""JSONL experiment event log."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

EVENT_TYPES = frozenset({
    "run_start", "run_end", "heartbeat", "estimate_update", "planner_call",
    "controller_decision", "switch_start", "switch_end", "route_choice", "route_flip",
    "request_start", "request_end", "prefetch", "prefetch_used", "peer_fetch",
    "problem", "failure", "probe", "scenario", "message",
})


def _default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if value == float("inf"):
        return "inf"
    return str(value)


def canonical(value: Any) -> Any:
    """JSON-safe, order-stable form (tuple dict keys become "a|b")."""
    if is_dataclass(value) and not isinstance(value, type):
        value = {f: getattr(value, f) for f in value.__dataclass_fields__}
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): canonical(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return str(value)
    return value


def stable_hash(value: Any) -> str:
    """Deterministic hash of planner inputs (used for replay)."""
    text = json.dumps(canonical(value), sort_keys=True, default=_default, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EventLog:
    """Thread-safe JSONL writer. ``clock`` is injected so a simulator can drive it."""

    def __init__(
        self,
        path: Optional[str | Path] = None,
        *,
        run_id: Optional[str] = None,
        seed: int = 0,
        policy: str = "B0",
        flags: Optional[Dict[str, bool]] = None,
        clock: Callable[[], float] = time.time,
        keep_in_memory: bool = False,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.seed = seed
        self.policy = policy
        self.flags = dict(flags or {})
        self.clock = clock
        self.epoch: Optional[int] = None
        self.events: List[Dict[str, Any]] = []
        self._keep = keep_in_memory or path is None
        self._lock = threading.Lock()
        self._fh: Optional[TextIO] = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def emit(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type!r}")
        at = fields.pop("at_ts", None)  # explicit event time (simulator completions)
        record = {
            "ts": self.clock() if at is None else at,
            "run_id": self.run_id,
            "seed": self.seed,
            "policy": self.policy,
            "flags": self.flags,
            "epoch": fields.pop("epoch", self.epoch),
            "event": event_type,
            **fields,
        }
        line = json.dumps(record, default=_default, separators=(",", ":"))
        with self._lock:
            if self._keep:
                self.events.append(json.loads(line))
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def read_events(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
