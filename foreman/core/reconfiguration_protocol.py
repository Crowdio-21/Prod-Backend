"""Versioned wire messages for shard reconfiguration.

These messages intentionally use a small envelope instead of extending the
SDK's installed ``MessageType`` enum. Older SDKs can ignore them safely and
new workers can parse them before falling back to the legacy protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

PROTOCOL_VERSION = 1
PREFETCH = "prefetch"
PREFETCH_DONE = "prefetch_done"
ASSIGN = "assign"
LOAD_COMPLETE = "load_complete"
CUTOVER = "cutover"
DRAIN = "drain"
UNLOAD = "unload"
VERIFY = "verify"
VERIFY_RESULT = "verify_result"
RECONFIGURATION_TYPES = {
    PREFETCH, PREFETCH_DONE, ASSIGN, LOAD_COMPLETE, CUTOVER, DRAIN, UNLOAD,
    VERIFY, VERIFY_RESULT,
}


@dataclass(frozen=True)
class ReconfigurationMessage:
    type: str
    data: Mapping[str, Any]
    epoch: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "crowdio.reconfiguration",
            "version": PROTOCOL_VERSION,
            "type": self.type,
            "epoch": self.epoch,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReconfigurationMessage | None":
        if value.get("protocol") != "crowdio.reconfiguration":
            return None
        if value.get("version") != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported reconfiguration protocol version: {value.get('version')}")
        message_type = value.get("type")
        if message_type not in RECONFIGURATION_TYPES:
            raise ValueError(f"Unknown reconfiguration message: {message_type}")
        return cls(message_type, value.get("data", {}), value.get("epoch"))
