"""Per-heartbeat conditions log for experiments (RSSI, link speed, bandwidth, free RAM, battery).

Off unless ``CROWDIO_EXPERIMENT_LOG`` names a JSONL file. One row per worker at most every
``CROWDIO_EXPERIMENT_LOG_INTERVAL`` seconds (default 2), so a run can later be joined with the
radio and device conditions the phones saw while it ran. Nothing here affects scheduling.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Mapping, Optional

_lock = threading.Lock()
_last: Dict[str, float] = {}


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def record_heartbeat(worker_id: str, data: Mapping[str, Any]) -> None:
    path = os.getenv("CROWDIO_EXPERIMENT_LOG")
    if not path or not isinstance(data, Mapping):
        return
    now = time.time()
    interval = float(os.getenv("CROWDIO_EXPERIMENT_LOG_INTERVAL", "2"))
    if now - _last.get(worker_id, 0.0) < interval:
        return
    net = data.get("net") if isinstance(data.get("net"), Mapping) else {}
    res = data.get("res") if isinstance(data.get("res"), Mapping) else {}
    row = {
        "t": round(now, 3), "worker_id": worker_id,
        "rssi_dbm": _number(net.get("rssi")), "link_mbps": _number(net.get("link_mbps")),
        "band": net.get("band"), "bw_down_bps": _number(net.get("bw_down")), "bw_up_bps": _number(net.get("bw_up")),
        "free_ram_mb": _number(res.get("free_ram_mb")), "battery_pct": _number(res.get("battery_pct")),
        "cpu_load": _number(res.get("cpu_load")), "thermal": res.get("thermal"),
    }
    if all(v is None for k, v in row.items() if k not in ("t", "worker_id", "band", "thermal")):
        return  # a PING reply with no phone telemetry
    _last[worker_id] = now
    try:
        with _lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass
