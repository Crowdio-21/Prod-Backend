"""Normalise worker heartbeat metrics into the ``workers`` table columns.

Workers send metrics in three shapes:

* PC reply to PING:      ``data["performance_metrics"] = {cpu_usage_percent, ...}``
* PC periodic heartbeat and Android reply to PING: the same keys at the top level of ``data``
* Android ``worker_heartbeat``: ``data["res"] = {free_ram_mb, free_disk_mb, battery_pct,
  cpu_load (0..1), thermal}`` and ``data["net"] = {bw_down, bw_up (bits/s), ...}``

Only metrics that are present are returned, so a partial heartbeat never
overwrites a known value with NULL.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

COLUMNS = (
    "cpu_usage_percent",
    "ram_available_mb",
    "battery_level",
    "is_charging",
    "network_speed_mbps",
    "storage_available_gb",
)


def _number(value: Any):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_performance_metrics(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``workers`` columns found in a heartbeat payload."""
    if not isinstance(data, Mapping):
        return {}
    nested = data.get("performance_metrics")
    source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else data
    metrics: Dict[str, Any] = {}
    for column in COLUMNS:
        if column == "is_charging":
            if isinstance(source.get(column), bool):
                metrics[column] = source[column]
            continue
        value = _number(source.get(column))
        if value is not None:
            metrics[column] = value

    res = data.get("res")
    if isinstance(res, Mapping):
        mapped = {
            "ram_available_mb": _number(res.get("free_ram_mb")),
            "battery_level": _number(res.get("battery_pct")),
            "storage_available_gb": (
                _number(res.get("free_disk_mb")) / 1024.0 if _number(res.get("free_disk_mb")) is not None else None
            ),
            "cpu_usage_percent": (
                _number(res.get("cpu_load")) * 100.0 if _number(res.get("cpu_load")) is not None else None
            ),
        }
        metrics.update({k: v for k, v in mapped.items() if v is not None and k not in metrics})
    net = data.get("net")
    if isinstance(net, Mapping) and "network_speed_mbps" not in metrics:
        # Android measures bits/s on timed, network-limited transfers.
        down = _number(net.get("bw_down"))
        if down is not None:
            metrics["network_speed_mbps"] = down / 1e6
    return metrics
