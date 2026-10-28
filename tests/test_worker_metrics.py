import asyncio
import json

import pytest

from foreman.core.worker_metrics import extract_performance_metrics

PC = {"cpu_usage_percent": 12.5, "ram_available_mb": 2048.0, "battery_level": None, "is_charging": None,
      "network_speed_mbps": 3.2, "storage_available_gb": 120.0, "timestamp": "t"}


def test_pc_ping_reply_nested():
    metrics = extract_performance_metrics({"worker_id": "w", "performance_metrics": PC})
    assert metrics == {"cpu_usage_percent": 12.5, "ram_available_mb": 2048.0,
                       "network_speed_mbps": 3.2, "storage_available_gb": 120.0}


def test_pc_periodic_heartbeat_and_android_pong_flat():
    flat = {"worker_id": "w", "status": "online", **PC, "battery_level": 77.0, "is_charging": True}
    metrics = extract_performance_metrics(flat)
    assert metrics["battery_level"] == 77.0 and metrics["is_charging"] is True
    assert metrics["ram_available_mb"] == 2048.0


def test_android_worker_heartbeat():
    data = {"worker_id": "phone", "net": {"bw_down": 24e6, "bw_up": 8e6},
            "res": {"free_ram_mb": 900, "free_disk_mb": 10240, "battery_pct": 64, "cpu_load": 0.25, "thermal": 1}}
    assert extract_performance_metrics(data) == {
        "ram_available_mb": 900.0, "battery_level": 64.0, "storage_available_gb": 10.0,
        "cpu_usage_percent": 25.0, "network_speed_mbps": 24.0,
    }


def test_missing_values_are_left_out_not_nulled():
    assert extract_performance_metrics({"worker_id": "w", "status": "online"}) == {}
    assert extract_performance_metrics({"res": {"battery_pct": None}}) == {}


def test_ping_loop_pings_every_connected_worker():
    from foreman.core.ws_manager import WebSocketManager

    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, text):
            self.sent.append(json.loads(text)["type"])

    sockets = {"a": Socket(), "b": Socket()}

    class Connections:
        def get_all_worker_ids(self):
            return list(sockets)

        def get_worker_websocket(self, worker_id):
            return sockets[worker_id]

    manager = WebSocketManager.__new__(WebSocketManager)
    manager.connection_manager = Connections()

    async def run():
        task = asyncio.create_task(manager.ping_workers(interval_s=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert all(s.sent and set(s.sent) == {"ping"} for s in sockets.values())
