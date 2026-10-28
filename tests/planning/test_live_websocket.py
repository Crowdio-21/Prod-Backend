"""End-to-end smoke test over real WebSockets with the real WebSocketManager.

A raw-socket worker registers with a legacy ``worker_ready`` and then speaks
protocol v2; a client submits planned inference. Runs in a temp directory so
the SQLite database is throwaway.
"""

import asyncio
import json

import pytest
import websockets

from tests.planning.fake_workers import make_manifest


async def _worker(port: int, worker_id: str, shard: str, increment: int, stop: asyncio.Event,
                  worker_type: str = "android_kotlin"):
    from common.protocol import create_worker_ready_message

    async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
        ready = create_worker_ready_message(worker_id)
        ready.data["worker_type"] = worker_type
        await ws.send(ready.to_json())
        report = {"protocol": "crowdio.reconfiguration", "version": 2, "type": "report", "data": {"state": {
            "worker_id": worker_id, "device_class": "phone", "resident_shard": shard,
            "session_state": "READY", "disk_shards": [{"id": shard, "bytes": 1}],
            "observed_up_bps": 1e7, "observed_down_bps": 1e7, "rtt_to_foreman_ms": 5,
            "stage_stats": {"compute_ms_p50": 5, "samples": 3, "shard_id": shard}}}}
        await ws.send(json.dumps(report))
        while not stop.is_set():
            try:
                raw = json.loads(await asyncio.wait_for(ws.recv(), 0.2))
            except asyncio.TimeoutError:
                continue
            if raw.get("protocol") != "crowdio.reconfiguration":
                continue  # legacy pings etc.
            if raw.get("msg_id") and raw["type"] not in ("run", "load", "ack"):
                await ws.send(json.dumps({"protocol": "crowdio.reconfiguration", "version": 2, "type": "msg_ack",
                                          "data": {"ack_of": raw["msg_id"]}}))
            if raw["type"] == "load":
                data = raw["data"]
                await ws.send(json.dumps({"protocol": "crowdio.reconfiguration", "version": 2, "type": "load_result",
                                          "data": {"shard_id": data["shard_id"], "epoch": data["epoch"], "ok": True,
                                                   "timings": {"load_s": 1.0, "init_s": 0.5}}}))
            if raw["type"] == "run":
                data = raw["data"]
                out = {"value": data["activation"]["value"] + increment}
                await ws.send(json.dumps({"protocol": "crowdio.reconfiguration", "version": 2,
                                          "type": "activation", "epoch": data["epoch"], "data": {
                                              "stage": data["stage"], "epoch": data["epoch"],
                                              "request_id": data["request_id"], "tensor": out,
                                              "compute_ms": 5}}))


@pytest.mark.parametrize("worker_type,scope", [
    ("android_kotlin", None),  # default: DNN shards run on mobile workers
    ("pc_python", "pc"),  # validating partitions on PC workers
])
def test_live_foreman_round_trip(tmp_path, monkeypatch, worker_type, scope):
    manifest = make_manifest()
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CROWDIO_FLAGS", "adaptive_planning")
    if scope is None:
        monkeypatch.delenv("CROWDIO_DNN_WORKERS", raising=False)
    else:
        monkeypatch.setenv("CROWDIO_DNN_WORKERS", scope)
    monkeypatch.setenv("CROWDIO_MANIFEST", str(tmp_path / "manifest.json"))

    import sys

    for stream in (sys.stdout, sys.stderr):  # same guard as foreman/main.py (emoji logs on cp1252)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    async def scenario():
        from foreman.core.ws_manager import WebSocketManager
        from foreman.db.base import init_db

        await init_db()
        manager = WebSocketManager()
        assert manager.plan_coordinator is not None
        coordinator = manager.plan_coordinator
        coordinator.config = coordinator.config.with_(controller=coordinator.config.controller)
        for size in (1, 10**6):
            coordinator.learner.observe("phone", size, load=1.0, init=0.5, write=0.2, unload=0.3, peak_mem=size)
        server = await websockets.serve(manager.handle_connection, "127.0.0.1", 0, max_size=None)
        port = server.sockets[0].getsockname()[1]
        stop = asyncio.Event()
        workers = [asyncio.create_task(_worker(port, w, s, inc, stop, worker_type))
                   for w, s, inc in (("w1", "a", 1), ("w2", "b", 2), ("w3", "c", 1))]
        try:
            for _ in range(100):  # wait until all three reports are ingested
                if all(coordinator.telemetry._workers.get(w) and coordinator.telemetry._workers[w].state
                       and coordinator.telemetry._workers[w].state.resident_shard for w in ("w1", "w2", "w3")):
                    break
                await asyncio.sleep(0.05)
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as client:
                await client.send(json.dumps({"protocol": "crowdio.reconfiguration", "version": 2,
                                              "type": "submit_inference",
                                              "data": {"model_id": "m", "inputs": [{"value": i} for i in range(5)]}}))
                result = None
                for _ in range(100):
                    try:
                        message = json.loads(await asyncio.wait_for(client.recv(), 5))
                    except asyncio.TimeoutError:
                        decisions = [(e["event"], e.get("action"), e.get("reason"), e.get("kind"))
                                     for e in coordinator.log.events[-15:]]
                        raise AssertionError(f"no result; status={coordinator.status()} events={decisions}")
                    if message["type"] == "inference_result":
                        result = message["data"]["results"]
                        break
            assert result is not None
            assert [r["tensor"]["value"] for r in result] == [i + 4 for i in range(5)]
            assert coordinator.plan.assignment() == (("a", "w1"), ("b", "w2"), ("c", "w3"))
        finally:
            stop.set()
            await asyncio.gather(*workers, return_exceptions=True)
            server.close()
            await server.wait_closed()
            await coordinator.stop()
            from foreman.db.base import engine

            await engine.dispose()

    asyncio.run(scenario())
