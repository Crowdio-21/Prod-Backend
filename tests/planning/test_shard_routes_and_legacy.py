import asyncio
import hashlib
import json

import pytest
from fastapi import FastAPI
import httpx

from foreman.api import shard_routes
from foreman.core.foreman_policy import Assignment, AssignmentRecord, ReconfigurationExecutor, WorkerTelemetry
from foreman.planning.contract import (
    AtomSpec,
    Manifest,
    ShardSpec,
    TensorContract,
    TensorSpec,
)
from foreman.planning.registry import ShardRegistry


@pytest.fixture()
def client(tmp_path, monkeypatch):
    data = bytes(range(256)) * 40  # 10240 bytes
    (tmp_path / "s.onnx").write_bytes(data)
    tensor = TensorContract(inputs=[TensorSpec(name="x", dtype="float32", shape=[1])],
                            outputs=[TensorSpec(name="y", dtype="float32", shape=[1])])
    manifest = Manifest(
        model_id="m", cut_grid=1, atoms=[AtomSpec(atom_index=0, name="all")],
        shards=[ShardSpec(shard_id="s", model_id="m", role="full", start_atom=0, end_atom=0, bytes=len(data),
                          flops=1, in_act_bytes_per_token=1, out_act_bytes_per_token=1, tensor_contract=tensor,
                          sha256=hashlib.sha256(data).hexdigest(), file_name="s.onnx")],
        default_composition=["s"],
    )
    registry = ShardRegistry(manifest, tmp_path)
    monkeypatch.setattr(shard_routes, "_registry", lambda: registry)
    app = FastAPI()
    app.include_router(shard_routes.router)
    return _SyncClient(app), data, registry


class _SyncClient:
    """Minimal sync wrapper over httpx.ASGITransport (Starlette 0.27's TestClient needs httpx<0.28)."""

    def __init__(self, app):
        self.app = app

    def _call(self, method, url, **kwargs):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t") as c:
                return await c.request(method, url, **kwargs)
        return asyncio.run(go())

    def get(self, url, **kwargs):
        return self._call("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._call("POST", url, **kwargs)


def test_full_download_has_checksum_headers(client):
    http, data, registry = client
    response = http.get("/api/shards/s")
    assert response.status_code == 200 and response.content == data
    assert response.headers["x-content-sha256"] == hashlib.sha256(data).hexdigest()
    assert registry.verify("s")


def test_resumable_range_transfer(client):
    http, data, _ = client
    first = http.get("/api/shards/s", headers={"Range": "bytes=0-4095"})
    rest = http.get("/api/shards/s", headers={"Range": "bytes=4096-"})
    assert first.status_code == rest.status_code == 206
    assert first.headers["content-range"] == f"bytes 0-4095/{len(data)}"
    assert first.content + rest.content == data
    suffix = http.get("/api/shards/s", headers={"Range": "bytes=-10"})
    assert suffix.content == data[-10:]
    assert http.get("/api/shards/s", headers={"Range": f"bytes={len(data)}-"}).status_code == 416


def test_unknown_shard_and_probe(client):
    http, _, _ = client
    assert http.get("/api/shards/nope").status_code == 404
    assert len(http.get("/api/probe/1024").content) == 1024
    assert http.post("/api/probe", content=b"x" * 2048).json() == {"received": 2048}
    assert http.get("/api/shards/manifest").json()["model_id"] == "m"


def test_legacy_executor_unloads_the_shard_being_served():
    calls = []

    class Transport:
        async def prefetch(self, w, s): calls.append(("prefetch", w, s))
        async def assign(self, w, s, e): calls.append(("assign", w, s)); return {}
        async def cutover(self, a, e): calls.append(("cutover",))
        async def drain(self, w): calls.append(("drain", w))
        async def unload(self, w, s): calls.append(("unload", w, s))
        async def verify(self, a, e): calls.append(("verify",))

    current = AssignmentRecord(Assignment({"x": "w1", "y": "w2"}))
    target = Assignment({"x": "w2", "y": "w1"})
    workers = {w: WorkerTelemetry(w, disk_shards={"x", "y"}) for w in ("w1", "w2")}
    asyncio.run(ReconfigurationExecutor(Transport()).execute(current, target, workers, now=0))
    unloads = [c for c in calls if c[0] == "unload"]
    assert ("unload", "w2", "y") in unloads and ("unload", "w1", "x") in unloads
