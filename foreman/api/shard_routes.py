"""Shard serving with checksums and resumable range transfers, plus probe endpoints.

GET  /api/shards/manifest      validated manifest (shared contract)
GET  /api/shards/{shard_id}    file; supports ``Range: bytes=a-b`` (206), sends
                               ``X-Content-SHA256`` and an ETag of the sha256
GET  /api/probe/{size}         ``size`` bytes for passive/active bandwidth probes
POST /api/probe                accepts an upload probe and reports its size
GET  /api/plan/status          plan coordinator status (adaptive_planning)
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from foreman.planning.registry import ShardRegistry

from .websockets import get_ws_manager

router = APIRouter(tags=["shards"])

MAX_PROBE_BYTES = 16 * 1024 * 1024
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


@lru_cache(maxsize=1)
def _fallback_registry() -> ShardRegistry:
    from foreman.core.plan_runtime import load_run_config, manifest_path

    return ShardRegistry.load(manifest_path(load_run_config()))


def _registry() -> ShardRegistry:
    manager = get_ws_manager()
    coordinator = getattr(manager, "plan_coordinator", None)
    if coordinator is not None:
        return coordinator.registry
    try:
        return _fallback_registry()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No shard manifest configured") from exc


def _iter_file(path, start: int, end: int, chunk: int = 1 << 20) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            block = handle.read(min(chunk, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


@router.get("/api/shards/manifest")
async def get_manifest():
    return JSONResponse(_registry().manifest.model_dump(mode="json"))


@router.get("/api/shards/{shard_id}")
async def get_shard(shard_id: str, request: Request):
    registry = _registry()
    try:
        spec = registry.spec(shard_id)
        path = registry.path(shard_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Unknown shard") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Shard file missing")
    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "ETag": f'"{spec.sha256}"',
        "X-Content-SHA256": spec.sha256,
    }
    header = request.headers.get("range")
    if not header:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_iter_file(path, 0, size - 1), media_type="application/octet-stream",
                                 headers=headers)
    match = _RANGE.match(header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise HTTPException(status_code=416, detail="Invalid range")
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:  # suffix range: last N bytes
        start = max(0, size - int(match.group(2)))
        end = size - 1
    if start >= size or end < start:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    end = min(end, size - 1)
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(_iter_file(path, start, end), status_code=206,
                             media_type="application/octet-stream", headers=headers)


@router.get("/api/probe/{size}")
async def probe_download(size: int):
    if size <= 0 or size > MAX_PROBE_BYTES:
        raise HTTPException(status_code=400, detail=f"size must be 1..{MAX_PROBE_BYTES}")
    return Response(content=os.urandom(size), media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"})


@router.post("/api/probe")
async def probe_upload(request: Request):
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_PROBE_BYTES:
            raise HTTPException(status_code=413, detail="probe too large")
    return {"received": received}


@router.get("/api/plan/status")
async def plan_status():
    manager = get_ws_manager()
    coordinator = getattr(manager, "plan_coordinator", None)
    if coordinator is None:
        return {"enabled": False}
    return {"enabled": True, **coordinator.status()}
