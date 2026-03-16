#!/usr/bin/env python3
"""Tiny HTTP tensor upload server for .npy payloads."""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="Tensor Upload Server")
STORE = Path("tensor_store")
STORE.mkdir(exist_ok=True)
MODEL_STORE = Path(".model_store")
MODEL_STORE.mkdir(exist_ok=True)

# Set this to your PC LAN IP so Android clients receive a reachable URL.
PC_IP = os.getenv("TENSOR_SERVER_IP", "192.168.1.7")
PORT = int(os.getenv("TENSOR_SERVER_PORT", "8001"))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Android uploads raw .npy bytes; keep extension consistent for numpy loading.
    name = f"{uuid.uuid4().hex}.npy"
    dst = STORE / name
    data = await file.read()
    dst.write_bytes(data)
    return {"file_url": f"http://{PC_IP}:{PORT}/files/{name}"}


@app.get("/files/{name}")
async def files(name: str):
    p = STORE / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(p)


@app.get("/.model_store/{subpath:path}")
async def model_files(subpath: str):
    # Prevent path traversal and serve model artifacts from .model_store.
    root = MODEL_STORE.resolve()
    target = (MODEL_STORE / subpath).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
