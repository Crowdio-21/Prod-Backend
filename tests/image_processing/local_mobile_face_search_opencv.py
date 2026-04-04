
import asyncio
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crowdio import (
    crowdio_connect,
    crowdio_disconnect,
    CROWDio,
    CROWDio_map,
    CROWDioConstant,
)
from common.protocol import create_submit_broadcast_job_message


# =============================================================
# Editable settings (no command-line arguments required)
# =============================================================
HOST = "localhost"
PORT = 9000

# Set True to use crowdio.Constant.FILE_DIR in task config.
# In this mode, ALIAS_FILE_DIR must point to the real folder path.
USE_ALIAS = True
ALIAS_FILE_DIR = r"C:\Users\User\Prod-Backend\tests\image_processing\child"

# Used only when USE_ALIAS is False.
IMAGE_DIR = r"C:\path\to\photos"

# Required input: JSON file containing list[list[float]] embeddings.
CHILD_EMBEDDINGS_JSON = os.path.join(
    os.path.dirname(__file__), "pipeline_output", "child_embeddings2.json"
)

THRESHOLD = 0.75
MAX_RESULTS = 20
RECURSIVE = True
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"]
MIN_FACE_SIZE = 60
SIGNATURE_FACE_SIZE = 64
SIGNATURE_GRID = 16

CLIENT_OUTPUT_JSON = os.path.join(
    os.path.dirname(__file__), "pipeline_output", "face_search_result.json"
)


def get_connected_worker_count(host, port):
    """Return number of online workers from foreman dashboard API."""
    url = f"http://localhost:8000/api/workers"
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        workers = resp.json()
    except Exception as exc:
        print(f"Warning: failed to fetch workers from {url} ({exc}); fallback to 1 task")
        return 1

    if not isinstance(workers, list):
        return 1

    online = 0
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        status = str(worker.get("status", "")).lower()
        # Count only workers that can accept assignments.
        if status in {"online", "idle", "available"}:
            online += 1

    return max(online, 1)


async def CROWDio_broadcast(task_func, base_config, host, port):
    """Best-effort broadcast to all connected workers using one task per worker."""
    worker_count = get_connected_worker_count(host, port)
    # Build protocol payload to lock test behavior to SUBMIT_BROADCAST_JOB contract.
    _ = create_submit_broadcast_job_message(
        func_code="<resolved-by-sdk>",
        base_args=dict(base_config),
        job_id="client-broadcast-preview",
        target_workers=worker_count,
    )
    task_args = [dict(base_config) for _ in range(worker_count)]
    print(f"broadcast_mode : one task per worker ({worker_count} task(s))")
    results = await CROWDio_map(task_func, task_args)
    return results, worker_count


@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["scanned_images", "matches", "errors"],
)
def face_search_on_device(config):
    """Run on a worker: scan images, detect faces, and match to child embeddings."""

    import builtins
    import glob
    import os
    import platform
    import time

    import numpy as np

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        return {
            "device_id": platform.node() or "unknown-device",
            "scanned_images": 0,
            "matches": [],
            "errors": [
                "OpenCV import failed on worker.",
                f"Install dependency: opencv-python-headless ({exc})",
            ],
        }

    def resolve_path_alias(value):
        if not isinstance(value, str) or not value.startswith("@CROWDIO:"):
            return value
        alias_map = getattr(builtins, "_crowdio_path_aliases", {})
        return alias_map.get(value, value)

    def maybe_inject_aliases():
        # Mobile runtime should inject aliases automatically.
        # For local testing, allow alias map from task config.
        aliases = config.get("path_aliases")
        if not isinstance(aliases, dict):
            return
        existing = getattr(builtins, "_crowdio_path_aliases", None)
        if not isinstance(existing, dict):
            existing = {}
        # Preserve runtime-provided alias values (mobile should win).
        # Only fill missing aliases from config for local fallback.
        for key, value in aliases.items():
            k = str(key)
            if k not in existing:
                existing[k] = str(value)
        builtins._crowdio_path_aliases = existing

    def imread_unicode(path):
        # cv2.imread can fail on unicode paths; imdecode is robust on Windows.
        with open(path, "rb") as fh:
            data = np.frombuffer(fh.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def cosine_similarity(a, b):
        av = np.asarray(a, dtype=np.float32)
        bv = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 0:
            return 0.0
        return float(np.dot(av, bv) / denom)

    def extract_face_signature(
        gray_image,
        x,
        y,
        w,
        h,
        face_size,
        grid,
    ):
        roi = gray_image[y : y + h, x : x + w]
        if roi is None or roi.size == 0:
            return None

        face = cv2.resize(roi, (face_size, face_size), interpolation=cv2.INTER_AREA)
        face = cv2.equalizeHist(face)
        dct = cv2.dct(face.astype(np.float32) / 255.0)

        feature = dct[:grid, :grid].flatten().astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-8:
            return None

        return (feature / norm).tolist()

    maybe_inject_aliases()

    device_id = platform.node() or "unknown-device"

    image_dir = resolve_path_alias(config.get("image_dir"))
    child_embeddings = config.get("child_embeddings", [])
    threshold = float(config.get("threshold", 0.75))
    max_results = int(config.get("max_results", 20))
    recursive = bool(config.get("recursive", True))
    extensions = config.get("extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"])
    min_face_size = int(config.get("min_face_size", 60))
    signature_face_size = int(config.get("signature_face_size", 64))
    signature_grid = int(config.get("signature_grid", 16))

    if signature_face_size <= 0 or signature_grid <= 0:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": ["signature_face_size and signature_grid must be > 0"],
        }

    signature_dim = signature_grid * signature_grid

    if not isinstance(extensions, list) or not extensions:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"]
    normalized_exts = set()
    for ext in extensions:
        if not isinstance(ext, str):
            continue
        value = ext.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        normalized_exts.add(value)
    if not normalized_exts:
        normalized_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

    if isinstance(image_dir, str) and image_dir.startswith("@CROWDIO:"):
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": [
                "Unresolved path alias for image_dir.",
                "Ensure runtime injects builtins._crowdio_path_aliases or pass path_aliases in config.",
            ],
        }

    if not isinstance(image_dir, str) or not image_dir or not os.path.isdir(image_dir):
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": [f"Image directory not found: {image_dir}"],
        }

    if not isinstance(child_embeddings, list) or not child_embeddings:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": ["Provide non-empty child_embeddings (list[list[float]])"],
        }

    normalized_embeddings = []
    for emb in child_embeddings:
        if not isinstance(emb, list) or not emb:
            continue
        try:
            vector = [float(x) for x in emb]
            if len(vector) == signature_dim:
                normalized_embeddings.append(vector)
        except Exception:
            continue

    child_embeddings = normalized_embeddings
    if not child_embeddings:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": [
                "No valid child embeddings available for OpenCV signature matching.",
                f"Expected embedding length: {signature_dim}",
            ],
        }

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": [f"Failed to load Haar cascade: {cascade_path}"],
        }

    image_paths = []
    files_seen = 0
    if recursive:
        for root, _, files in os.walk(image_dir):
            for name in files:
                files_seen += 1
                ext = os.path.splitext(name)[1].lower()
                if ext in normalized_exts:
                    image_paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(image_dir):
            path = os.path.join(image_dir, name)
            if not os.path.isfile(path):
                continue
            files_seen += 1
            ext = os.path.splitext(name)[1].lower()
            if ext in normalized_exts:
                image_paths.append(path)
    image_paths = sorted(image_paths)

    if not image_paths:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": [
                f"No supported image files found under: {image_dir}",
                f"Supported extensions: {sorted(normalized_exts)}",
                f"Files seen in folder tree: {files_seen}",
            ],
            "elapsed": 0,
            "image_dir": image_dir,
            "threshold": threshold,
            "max_results": max_results,
            "recursive": recursive,
            "method": "opencv_haar_dct",
            "min_face_size": min_face_size,
            "signature_face_size": signature_face_size,
            "signature_grid": signature_grid,
            "expected_embedding_dim": signature_dim,
            "supported_extensions": sorted(normalized_exts),
            "files_seen": files_seen,
        }

    start = time.time()
    errors = []
    matches = []

    for path in image_paths:
        try:
            image = imread_unicode(path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size),
            )

            for face_index, (x, y, w, h) in enumerate(faces):
                signature = extract_face_signature(
                    gray,
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                    signature_face_size,
                    signature_grid,
                )
                if signature is None:
                    continue

                best_score = 0.0
                for child_embedding in child_embeddings:
                    score = cosine_similarity(child_embedding, signature)
                    if score > best_score:
                        best_score = score

                if best_score >= threshold:
                    matches.append(
                        {
                            "image": path,
                            "face_id": face_index,
                            "face_box": [int(x), int(y), int(w), int(h)],
                            "similarity": round(best_score, 4),
                        }
                    )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    matches.sort(key=lambda m: float(m.get("similarity", 0.0)), reverse=True)

    return {
        "device_id": device_id,
        "scanned_images": len(image_paths),
        "matches": matches[:max_results],
        "errors": errors,
        "elapsed": round(time.time() - start, 3),
        "image_dir": image_dir,
        "threshold": threshold,
        "max_results": max_results,
        "recursive": recursive,
        "method": "opencv_haar_dct",
        "min_face_size": min_face_size,
        "signature_face_size": signature_face_size,
        "signature_grid": signature_grid,
        "expected_embedding_dim": signature_dim,
        "supported_extensions": sorted(normalized_exts),
        "files_seen": files_seen,
    }


async def main():
    if USE_ALIAS:
        image_dir_value = CROWDioConstant.FILE_DIR
        if not ALIAS_FILE_DIR:
            raise SystemExit("Set ALIAS_FILE_DIR when USE_ALIAS is True")
        path_aliases = {CROWDioConstant.FILE_DIR: ALIAS_FILE_DIR}
    else:
        if not IMAGE_DIR:
            raise SystemExit("Set IMAGE_DIR when USE_ALIAS is False")
        image_dir_value = IMAGE_DIR
        path_aliases = None

    if not os.path.isfile(CHILD_EMBEDDINGS_JSON):
        raise SystemExit(f"Embeddings file not found: {CHILD_EMBEDDINGS_JSON}")

    with open(CHILD_EMBEDDINGS_JSON, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, list):
        raise SystemExit("CHILD_EMBEDDINGS_JSON must contain a JSON list of embeddings")

    child_embeddings = []
    for row in loaded:
        if not isinstance(row, list) or not row:
            continue
        try:
            child_embeddings.append([float(x) for x in row])
        except Exception:
            continue

    if not child_embeddings:
        raise SystemExit("No valid child embeddings found in CHILD_EMBEDDINGS_JSON")

    task_config = {
        "image_dir": image_dir_value,
        "child_embeddings": child_embeddings,
        "threshold": THRESHOLD,
        "max_results": MAX_RESULTS,
        "recursive": RECURSIVE,
        "extensions": SUPPORTED_EXTENSIONS,
        "min_face_size": MIN_FACE_SIZE,
        "signature_face_size": SIGNATURE_FACE_SIZE,
        "signature_grid": SIGNATURE_GRID,
    }

    if path_aliases is not None:
        task_config["path_aliases"] = path_aliases

    print("\n" + "=" * 64)
    print("Crowdio Face Search Demo (No CLI)")
    print("=" * 64)
    print(f"image_dir      : {task_config['image_dir']}")
    print(f"embeddings     : {CHILD_EMBEDDINGS_JSON}")
    print(f"threshold      : {task_config['threshold']}")
    print(f"max_results    : {task_config['max_results']}")
    print(f"recursive      : {task_config['recursive']}")
    print(f"extensions     : {task_config['extensions']}")
    print(f"min_face_size  : {task_config['min_face_size']}")
    print(f"signature_size : {task_config['signature_face_size']}")
    print(f"signature_grid : {task_config['signature_grid']}")
    print("input_mode     : child_embeddings")
    print(f"host:port      : {HOST}:{PORT}")
    print("=" * 64)

    await crowdio_connect(HOST, PORT)
    try:
        started = time.time()
        results, broadcast_target_workers = await CROWDio_broadcast(
            face_search_on_device, task_config, HOST, PORT
        )
        wall = time.time() - started

        normalized_results = [r for r in results if isinstance(r, dict)] if results else []

        print("\nBroadcast Results")
        print("-" * 64)
        print(f"broadcast_target_devices : {broadcast_target_workers}")
        print(f"worker_results : {len(normalized_results)}")
        print(f"wall_time      : {wall:.2f}s")

        combined_matches = []
        total_errors = 0
        total_scanned = 0

        for idx, result in enumerate(normalized_results, start=1):
            worker_id = result.get("worker_id")
            device_id = result.get("device_id") or f"unknown-device-{idx}"
            source_id = worker_id or device_id
            scanned = int(result.get("scanned_images") or 0)
            matches = result.get("matches", [])
            errors = result.get("errors", [])
            elapsed = result.get("elapsed")

            total_scanned += scanned
            total_errors += len(errors)

            print(
                f"[{idx}] worker={worker_id} device={device_id} scanned={scanned} "
                f"matches={len(matches)} errors={len(errors)} elapsed={elapsed}s"
            )

            for m in matches:
                if not isinstance(m, dict):
                    continue
                enriched = dict(m)
                enriched["source_device_id"] = source_id
                enriched["source_worker_id"] = worker_id
                combined_matches.append(enriched)

        combined_matches.sort(key=lambda m: float(m.get("similarity", 0.0)), reverse=True)

        payload = {
            "wall_time": round(wall, 3),
            "broadcast_target_devices": broadcast_target_workers,
            "worker_results": len(normalized_results),
            "total_scanned_images": total_scanned,
            "total_errors": total_errors,
            "results_by_device": normalized_results,
            "combined_matches": combined_matches[:MAX_RESULTS],
        }

        os.makedirs(os.path.dirname(CLIENT_OUTPUT_JSON), exist_ok=True)
        with open(CLIENT_OUTPUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("-" * 64)
        print(f"Saved JSON -> {CLIENT_OUTPUT_JSON}")

        if combined_matches:
            print("\nTop matches (with device)")
            print("-" * 64)
            for m in combined_matches[: min(10, len(combined_matches))]:
                print(
                    f"{m.get('similarity'):>6}  "
                    f"[{m.get('source_device_id')}]  {m.get('image')}"
                )
        else:
            print("\nNo matches returned by any worker.")
    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
