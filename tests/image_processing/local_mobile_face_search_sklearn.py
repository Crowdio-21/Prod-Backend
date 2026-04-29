#!/usr/bin/env python3
"""Crowdio face search using OpenCV LBP features + scikit-learn NearestNeighbors.

This test script is intended for mobile-friendly execution via Chaquopy:
- no DeepFace dependency
- OpenCV Haar face detection
- LBP feature extraction
- scikit-learn nearest-neighbor matching against reference embeddings

Reference embeddings should come from:
`tests/image_processing/generate_face_embeddings_opencv.py`
"""

import asyncio
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crowdio import crowdio_connect, crowdio_map, crowdio_disconnect, CROWDio, CROWDioConstant
from common.protocol import create_submit_broadcast_job_message  # noqa: E402

# =============================================================
# Editable settings (no command-line arguments required)
# =============================================================
HOST = "localhost"
PORT = 9000

# Set True to use crowdio Constant path alias in task config.
USE_ALIAS = True
ALIAS_FILE_DIR = r"C:\Users\User\Prod-Backend\tests\image_processing\child"

# Used only when USE_ALIAS is False.
IMAGE_DIR = r"C:\path\to\photos"

# Required input: JSON file containing list[list[float]] embeddings.
CHILD_EMBEDDINGS_JSON = os.path.join(
    os.path.dirname(__file__), "pipeline_output", "child_embeddings.json"
)

# FIX 1: Lowered threshold — LBP cosine similarity rarely exceeds 0.65.
# Set DEBUG_MODE=True to see all detected faces and their scores
# even when they fall below threshold.
THRESHOLD = 0.45
DEBUG_MODE = True

MAX_RESULTS = 20
TOP_K_NEIGHBORS = 3
RECURSIVE = True
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"]
MIN_FACE_SIZE = 10
MAX_FACE_SIZE = 3000          # FIX 3: Reject implausibly large detections (0 = no limit)
SIGNATURE_FACE_SIZE = 64
LBP_GRID = 8
SIGNATURE_DIM = LBP_GRID * LBP_GRID * 32
NMS_OVERLAP_THRESHOLD = 0.3  # FIX 2: IoU threshold for Non-Maximum Suppression
BEST_MATCH_PER_IMAGE = True  # FIX 4: Return only best-scoring face per image

CLIENT_OUTPUT_JSON = os.path.join(
    os.path.dirname(__file__), "pipeline_output", "face_search_result_sklearn.json"
)


def get_connected_worker_count(host, port):
    """Return number of online workers from foreman dashboard API."""
    url = "http://localhost:8000/api/workers"
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
        if status in {"online", "idle", "available"}:
            online += 1

    return max(online, 1)


async def CROWDio_broadcast(task_func, base_config, host, port):
    """Best-effort broadcast to all connected workers using one task per worker."""
    worker_count = get_connected_worker_count(host, port)

    _ = create_submit_broadcast_job_message(
        func_code="<resolved-by-sdk>",
        base_args=dict(base_config),
        job_id="client-broadcast-preview",
        target_workers=worker_count,
    )

    task_args = [dict(base_config) for _ in range(worker_count)]
    print(f"broadcast_mode : one task per worker ({worker_count} task(s))")
    results = await crowdio_map(task_func, task_args)
    return results, worker_count


@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["scanned_images", "matches", "errors"],
)
def face_search_on_device(config):
    """Run on worker: scan images, detect faces, and match using sklearn NN."""

    import builtins
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
            "debug_candidates": [],
            "errors": [
                "OpenCV import failed on worker.",
                f"Install dependency: opencv-python-headless ({exc})",
            ],
        }

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover
        return {
            "device_id": platform.node() or "unknown-device",
            "scanned_images": 0,
            "matches": [],
            "debug_candidates": [],
            "errors": [
                "scikit-learn import failed on worker.",
                f"Install dependency: scikit-learn ({exc})",
            ],
        }

    def resolve_path_alias(value):
        if not isinstance(value, str) or not value.startswith("@CROWDIO:"):
            return value
        alias_map = getattr(builtins, "_crowdio_path_aliases", {})
        return alias_map.get(value, value)

    def maybe_inject_aliases():
        aliases = config.get("path_aliases")
        if not isinstance(aliases, dict):
            return
        existing = getattr(builtins, "_crowdio_path_aliases", None)
        if not isinstance(existing, dict):
            existing = {}
        for key, value in aliases.items():
            k = str(key)
            if k not in existing:
                existing[k] = str(value)
        builtins._crowdio_path_aliases = existing

    def imread_robust(path):
        with open(path, "rb") as fh:
            data = np.frombuffer(fh.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def extract_face_signature_lbp(gray_image, x, y, w, h, face_size=64, lbp_grid=8):
        roi = gray_image[y : y + h, x : x + w]
        if roi is None or roi.size == 0:
            return None

        face = cv2.resize(roi, (face_size, face_size), interpolation=cv2.INTER_AREA)
        face = cv2.equalizeHist(face)

        lbp = np.zeros_like(face, dtype=np.uint8)
        for i in range(1, face_size - 1):
            for j in range(1, face_size - 1):
                center = face[i, j]
                neighbors = [
                    face[i - 1, j - 1],
                    face[i - 1, j],
                    face[i - 1, j + 1],
                    face[i, j + 1],
                    face[i + 1, j + 1],
                    face[i + 1, j],
                    face[i + 1, j - 1],
                    face[i, j - 1],
                ]
                code = 0
                for k, n in enumerate(neighbors):
                    if n >= center:
                        code |= 1 << k
                lbp[i, j] = code

        cell_h, cell_w = face_size // lbp_grid, face_size // lbp_grid
        hist_parts = []
        for r in range(lbp_grid):
            for c in range(lbp_grid):
                cell = lbp[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                hist, _ = np.histogram(cell, bins=32, range=(0, 256))
                h_vec = hist.astype(np.float32)
                n = np.linalg.norm(h_vec)
                if n > 0:
                    h_vec /= n
                hist_parts.append(h_vec)

        feature = np.concatenate(hist_parts)
        norm = np.linalg.norm(feature)
        if norm <= 1e-8:
            return None
        return (feature / norm).astype(np.float32)

    def apply_nms(faces, overlap_threshold=0.3):
        """Remove overlapping face detections keeping the largest box."""
        if len(faces) == 0:
            return []

        boxes = [
            (int(x), int(y), int(x + w), int(y + h), int(w), int(h))
            for (x, y, w, h) in faces
        ]
        boxes.sort(key=lambda b: b[4] * b[5], reverse=True)  # largest area first

        kept = []
        suppressed = set()

        for i, (x1, y1, x2, y2, w, h) in enumerate(boxes):
            if i in suppressed:
                continue
            kept.append((x1, y1, w, h))
            area_i = w * h
            for j, (ox1, oy1, ox2, oy2, ow, oh) in enumerate(boxes):
                if j <= i or j in suppressed:
                    continue
                inter_x1 = max(x1, ox1)
                inter_y1 = max(y1, oy1)
                inter_x2 = min(x2, ox2)
                inter_y2 = min(y2, oy2)
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                intersection = inter_w * inter_h
                union = area_i + (ow * oh) - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou > overlap_threshold:
                    suppressed.add(j)

        return kept

    maybe_inject_aliases()

    # Read all config values including new fix parameters
    debug_mode = bool(config.get("debug_mode", False))
    best_match_per_image = bool(config.get("best_match_per_image", True))
    device_id = platform.node() or "unknown-device"

    image_dir = resolve_path_alias(config.get("image_dir"))
    child_embeddings = config.get("child_embeddings", [])
    threshold = float(config.get("threshold", 0.45))
    max_results = int(config.get("max_results", 20))
    recursive = bool(config.get("recursive", True))
    extensions = config.get(
        "extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"]
    )
    min_face_size = int(config.get("min_face_size", 60))
    max_face_size = int(config.get("max_face_size", 0))
    signature_face_size = int(config.get("signature_face_size", 64))
    lbp_grid = int(config.get("lbp_grid", 8))
    signature_dim = int(config.get("signature_dim", lbp_grid * lbp_grid * 32))
    top_k = max(1, int(config.get("top_k_neighbors", 3)))
    nms_overlap_threshold = float(config.get("nms_overlap_threshold", 0.3))

    if signature_face_size <= 0 or lbp_grid <= 0:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "debug_candidates": [],
            "errors": ["signature_face_size and lbp_grid must be > 0"],
        }

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
            "debug_candidates": [],
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
            "debug_candidates": [],
            "errors": [f"Image directory not found: {image_dir}"],
        }

    if not isinstance(child_embeddings, list) or not child_embeddings:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "debug_candidates": [],
            "errors": ["Provide non-empty child_embeddings (list[list[float]])"],
        }

    # FIX 3: Log dimension mismatches instead of silently dropping embeddings
    normalized_embeddings = []
    skipped_dim_mismatch = 0
    errors = []

    for emb in child_embeddings:
        if not isinstance(emb, list) or not emb:
            continue
        try:
            vector = np.asarray([float(x) for x in emb], dtype=np.float32)
            if vector.shape[0] == signature_dim:
                normalized_embeddings.append(vector)
            else:
                skipped_dim_mismatch += 1
        except Exception:
            continue

    if skipped_dim_mismatch > 0:
        errors.append(
            f"Skipped {skipped_dim_mismatch} embedding(s) due to dimension mismatch. "
            f"Expected dim={signature_dim}. "
            f"Regenerate embeddings with --lbp-grid={lbp_grid} "
            f"--signature-face-size={signature_face_size}."
        )

    if not normalized_embeddings:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "debug_candidates": [],
            "errors": errors + [
                "No valid child embeddings available after dimension filtering.",
                f"Expected embedding length: {signature_dim}",
            ],
        }

    try:
        ref_matrix = np.vstack(normalized_embeddings)
        nn_model = NearestNeighbors(
            n_neighbors=min(top_k, ref_matrix.shape[0]),
            metric="cosine",
            algorithm="brute",
        )
        nn_model.fit(ref_matrix)
    except Exception as exc:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "debug_candidates": [],
            "errors": errors + [f"Failed to build sklearn NearestNeighbors model: {exc}"],
        }

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "debug_candidates": [],
            "errors": errors + [f"Failed to load Haar cascade: {cascade_path}"],
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
            "debug_candidates": [],
            "errors": errors + [
                f"No supported image files found under: {image_dir}",
                f"Supported extensions: {sorted(normalized_exts)}",
                f"Files seen in folder tree: {files_seen}",
            ],
            "elapsed": 0,
            "image_dir": image_dir,
            "threshold": threshold,
            "max_results": max_results,
            "recursive": recursive,
            "method": "opencv_haar_lbp_sklearn_nn",
            "min_face_size": min_face_size,
            "signature_face_size": signature_face_size,
            "lbp_grid": lbp_grid,
            "expected_embedding_dim": signature_dim,
            "top_k_neighbors": top_k,
            "supported_extensions": sorted(normalized_exts),
            "files_seen": files_seen,
        }

    start = time.time()
    matches = []
    debug_candidates = []       # FIX 4: Capture below-threshold faces for diagnostics
    faces_detected_total = 0
    faces_after_nms_total = 0

    for path in image_paths:
        try:
            image = imread_robust(path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # FIX 2: Aligned detection parameters with embedding generation script
            raw_faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(min_face_size, min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            faces_detected_total += len(raw_faces)

            # FIX 2: Apply NMS to remove overlapping / duplicate detections
            faces = apply_nms(raw_faces, overlap_threshold=nms_overlap_threshold)

            # FIX 3: Filter out faces exceeding max_face_size
            if max_face_size > 0:
                faces = [(x, y, w, h) for (x, y, w, h) in faces if w <= max_face_size]

            faces_after_nms_total += len(faces)

            for face_index, (x, y, w, h) in enumerate(faces):
                signature = extract_face_signature_lbp(
                    gray,
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                    signature_face_size,
                    lbp_grid,
                )
                if signature is None:
                    continue

                distances, indices = nn_model.kneighbors(signature.reshape(1, -1))
                best_dist = float(distances[0][0])
                best_idx = int(indices[0][0])
                best_similarity = max(0.0, 1.0 - best_dist)

                # FIX 4: Build candidate regardless of threshold for diagnostics
                neighbors = []
                for dist, idx in zip(distances[0], indices[0]):
                    neighbors.append(
                        {
                            "child_index": int(idx),
                            "similarity": round(max(0.0, 1.0 - float(dist)), 4),
                        }
                    )

                candidate = {
                    "image": path,
                    "face_id": face_index,
                    "face_box": [int(x), int(y), int(w), int(h)],
                    "child_index": best_idx,
                    "similarity": round(best_similarity, 4),
                    "neighbors": neighbors,
                    "above_threshold": best_similarity >= threshold,
                }

                if best_similarity >= threshold:
                    matches.append(candidate)
                elif debug_mode:
                    debug_candidates.append(candidate)

        except Exception as exc:
            errors.append(f"{path}: {exc}")

    # FIX 4: Keep only the best-scoring face per image
    if best_match_per_image:
        best_per_image = {}
        for m in matches:
            img_key = m["image"]
            if (
                img_key not in best_per_image
                or m["similarity"] > best_per_image[img_key]["similarity"]
            ):
                best_per_image[img_key] = m
        matches = list(best_per_image.values())

        if debug_mode:
            best_debug_per_image = {}
            for d in debug_candidates:
                img_key = d["image"]
                if (
                    img_key not in best_debug_per_image
                    or d["similarity"] > best_debug_per_image[img_key]["similarity"]
                ):
                    best_debug_per_image[img_key] = d
            debug_candidates = list(best_debug_per_image.values())

    matches.sort(key=lambda m: float(m.get("similarity", 0.0)), reverse=True)
    debug_candidates.sort(key=lambda m: float(m.get("similarity", 0.0)), reverse=True)

    return {
        "device_id": device_id,
        "scanned_images": len(image_paths),
        "faces_detected_raw": faces_detected_total,
        "faces_after_nms": faces_after_nms_total,
        "matches": matches[:max_results],
        "debug_candidates": debug_candidates[:20] if debug_mode else [],
        "errors": errors,
        "elapsed": round(time.time() - start, 3),
        "image_dir": image_dir,
        "threshold": threshold,
        "max_results": max_results,
        "recursive": recursive,
        "method": "opencv_haar_lbp_sklearn_nn",
        "min_face_size": min_face_size,
        "signature_face_size": signature_face_size,
        "lbp_grid": lbp_grid,
        "expected_embedding_dim": signature_dim,
        "top_k_neighbors": top_k,
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

    if isinstance(loaded, dict) and isinstance(loaded.get("embeddings"), list):
        loaded = loaded["embeddings"]

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
        "debug_mode": DEBUG_MODE,                       # FIX 4
        "best_match_per_image": BEST_MATCH_PER_IMAGE,  # FIX 4
        "max_results": MAX_RESULTS,
        "top_k_neighbors": TOP_K_NEIGHBORS,
        "recursive": RECURSIVE,
        "extensions": SUPPORTED_EXTENSIONS,
        "min_face_size": MIN_FACE_SIZE,
        "max_face_size": MAX_FACE_SIZE,                 # FIX 3
        "nms_overlap_threshold": NMS_OVERLAP_THRESHOLD, # FIX 2
        "signature_face_size": SIGNATURE_FACE_SIZE,
        "lbp_grid": LBP_GRID,
        "signature_dim": SIGNATURE_DIM,
    }

    if path_aliases is not None:
        task_config["path_aliases"] = path_aliases

    print("\n" + "=" * 64)
    print("Crowdio Face Search Demo (sklearn, no DeepFace)")
    print("=" * 64)
    print(f"image_dir             : {task_config['image_dir']}")
    print(f"embeddings            : {CHILD_EMBEDDINGS_JSON}")
    print(f"threshold             : {task_config['threshold']}")
    print(f"debug_mode            : {task_config['debug_mode']}")
    print(f"best_match_per_image  : {task_config['best_match_per_image']}")
    print(f"max_results           : {task_config['max_results']}")
    print(f"top_k_neighbors       : {task_config['top_k_neighbors']}")
    print(f"recursive             : {task_config['recursive']}")
    print(f"extensions            : {task_config['extensions']}")
    print(f"min_face_size         : {task_config['min_face_size']}")
    print(f"max_face_size         : {task_config['max_face_size']}")
    print(f"nms_overlap_threshold : {task_config['nms_overlap_threshold']}")
    print(f"signature_size        : {task_config['signature_face_size']}")
    print(f"lbp_grid              : {task_config['lbp_grid']}")
    print("input_mode            : child_embeddings")
    print(f"host:port             : {HOST}:{PORT}")
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
        print(f"worker_results           : {len(normalized_results)}")
        print(f"wall_time                : {wall:.2f}s")

        combined_matches = []
        combined_debug = []
        total_errors = 0
        total_scanned = 0
        total_faces_detected_raw = 0
        total_faces_after_nms = 0

        for idx, result in enumerate(normalized_results, start=1):
            worker_id = result.get("worker_id")
            device_id = result.get("device_id") or f"unknown-device-{idx}"
            source_id = worker_id or device_id
            scanned = int(result.get("scanned_images") or 0)
            faces_raw = int(result.get("faces_detected_raw") or 0)
            faces_nms = int(result.get("faces_after_nms") or 0)
            matches = result.get("matches", [])
            debug_candidates = result.get("debug_candidates", [])
            errors = result.get("errors", [])
            elapsed = result.get("elapsed")

            total_scanned += scanned
            total_errors += len(errors)
            total_faces_detected_raw += faces_raw
            total_faces_after_nms += faces_nms

            print(
                f"[{idx}] worker={worker_id} device={device_id} "
                f"scanned={scanned} faces_raw={faces_raw} faces_nms={faces_nms} "
                f"matches={len(matches)} debug_candidates={len(debug_candidates)} "
                f"errors={len(errors)} elapsed={elapsed}s"
            )

            if errors:
                for err in errors:
                    print(f"      ERROR: {err}")

            for m in matches:
                if not isinstance(m, dict):
                    continue
                enriched = dict(m)
                enriched["source_device_id"] = source_id
                enriched["source_worker_id"] = worker_id
                combined_matches.append(enriched)

            for d in debug_candidates:
                if not isinstance(d, dict):
                    continue
                enriched = dict(d)
                enriched["source_device_id"] = source_id
                enriched["source_worker_id"] = worker_id
                combined_debug.append(enriched)

        combined_matches.sort(key=lambda m: float(m.get("similarity", 0.0)), reverse=True)
        combined_debug.sort(key=lambda m: float(m.get("similarity", 0.0)), reverse=True)

        print(f"\nTotal images scanned        : {total_scanned}")
        print(f"Total faces detected (raw)  : {total_faces_detected_raw}")
        print(f"Total faces after NMS       : {total_faces_after_nms}")
        print(f"Total matches above threshold: {len(combined_matches)}")

        payload = {
            "wall_time": round(wall, 3),
            "broadcast_target_devices": broadcast_target_workers,
            "worker_results": len(normalized_results),
            "total_scanned_images": total_scanned,
            "total_faces_detected_raw": total_faces_detected_raw,
            "total_faces_after_nms": total_faces_after_nms,
            "total_errors": total_errors,
            "results_by_device": normalized_results,
            "combined_matches": combined_matches[:MAX_RESULTS],
            "combined_debug_candidates": combined_debug[:20],
        }

        output_dir = os.path.dirname(CLIENT_OUTPUT_JSON)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(CLIENT_OUTPUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("-" * 64)
        print(f"Saved JSON -> {CLIENT_OUTPUT_JSON}")

        if combined_matches:
            print("\nTop matches (with device)")
            print("-" * 64)
            for m in combined_matches[: min(10, len(combined_matches))]:
                print(
                    f"  similarity={m.get('similarity'):>6}  "
                    f"[{m.get('source_device_id')}]  {m.get('image')}"
                )
        else:
            print("\nNo matches returned by any worker.")
            if combined_debug:
                print("\nDebug: Below-threshold candidates (highest similarity first)")
                print("-" * 64)
                for d in combined_debug[:10]:
                    print(
                        f"  similarity={d.get('similarity'):>6}  "
                        f"[{d.get('source_device_id')}]  {d.get('image')}"
                    )
                print(
                    f"\nHint: Best similarity seen was {combined_debug[0].get('similarity')}. "
                    f"Consider lowering THRESHOLD below this value."
                )
            elif total_faces_detected_raw == 0:
                print(
                    "\nHint: Zero faces were detected in any image. "
                    "Check that images contain clear, forward-facing faces "
                    "and consider lowering MIN_FACE_SIZE."
                )
    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())