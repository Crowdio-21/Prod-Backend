
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import crowdio_connect, crowdio_disconnect, CROWDio, CROWDio_map


# =============================================================
# Editable settings (no command-line arguments required)
# =============================================================
HOST = "localhost"
PORT = 9000

# Set True to use crowdio.Constant.FILE_DIR in task config.
# In this mode, ALIAS_FILE_DIR must point to the real folder path.
USE_ALIAS = True
ALIAS_FILE_DIR = r"C:\Users\User\Prod-Backend\tests\image_processing\pipeline_output\child_embeddings.json"

# Used only when USE_ALIAS is False.
IMAGE_DIR = r"C:\path\to\photos"

# Required input: JSON file containing list[list[float]] embeddings.
CHILD_EMBEDDINGS_JSON = os.path.join(
    os.path.dirname(__file__), "pipeline_output", "child_embeddings.json"
)

THRESHOLD = 0.75
MAX_RESULTS = 20
RECURSIVE = True
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"

CLIENT_OUTPUT_JSON = os.path.join(
    os.path.dirname(__file__), "pipeline_output", "face_search_result.json"
)


@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["scanned_images", "matches", "errors"],
)
def face_search_on_device(config: dict[str, Any]) -> dict[str, Any]:
    """Run on a worker: scan images, detect faces, and match to query face."""

    import builtins
    import glob
    import platform

    import numpy as np

    try:
        from deepface import DeepFace  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {
            "device_id": platform.node() or "unknown-device",
            "scanned_images": 0,
            "matches": [],
            "errors": [
                "DeepFace import failed on worker.",
                f"Install dependency: deepface ({exc})",
            ],
        }

    def resolve_path_alias(value: Any) -> Any:
        if not isinstance(value, str) or not value.startswith("@CROWDIO:"):
            return value
        alias_map = getattr(builtins, "_crowdio_path_aliases", {})
        return alias_map.get(value, value)

    def maybe_inject_aliases() -> None:
        # Mobile runtime should inject aliases automatically.
        # For local testing, allow alias map from task config.
        aliases = config.get("path_aliases")
        if not isinstance(aliases, dict):
            return
        existing = getattr(builtins, "_crowdio_path_aliases", None)
        if not isinstance(existing, dict):
            existing = {}
        existing.update({str(k): str(v) for k, v in aliases.items()})
        builtins._crowdio_path_aliases = existing

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        av = np.asarray(a, dtype=np.float32)
        bv = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 0:
            return 0.0
        return float(np.dot(av, bv) / denom)

    maybe_inject_aliases()


    device_id = platform.node() or "unknown-device"

    image_dir = resolve_path_alias(config.get("image_dir"))
    child_embeddings = config.get("child_embeddings", [])
    threshold = float(config.get("threshold", 0.75))
    max_results = int(config.get("max_results", 20))
    recursive = bool(config.get("recursive", True))
    model_name = str(config.get("model_name", "ArcFace"))
    detector_backend = str(config.get("detector_backend", "mtcnn"))

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

    normalized_embeddings: list[list[float]] = []
    for emb in child_embeddings:
        if not isinstance(emb, list) or not emb:
            continue
        try:
            normalized_embeddings.append([float(x) for x in emb])
        except Exception:
            continue

    child_embeddings = normalized_embeddings
    if not child_embeddings:
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": ["No valid child embeddings available"],
        }

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths: list[str] = []
    if recursive:
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(image_dir, "**", pattern), recursive=True))
    else:
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_paths.sort()

    start = time.time()
    errors: list[str] = []
    matches: list[dict[str, Any]] = []

    for path in image_paths:
        try:
            faces = DeepFace.extract_faces(
                img_path=path,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
            for face_index, face in enumerate(faces):
                face_img = face.get("face") if isinstance(face, dict) else None
                if face_img is None:
                    continue

                repr_result = DeepFace.represent(
                    img_path=face_img,
                    model_name=model_name,
                    detector_backend="skip",
                    enforce_detection=False,
                )
                if not repr_result:
                    continue
                face_embedding = repr_result[0].get("embedding", [])
                if not face_embedding:
                    continue

                best_score = 0.0
                for child_embedding in child_embeddings:
                    score = cosine_similarity(child_embedding, face_embedding)
                    if score > best_score:
                        best_score = score

                if best_score >= threshold:
                    matches.append(
                        {
                            "image": path,
                            "face_id": face_index,
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
        "model_name": model_name,
        "detector_backend": detector_backend,
    }


async def main() -> None:
    if USE_ALIAS:
        image_dir_value = CROWDio.Constant.FILE_DIR
        if not ALIAS_FILE_DIR:
            raise SystemExit("Set ALIAS_FILE_DIR when USE_ALIAS is True")
        path_aliases = {CROWDio.Constant.FILE_DIR: ALIAS_FILE_DIR}
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

    child_embeddings: list[list[float]] = []
    for row in loaded:
        if not isinstance(row, list) or not row:
            continue
        try:
            child_embeddings.append([float(x) for x in row])
        except Exception:
            continue

    if not child_embeddings:
        raise SystemExit("No valid child embeddings found in CHILD_EMBEDDINGS_JSON")

    task_config: dict[str, Any] = {
        "image_dir": image_dir_value,
        "child_embeddings": child_embeddings,
        "threshold": THRESHOLD,
        "max_results": MAX_RESULTS,
        "recursive": RECURSIVE,
        "model_name": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
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
    print(f"model_name     : {task_config['model_name']}")
    print(f"detector       : {task_config['detector_backend']}")
    print("input_mode     : child_embeddings")
    print(f"host:port      : {HOST}:{PORT}")
    print("=" * 64)

    await crowdio_connect(HOST, PORT)
    try:
        started = time.time()
        results = await CROWDio_map(face_search_on_device, [task_config])
        wall = time.time() - started

        result = results[0] if results else {}
        print("\nResult")
        print("-" * 64)
        print(f"device_id      : {result.get('device_id')}")
        print(f"scanned_images : {result.get('scanned_images')}")
        print(f"matches        : {len(result.get('matches', []))}")
        print(f"errors         : {len(result.get('errors', []))}")
        print(f"elapsed(worker): {result.get('elapsed')}s")
        print(f"wall_time      : {wall:.2f}s")

        os.makedirs(os.path.dirname(CLIENT_OUTPUT_JSON), exist_ok=True)
        with open(CLIENT_OUTPUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print("-" * 64)
        print(f"Saved JSON -> {CLIENT_OUTPUT_JSON}")

        # Print top matches
        matches = result.get("matches", [])
        if matches:
            print("\nTop matches")
            print("-" * 64)
            for m in matches[: min(10, len(matches))]:
                print(f"{m.get('similarity'):>6}  {m.get('image')}")
        else:
            errs = result.get("errors", [])
            if errs:
                print("\nWorker errors")
                print("-" * 64)
                for e in errs[:10]:
                    print(e)
    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
