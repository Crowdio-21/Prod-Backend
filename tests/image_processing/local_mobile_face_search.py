#!/usr/bin/env python3
"""Crowdio face search demo (local test + mobile-friendly path aliases).

What this is:
- A self-contained Crowdio demo you can run against Foreman + one Worker.
- Uses `crowdio.Constant.FILE_DIR` style aliases ("@CROWDIO:...") and the same
  `builtins._crowdio_path_aliases` pattern used in the existing image demos.

What this is NOT:
- Production-grade tuning. This uses DeepFace defaults and is intended as an
    end-to-end Crowdio mobile workflow reference.

Usage (local PC worker):
  1) Start foreman:  python tests/run_foreman_simple.py
  2) Start worker:   python tests/run_worker_simple.py
  3) Run this demo:
     python tests/image_processing/local_mobile_face_search.py \
       --image-dir "C:/path/to/photos" \
       --query-image "C:/path/to/query.jpg"

Usage (alias-style, similar to mobile):
  python tests/image_processing/local_mobile_face_search.py \
    --use-alias --alias-file-dir "C:/path/to/photos" --query-image "C:/path/to/query.jpg"

Notes:
- The worker must have `deepface` and dependencies installed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import connect, crowdio, disconnect, map as distributed_map


@crowdio.task(
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
        # Mobile runtime is expected to inject this BEFORE task execution.
        # For local testing, allow passing aliases in config.
        aliases = config.get("path_aliases")
        if not isinstance(aliases, dict):
            return
        existing = getattr(builtins, "_crowdio_path_aliases", None)
        if not isinstance(existing, dict):
            existing = {}
        existing.update({str(k): str(v) for k, v in aliases.items()})
        builtins._crowdio_path_aliases = existing

    def decode_query_image_b64(payload: str) -> np.ndarray:
        from io import BytesIO
        from PIL import Image

        raw = base64.b64decode(payload)
        image = Image.open(BytesIO(raw)).convert("RGB")
        return np.asarray(image)

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
    query_b64 = config.get("query_image_base64")
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

    if not isinstance(child_embeddings, list):
        child_embeddings = []

    if not child_embeddings and (not isinstance(query_b64, str) or not query_b64.strip()):
        return {
            "device_id": device_id,
            "scanned_images": 0,
            "matches": [],
            "errors": ["Provide child_embeddings or query_image_base64"],
        }

    if not child_embeddings:
        try:
            query_img = decode_query_image_b64(query_b64)
            query_repr = DeepFace.represent(
                img_path=query_img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
            child_embeddings = [r.get("embedding", []) for r in query_repr if isinstance(r, dict)]
        except Exception as exc:
            return {
                "device_id": device_id,
                "scanned_images": 0,
                "matches": [],
                "errors": [f"Failed to build child embedding(s) from query image: {exc}"],
            }

    child_embeddings = [emb for emb in child_embeddings if isinstance(emb, list) and emb]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crowdio face search demo (local/mobile alias-friendly).")

    parser.add_argument("--host", default="localhost", help="Foreman host.")
    parser.add_argument("--port", type=int, default=9000, help="Foreman port (WebSocket).")

    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory with photos to scan (local path). If omitted, use --use-alias.",
    )
    parser.add_argument(
        "--query-image",
        required=True,
        help="Local path to a query image containing a face.",
    )

    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold (0-1).")
    parser.add_argument("--max-results", type=int, default=20, help="Max matches to return.")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive scanning.")
    parser.add_argument(
        "--model-name",
        default="ArcFace",
        help="DeepFace model name (e.g. ArcFace, Facenet512, VGG-Face).",
    )
    parser.add_argument(
        "--detector-backend",
        default="mtcnn",
        help="DeepFace detector backend (e.g. mtcnn, retinaface, opencv).",
    )
    parser.add_argument(
        "--child-embeddings-json",
        default=None,
        help="Optional JSON file containing child_embeddings as list[list[float]].",
    )

    parser.add_argument(
        "--use-alias",
        action="store_true",
        help="Use crowdio.Constant.FILE_DIR in config (mobile-style).",
    )
    parser.add_argument(
        "--alias-file-dir",
        default=None,
        help="When using --use-alias, map @CROWDIO:FILE_DIR to this real path (for local testing).",
    )

    parser.add_argument(
        "--client-output-json",
        default=os.path.join(os.path.dirname(__file__), "pipeline_output", "face_search_result.json"),
        help="Where to save the returned result JSON on the client machine.",
    )

    return parser.parse_args()


def read_file_as_base64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


async def main() -> None:
    args = parse_args()

    if args.use_alias:
        image_dir_value = crowdio.Constant.FILE_DIR
        if not args.alias_file_dir:
            raise SystemExit("--use-alias requires --alias-file-dir")
        path_aliases = {crowdio.Constant.FILE_DIR: args.alias_file_dir}
    else:
        if not args.image_dir:
            raise SystemExit("Provide --image-dir or use --use-alias")
        image_dir_value = args.image_dir
        path_aliases = None

    child_embeddings: list[list[float]] = []
    if args.child_embeddings_json:
        with open(args.child_embeddings_json, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, list):
            child_embeddings = [row for row in loaded if isinstance(row, list)]

    task_config: dict[str, Any] = {
        "image_dir": image_dir_value,
        "threshold": args.threshold,
        "max_results": args.max_results,
        "recursive": not args.no_recursive,
        "model_name": args.model_name,
        "detector_backend": args.detector_backend,
    }

    if child_embeddings:
        task_config["child_embeddings"] = child_embeddings
    else:
        task_config["query_image_base64"] = read_file_as_base64(args.query_image)

    # For local testing, allow injecting alias map into worker.
    if path_aliases is not None:
        task_config["path_aliases"] = path_aliases

    print("\n" + "=" * 64)
    print("Crowdio Face Search Demo")
    print("=" * 64)
    print(f"image_dir      : {task_config['image_dir']}")
    print(f"query_image    : {args.query_image}")
    print(f"threshold      : {task_config['threshold']}")
    print(f"max_results    : {task_config['max_results']}")
    print(f"recursive      : {task_config['recursive']}")
    print(f"model_name     : {task_config['model_name']}")
    print(f"detector       : {task_config['detector_backend']}")
    print(f"input_mode     : {'child_embeddings' if child_embeddings else 'query_image'}")
    print(f"host:port      : {args.host}:{args.port}")
    print("=" * 64)

    await connect(args.host, args.port)
    try:
        started = time.time()
        results = await distributed_map(face_search_on_device, [task_config])
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

        os.makedirs(os.path.dirname(args.client_output_json), exist_ok=True)
        with open(args.client_output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print("-" * 64)
        print(f"Saved JSON -> {args.client_output_json}")

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
        await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
