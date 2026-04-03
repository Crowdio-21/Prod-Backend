#!/usr/bin/env python3
r"""Generate DeepFace face embeddings from a local photo.

This script is meant for the Crowdio mobile face-search workflow:
- Use your own photo on the client side to generate reference embeddings.
- Save the embeddings as JSON.
- Pass that JSON into local_mobile_face_search.py via --child-embeddings-json.

Output format by default:
- A plain JSON list of embeddings, e.g. [[0.1, 0.2, ...], [0.3, 0.4, ...]]

Example:
  c:\Users\User\Prod-Backend\.venv\Scripts\python.exe \
    tests\image_processing\generate_face_embeddings.py \
    --image "C:\Users\User\Pictures\my_photo.jpg" \
    --output "C:\Users\User\Prod-Backend\tests\image_processing\pipeline_output\child_embeddings.json"

Then run:
  c:\Users\User\Prod-Backend\.venv\Scripts\python.exe \
    tests\image_processing\local_mobile_face_search.py \
    --use-alias --alias-file-dir "C:\photos" \
    --query-image "C:\Users\User\Pictures\my_photo.jpg" \
    --child-embeddings-json "C:\Users\User\Prod-Backend\tests\image_processing\pipeline_output\child_embeddings.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeepFace embeddings from one image.")
    parser.add_argument("--image", required=True, help="Path to the face photo to embed.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path. The file will contain a list[list[float]].",
    )
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
        "--all-faces",
        action="store_true",
        help="Store embeddings for every detected face in the image.",
    )
    parser.add_argument(
        "--face-index",
        type=int,
        default=0,
        help="When --all-faces is not used, choose this face index from the detected faces.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Write a metadata object instead of a plain list. Not recommended for the search script.",
    )
    return parser.parse_args()


def normalize_embedding(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    return []


def main() -> None:
    args = parse_args()

    try:
        from deepface import DeepFace  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"DeepFace import failed. Install dependencies first: {exc}")

    if not os.path.isfile(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    representation = DeepFace.represent(
        img_path=args.image,
        model_name=args.model_name,
        detector_backend=args.detector_backend,
        enforce_detection=False,
    )

    if not isinstance(representation, list) or not representation:
        raise SystemExit("No face embeddings were returned. Try a clearer photo or a different detector backend.")

    embeddings = []
    for item in representation:
        if isinstance(item, dict) and "embedding" in item:
            embedding = normalize_embedding(item["embedding"])
            if embedding:
                embeddings.append(embedding)

    if not embeddings:
        raise SystemExit("DeepFace returned no usable embeddings.")

    if not args.all_faces:
        face_index = max(0, min(args.face_index, len(embeddings) - 1))
        embeddings = [embeddings[face_index]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.include_metadata:
        payload = {
            "image": args.image,
            "model_name": args.model_name,
            "detector_backend": args.detector_backend,
            "face_count": len(embeddings),
            "embeddings": embeddings,
        }
    else:
        payload = embeddings

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Saved {len(embeddings)} embedding(s) -> {args.output}")
    print(f"model_name={args.model_name}")
    print(f"detector_backend={args.detector_backend}")


if __name__ == "__main__":
    main()
