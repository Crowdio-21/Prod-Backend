#!/usr/bin/env python3
r"""Generate OpenCV face signatures from one or more reference images.

This script produces embeddings compatible with:
tests/image_processing/local_mobile_face_search_opencv.py

Method:
- Detect faces with OpenCV Haar cascade.
- Build a normalized DCT signature from the detected face ROI.
- Save signatures as list[list[float]] JSON.

Output format by default:
- A plain JSON list of embeddings, e.g. [[0.1, 0.2, ...], [0.3, 0.4, ...]]

Example:
  c:\Users\User\Prod-Backend\.venv\Scripts\python.exe \
        tests\image_processing\generate_face_embeddings_opencv.py \
        --images-dir "C:\Users\User\Prod-Backend\tests\image_processing\child" \
    --output "C:\Users\User\Prod-Backend\tests\image_processing\pipeline_output\child_embeddings.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenCV face-signature embeddings for child reference images."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to one reference image.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Path to a directory of reference images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path. The file will contain a list[list[float]].",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --images-dir for images.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=60,
        help="Minimum face size in pixels for Haar detection.",
    )
    parser.add_argument(
        "--signature-face-size",
        type=int,
        default=64,
        help="Face ROI size before DCT signature extraction.",
    )
    parser.add_argument(
        "--signature-grid",
        type=int,
        default=16,
        help="Top-left DCT grid size. Embedding length = grid * grid.",
    )
    parser.add_argument(
        "--all-faces",
        action="store_true",
        help="Store embeddings for all detected faces in each image.",
    )
    parser.add_argument(
        "--face-index",
        type=int,
        default=0,
        help="When --all-faces is not used, choose this face index from faces sorted by size.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Write a metadata object instead of plain list output.",
    )
    return parser.parse_args()


def collect_image_paths(image: str | None, images_dir: str | None, recursive: bool) -> list[str]:
    import glob

    paths: list[str] = []

    if image:
        paths.append(image)

    if images_dir:
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        if recursive:
            for pattern in patterns:
                paths.extend(glob.glob(os.path.join(images_dir, "**", pattern), recursive=True))
        else:
            for pattern in patterns:
                paths.extend(glob.glob(os.path.join(images_dir, pattern)))

    deduped = sorted(set(paths))
    return [path for path in deduped if os.path.isfile(path)]


def imread_unicode(path: str, cv2: Any, np: Any) -> Any:
    with open(path, "rb") as fh:
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def extract_face_signature(
    gray_image: Any,
    x: int,
    y: int,
    w: int,
    h: int,
    face_size: int,
    grid: int,
    cv2: Any,
    np: Any,
) -> list[float] | None:
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


def main() -> None:
    args = parse_args()

    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"OpenCV import failed. Install dependencies first: {exc}")

    if not args.image and not args.images_dir:
        raise SystemExit("Provide --image or --images-dir")

    if args.signature_face_size <= 0 or args.signature_grid <= 0:
        raise SystemExit("--signature-face-size and --signature-grid must be > 0")

    image_paths = collect_image_paths(args.image, args.images_dir, args.recursive)
    if not image_paths:
        raise SystemExit("No input images found")

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise SystemExit(f"Failed to load Haar cascade: {cascade_path}")

    embeddings: list[list[float]] = []
    records: list[dict[str, Any]] = []

    for path in image_paths:
        image = imread_unicode(path, cv2, np)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(args.min_face_size, args.min_face_size),
        )

        if len(faces) == 0:
            continue

        faces_sorted = sorted(
            [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces],
            key=lambda box: box[2] * box[3],
            reverse=True,
        )

        if args.all_faces:
            selected = list(enumerate(faces_sorted))
        else:
            idx = max(0, min(args.face_index, len(faces_sorted) - 1))
            selected = [(idx, faces_sorted[idx])]

        for face_id, (x, y, w, h) in selected:
            signature = extract_face_signature(
                gray,
                x,
                y,
                w,
                h,
                args.signature_face_size,
                args.signature_grid,
                cv2,
                np,
            )
            if signature is None:
                continue

            embeddings.append(signature)
            records.append(
                {
                    "image": path,
                    "face_id": int(face_id),
                    "face_box": [x, y, w, h],
                }
            )

    if not embeddings:
        raise SystemExit("No usable face signatures generated from the provided images")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    expected_dim = args.signature_grid * args.signature_grid

    if args.include_metadata:
        payload = {
            "method": "opencv_haar_dct",
            "images_input_count": len(image_paths),
            "face_count": len(embeddings),
            "embedding_dim": expected_dim,
            "min_face_size": args.min_face_size,
            "signature_face_size": args.signature_face_size,
            "signature_grid": args.signature_grid,
            "records": records,
            "embeddings": embeddings,
        }
    else:
        payload = embeddings

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Saved {len(embeddings)} embedding(s) -> {args.output}")
    print("method=opencv_haar_dct")
    print(f"embedding_dim={expected_dim}")
    print(f"images_scanned={len(image_paths)}")


if __name__ == "__main__":
    main()
