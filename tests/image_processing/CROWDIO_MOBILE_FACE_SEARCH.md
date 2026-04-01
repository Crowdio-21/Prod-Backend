# Crowdio Mobile Face Search (Chaquopy)

This guide adapts the local phone face-search idea to the Crowdio framework used in this repository.

## Goal

Run face search on the phone, while keeping developer code readable and path-safe:

- Server sends one or more child embeddings.
- Android app invokes Python via Chaquopy.
- Chaquopy Python submits work through Crowdio to the mobile worker.
- Worker scans local photos (or a prebuilt embedding index) and returns matches.

## Why This Fits Crowdio

The existing Crowdio mobile flow already supports symbolic path aliases and mobile runtime path resolution.

Use Crowdio constants instead of hardcoded Android paths:

- `crowdio.Constant.FILE_DIR`
- `crowdio.Constant.CACHE_DIR`
- `crowdio.Constant.OUTPUT_DIR`

The mobile runtime resolves aliases by injecting `builtins._crowdio_path_aliases` before task execution.

## End-to-End Flow

1. Server sends missing-child query payload with `child_embeddings`.
2. Android receives payload and forwards it to Chaquopy Python.
3. Chaquopy submits a Crowdio task via `distributed_map`.
4. Worker resolves symbolic paths and runs face matching on local photos/index.
5. Worker returns top matches and metadata to Android.

## Minimal Crowdio Task (Worker Side)

```python
from crowdio import crowdio


@crowdio.task(checkpoint=True, checkpoint_interval=5.0)
def scan_photos_for_child(config):
    import os
    import glob
    import builtins
    import numpy as np
    from deepface import DeepFace

    def resolve_path_alias(value):
        if not isinstance(value, str) or not value.startswith("@CROWDIO:"):
            return value
        alias_map = getattr(builtins, "_crowdio_path_aliases", {})
        return alias_map.get(value, value)

    def cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    photo_dir = resolve_path_alias(config.get("photo_dir"))
    child_embeddings = config.get("child_embeddings", [])
    threshold = float(config.get("threshold", 0.70))
    model_name = config.get("model_name", "ArcFace")
    detector_backend = config.get("detector_backend", "mtcnn")
    max_results = int(config.get("max_results", 100))

    if isinstance(photo_dir, str) and photo_dir.startswith("@CROWDIO:"):
        return {"matches": [], "errors": ["Unresolved path alias for photo_dir"]}

    matches = []
    errors = []
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(glob.glob(os.path.join(photo_dir, "**", pattern), recursive=True))

    for image_path in sorted(image_paths):
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector_backend,
                enforce_detection=False,
            )

            for face in faces:
                emb_obj = DeepFace.represent(
                    img_path=face["face"],
                    model_name=model_name,
                    enforce_detection=False,
                )
                face_embedding = emb_obj[0]["embedding"]

                best_score = 0.0
                for child_embedding in child_embeddings:
                    score = cosine_similarity(child_embedding, face_embedding)
                    if score > best_score:
                        best_score = score

                if best_score >= threshold:
                    matches.append({
                        "image": image_path,
                        "similarity": best_score,
                    })
        except Exception as exc:
            errors.append(f"{image_path}: {exc}")

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return {
        "matches": matches[:max_results],
        "errors": errors,
        "scanned_images": len(image_paths),
    }
```

## Chaquopy Client Submit (Crowdio SDK)

```python
import asyncio
from crowdio import connect, disconnect, crowdio, map as distributed_map


async def run_mobile_face_search(host, port, child_embeddings):
    await connect(host, port)
    try:
        task_config = {
            # No hardcoded Android path in user code.
            "photo_dir": crowdio.Constant.FILE_DIR,
            "child_embeddings": child_embeddings,
            "threshold": 0.70,
            "model_name": "ArcFace",
            "detector_backend": "mtcnn",
            "max_results": 50,
        }

        # Single mobile worker job here; scale to multiple configs if needed.
        results = await distributed_map(scan_photos_for_child, [task_config])
        return results[0]
    finally:
        await disconnect()


# asyncio.run(run_mobile_face_search("localhost", 9000, child_embeddings))
```

## Android Runtime Path Injection

On-device runtime should resolve symbolic aliases before task code runs, for example:

```python
import builtins

builtins._crowdio_path_aliases = {
    "@CROWDIO:FILE_DIR": "/storage/emulated/0/DCIM/Camera",
    "@CROWDIO:CACHE_DIR": "/storage/emulated/0/Android/data/<app>/cache",
    "@CROWDIO:OUTPUT_DIR": "/storage/emulated/0/Android/data/<app>/files",
}
```

This keeps app-level path ownership in Android while task code stays portable.

## Performance Upgrade: Build a Local Embedding Index

Repeated full image scanning is expensive. Preferred mobile pattern:

1. Background indexer detects new photos.
2. Extract face embeddings once.
3. Store in local index (SQLite or JSON under `crowdio.Constant.CACHE_DIR`).
4. Search query compares child embeddings against stored vectors only.

### Suggested index record schema

```json
{
  "image": "IMG_3492.jpg",
  "face_id": 0,
  "embedding": [0.12, -0.33, 0.81],
  "updated_at": 1710000000
}
```

## Fast Vector Search (Index Mode)

```python
import numpy as np


def search_embeddings(child_embeddings, db_embeddings, threshold=0.70):
    # child_embeddings: list[list[float]]
    # db_embeddings: dict[str, list[float]]  e.g. {"photo.jpg#0": [..]}
    results = []

    child_arr = [np.asarray(e, dtype=np.float32) for e in child_embeddings]
    for key, emb in db_embeddings.items():
        emb_vec = np.asarray(emb, dtype=np.float32)
        emb_norm = np.linalg.norm(emb_vec)
        if emb_norm == 0:
            continue

        best = 0.0
        for child in child_arr:
            denom = np.linalg.norm(child) * emb_norm
            if denom == 0:
                continue
            score = float(np.dot(child, emb_vec) / denom)
            if score > best:
                best = score

        if best >= threshold:
            results.append({"key": key, "similarity": best})

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results
```

## Recommended Practical Architecture

### Background index job on phone

1. New photo detected.
2. Detect face(s).
3. Compute embedding(s).
4. Save to local index.

### Query job from server

1. Receive `child_embeddings` list.
2. Compare against local index.
3. Return top matches (with similarity + photo ID/path).

## Real-World Note: Multiple Child Embeddings

Children's appearance can change quickly. Do not rely on one embedding.

Server payload should include multiple vectors:

```python
child_embeddings = [
    embedding_1,
    embedding_2,
    embedding_3,
    # ... usually 5-10 reference vectors
]
```

Task should score each candidate face against all vectors and keep the best score.

## Dependency Notes

- `deepface` and backend dependencies can be heavy on mobile.
- Validate model startup time and RAM on target devices.
- If needed, switch from direct image scan mode to index-only mode for production.

## Summary

With Crowdio, you can keep this workflow mobile-first and path-safe:

- No hardcoded Android paths in task/client code.
- Use `crowdio.Constant.*` aliases.
- Resolve aliases in mobile runtime.
- Run face search through Crowdio tasks.
- Move to embedding index mode for production speed.
