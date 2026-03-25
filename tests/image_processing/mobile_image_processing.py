#!/usr/bin/env python3
"""
Mobile On-Device Image Processing
===================================

This test sends only the **processing code** to workers (phones / PCs).
Workers use images that already exist on their local filesystem — nothing
is transferred over the network except the function code and a small
config dict telling the worker which directory to scan and what filter
to apply.

This is ideal for:
  * Phones with a local camera roll / gallery
  * Edge devices with pre-downloaded datasets
  * Any scenario where you want to avoid sending large image data

Usage:
    # Start foreman + workers, then:
    uv run python tests/image_processing/mobile_image_processing.py

    # Specify a remote directory that exists ON THE WORKERS:
    uv run python tests/image_processing/mobile_image_processing.py \\
        --remote-dir /sdcard/DCIM/Camera \\
        --filter enhance --max-images 10

    # Process a single known path on the worker:
    uv run python tests/image_processing/mobile_image_processing.py \\
        --remote-image /sdcard/Download/photo.jpg --filter sharpen

    # Workers write output to a directory on their device:
    uv run python tests/image_processing/mobile_image_processing.py \\
        --remote-dir /sdcard/Pictures \\
        --output-dir /sdcard/CROWDio_output \\
        --filter grayscale --max-images 20

How it works:
    1. The client submits a ``map()`` job where each task arg is a small
       config dict (no image bytes).
    2. The function code is serialised and sent to the worker.
    3. The worker executes the function locally — reading images from its
       own filesystem, processing them, and writing results to disk.
    4. The worker returns a lightweight summary (paths, stats, thumbnails).
"""

import sys
import os
import asyncio
import time
import argparse

# Add project root to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import crowdio_connect, crowdio_disconnect, CROWDio, crowdio_map


# =====================================================================
# Worker function — runs entirely on the device
# =====================================================================

@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=3.0,
    checkpoint_state=[
        "image_paths",
        "results",
        "errors",
        "thumbnails",
    ],
)
def process_images_on_device(config):
    """
    Scan a directory on the LOCAL device for images, apply a filter,
    and write results to an output directory — all on-device.

    Args:
        config: dict with keys:
            - image_dir: str — directory on the device to scan for images
            - image_path: str — OR a single image path (if provided, image_dir is ignored)
            - output_dir: str — where to write processed images (created if missing)
            - filter_type: str — filter to apply (sharpen, blur, edge, enhance, grayscale, emboss, smooth, contour)
            - max_images: int — max images to process (0 = all)
            - tile_size: int — split images into tiles of this size before filtering (0 = no tiling)
            - create_thumbnail: bool — if True, return a small base64 thumbnail in the result

    Returns:
        dict with:
            - device_id: worker identifier (hostname)
            - processed: list of per-image results (path, size, time)
            - total_images: int
            - total_time: float
            - errors: list of error strings
            - thumbnails: optional list of base64-encoded thumbnails
    """
    import os
    import platform
    import traceback as _tb

    device_id = platform.node() or "unknown-device"

    try:
        import glob
        import time as _time
        from PIL import Image, ImageFilter, ImageEnhance
        import builtins, io
        import base64

        filter_type = config.get("filter_type", "sharpen")
        output_dir = config.get("output_dir", "")
        max_images = config.get("max_images", 0)
        tile_size = config.get("tile_size", 0)
        create_thumbnail = config.get("create_thumbnail", True)

        # --- Collect image paths ---
        # First check for paths injected by the Android runtime (gallery picker)
        image_paths = getattr(builtins, '_selected_images', [])

        # Fall back to config-based paths if no Android-injected images
        if not image_paths and config.get("image_path"):
            p = config["image_path"]
            if os.path.isfile(p):
                image_paths = [p]
            else:
                return {
                    "device_id": device_id,
                    "processed": [],
                    "total_images": 0,
                    "total_time": 0,
                    "errors": [f"File not found: {p}"],
                }
        elif not image_paths and config.get("image_dir"):
            scan_dir = config["image_dir"]
            if not os.path.isdir(scan_dir):
                return {
                    "device_id": device_id,
                    "processed": [],
                    "total_images": 0,
                    "total_time": 0,
                    "errors": [f"Directory not found: {scan_dir}"],
                }
            extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
            for ext in extensions:
                image_paths.extend(glob.glob(os.path.join(scan_dir, "**", ext), recursive=True))
            image_paths.sort()
        elif not image_paths:
            return {
                "device_id": device_id,
                "processed": [],
                "total_images": 0,
                "total_time": 0,
                "errors": ["No image_dir, image_path, or Android-injected images found"],
            }

        if max_images > 0:
            image_paths = image_paths[:max_images]

        if not image_paths:
            return {
                "device_id": device_id,
                "processed": [],
                "total_images": 0,
                "total_time": 0,
                "errors": ["No images found"],
            }

        # --- Prepare output ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # --- Filter helper ---
        def apply_filter(img, ftype):
            if ftype == "sharpen":
                return img.filter(ImageFilter.SHARPEN)
            elif ftype == "blur":
                return img.filter(ImageFilter.GaussianBlur(radius=2))
            elif ftype == "edge":
                return img.filter(ImageFilter.FIND_EDGES)
            elif ftype == "enhance":
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
                enhancer = ImageEnhance.Brightness(img)
                return enhancer.enhance(1.2)
            elif ftype == "grayscale":
                return img.convert("L").convert("RGB")
            elif ftype == "emboss":
                return img.filter(ImageFilter.EMBOSS)
            elif ftype == "smooth":
                return img.filter(ImageFilter.SMOOTH_MORE)
            elif ftype == "contour":
                return img.filter(ImageFilter.CONTOUR)
            return img

        # --- Process each image ---
        overall_start = _time.time()
        results = []
        errors = []
        thumbnails = []

        for img_path in image_paths:
            img_start = _time.time()
            try:
                img = Image.open(img_path).convert("RGB")
                width, height = img.size

                if tile_size > 0:
                    # Process tile by tile then reassemble
                    result_img = Image.new("RGB", (width, height))
                    for y in range(0, height, tile_size):
                        for x in range(0, width, tile_size):
                            tw = min(tile_size, width - x)
                            th = min(tile_size, height - y)
                            tile = img.crop((x, y, x + tw, y + th))
                            tile = apply_filter(tile, filter_type)
                            result_img.paste(tile, (x, y))
                else:
                    result_img = apply_filter(img, filter_type)

                # Save output
                basename = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"{basename}_{filter_type}.png"
                if output_dir:
                    out_path = os.path.join(output_dir, out_name)
                    result_img.save(out_path, format="PNG")
                else:
                    out_path = None

                elapsed = _time.time() - img_start

                # Optional thumbnail (128px wide, base64)
                if create_thumbnail:
                    thumb = result_img.copy()
                    thumb.thumbnail((128, 128))
                    buf = io.BytesIO()
                    thumb.save(buf, format="JPEG", quality=60)
                    thumbnails.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

                results.append({
                    "source_path": img_path,
                    "output_path": out_path,
                    "size": [width, height],
                    "filter": filter_type,
                    "time": round(elapsed, 3),
                })

                print(f"[on-device] {os.path.basename(img_path)} {width}x{height} "
                      f"filter={filter_type} time={elapsed:.3f}s")

            except Exception as e:
                errors.append(f"{img_path}: {str(e)}")
                print(f"[on-device] ERROR {os.path.basename(img_path)}: {e}")

        total_time = _time.time() - overall_start

        return {
            "device_id": device_id,
            "processed": results,
            "total_images": len(results),
            "total_time": round(total_time, 3),
            "errors": errors,
            "thumbnails": thumbnails if create_thumbnail else [],
        }

    except Exception as _e:
        return {
            "device_id": device_id,
            "processed": [],
            "total_images": 0,
            "total_time": 0,
            "errors": [f"FATAL: {_e}"],
            "thumbnails": [],
        }


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Send image-processing code to mobile workers. Workers process images selected via the gallery picker.",
    )

    parser.add_argument(
        "filter",
        nargs="?",
        default="sharpen",
        choices=[
            "sharpen", "blur", "edge", "enhance",
            "grayscale", "emboss", "smooth", "contour",
        ],
        help="Filter to apply (default: sharpen).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images per worker (0 = all selected).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory on the worker device (default: results only, no file save).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Foreman host (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Foreman WebSocket port (default: 9000).",
    )
    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

async def main():
    args = parse_args()

    # Config sent to the worker — images come from the phone's gallery picker
    config = {
        "filter_type": args.filter,
        "output_dir": args.output_dir,
        "max_images": args.max_images,
        "tile_size": 0,
        "create_thumbnail": True,
    }

    task_args = [config]

    print(f"\n{'='*60}")
    print(f"Mobile On-Device Image Processing")
    print(f"{'='*60}")
    print(f"  Filter          : {args.filter}")
    print(f"  Max images      : {args.max_images or 'all selected'}")
    print(f"  Output dir      : {args.output_dir or '(none — results only)'}")
    print(f"{'='*60}\n")

    # Connect and distribute
    await crowdio_connect(args.host, args.port)

    try:
        start = time.time()
        results = await crowdio_map(process_images_on_device, task_args)
        elapsed = time.time() - start

        # --- Summary ---
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")

        total_processed = 0
        total_errors = 0

        for i, res in enumerate(results):
            # Parse result if it's a string
            if isinstance(res, str):
                import json, ast
                raw = res  # keep original for debug
                parsed = None
                try:
                    parsed = json.loads(res)
                except Exception:
                    try:
                        parsed = ast.literal_eval(res)
                    except Exception:
                        pass
                if parsed is None:
                    print(f"\n  Worker {i}: could not parse result")
                    print(f"    Raw ({len(raw)} chars): {raw[:500]!r}")
                    total_errors += 1
                    continue
                res = parsed
            elif not isinstance(res, dict):
                print(f"\n  Worker {i}: unexpected result type {type(res).__name__}")
                print(f"    Value: {str(res)[:500]!r}")
                total_errors += 1
                continue

            device = res.get("device_id", "?")
            n = res.get("total_images", 0)
            t = res.get("total_time", 0)
            errs = res.get("errors", [])
            total_processed += n
            total_errors += len(errs)

            print(f"\n  Worker {i} ({device}):")
            print(f"    Images processed : {n}")
            print(f"    Time on device   : {t:.3f}s")
            if errs:
                print(f"    Errors           : {len(errs)}")
                for e in errs[:3]:
                    print(f"      - {e}")
            if n > 0:
                per_image = [p["time"] for p in res.get("processed", [])]
                avg = sum(per_image) / len(per_image) if per_image else 0
                print(f"    Avg per image    : {avg:.3f}s")

            # Save thumbnails locally if present
            thumbs = res.get("thumbnails", [])
            if thumbs:
                thumb_dir = os.path.join(
                    os.path.dirname(__file__), "pipeline_output", "thumbnails"
                )
                os.makedirs(thumb_dir, exist_ok=True)
                import base64
                for j, t64 in enumerate(thumbs):
                    path = os.path.join(thumb_dir, f"worker{i}_thumb_{j}.jpg")
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(t64))
                print(f"    Thumbnails saved : {len(thumbs)} → {thumb_dir}")

        print(f"\n  {'─'*40}")
        print(f"  Total images processed : {total_processed}")
        print(f"  Total errors           : {total_errors}")
        print(f"  Wall time              : {elapsed:.2f}s")
        print(f"{'='*60}")

    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
