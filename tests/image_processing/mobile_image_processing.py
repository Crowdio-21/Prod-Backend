#!/usr/bin/env python3
"""
Simple mobile image processing example using Crowdio constants.

Goal:
- Developers write readable config using crowdio.Constant.FILE_DIR.
- Mobile runtime resolves constants to real device paths before execution.

Expected mobile runtime behavior:
- Inject a dict into builtins._crowdio_path_aliases, for example:
  {
      "@CROWDIO:FILE_DIR": "/storage/emulated/0/MyPickedFolder",
      "@CROWDIO:OUTPUT_DIR": "/storage/emulated/0/CrowdioOutput"
  }
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import connect, disconnect, crowdio, map as distributed_map


@crowdio.task(
    checkpoint=True, checkpoint_interval=3.0, checkpoint_state=["processed", "errors"]
)
def process_images_on_device(config):
    """Run on worker. Reads images from a local device folder and applies one filter."""
    import glob
    import builtins
    import platform
    from PIL import Image, ImageEnhance, ImageFilter

    def resolve_path_alias(value):
        if not isinstance(value, str) or not value.startswith("@CROWDIO:"):
            return value
        alias_map = getattr(builtins, "_crowdio_path_aliases", {})
        return alias_map.get(value, value)

    def apply_filter(img, filter_name):
        if filter_name == "sharpen":
            return img.filter(ImageFilter.SHARPEN)
        if filter_name == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=2))
        if filter_name == "edge":
            return img.filter(ImageFilter.FIND_EDGES)
        if filter_name == "enhance":
            contrast = ImageEnhance.Contrast(img).enhance(1.4)
            return ImageEnhance.Brightness(contrast).enhance(1.15)
        if filter_name == "grayscale":
            return img.convert("L").convert("RGB")
        return img

    device_id = platform.node() or "unknown-device"

    image_dir = resolve_path_alias(config.get("image_dir"))
    output_dir = resolve_path_alias(config.get("output_dir"))
    filter_name = config.get("filter", "sharpen")
    max_images = int(config.get("max_images", 0))

    if isinstance(image_dir, str) and image_dir.startswith("@CROWDIO:"):
        return {
            "device_id": device_id,
            "processed": 0,
            "errors": [
                "Path alias was not resolved on worker. "
                "Ensure mobile runtime injects builtins._crowdio_path_aliases."
            ],
        }

    if not image_dir or not os.path.isdir(image_dir):
        return {
            "device_id": device_id,
            "processed": 0,
            "errors": [f"Image directory not found: {image_dir}"],
        }

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_paths.sort()

    if max_images > 0:
        image_paths = image_paths[:max_images]

    if not image_paths:
        return {
            "device_id": device_id,
            "processed": 0,
            "errors": [f"No images found in: {image_dir}"],
        }

    if output_dir and not output_dir.startswith("@CROWDIO:"):
        os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    processed = 0
    errors = []

    for src_path in image_paths:
        try:
            image = Image.open(src_path).convert("RGB")
            result = apply_filter(image, filter_name)

            if output_dir and not output_dir.startswith("@CROWDIO:"):
                name, _ = os.path.splitext(os.path.basename(src_path))
                dst_path = os.path.join(output_dir, f"{name}_{filter_name}.png")
                result.save(dst_path, format="PNG")

            processed += 1
        except Exception as exc:
            errors.append(f"{src_path}: {exc}")

    return {
        "device_id": device_id,
        "processed": processed,
        "errors": errors,
        "filter": filter_name,
        "elapsed": round(time.time() - start, 3),
        "image_dir": image_dir,
        "output_dir": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple mobile image processing using crowdio.Constant path aliases."
    )
    parser.add_argument(
        "filter",
        nargs="?",
        default="sharpen",
        choices=["sharpen", "blur", "edge", "enhance", "grayscale"],
        help="Filter to apply.",
    )
    parser.add_argument(
        "--max-images", type=int, default=10, help="Max images to process."
    )
    parser.add_argument("--host", default="localhost", help="Foreman host.")
    parser.add_argument("--port", type=int, default=9000, help="Foreman port.")
    parser.add_argument(
        "--output-dir",
        default=crowdio.Constant.OUTPUT_DIR,
        help="Output dir alias/path (default: crowdio.Constant.OUTPUT_DIR).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Developer-friendly config: no hardcoded Android paths.
    # Mobile runtime can resolve crowdio.Constant.FILE_DIR to selected folder path.
    task_config = {
        "image_dir": crowdio.Constant.FILE_DIR,
        "output_dir": args.output_dir,
        "filter": args.filter,
        "max_images": args.max_images,
    }

    print("\n" + "=" * 64)
    print("Mobile Image Processing (Constant Alias Example)")
    print("=" * 64)
    print(f"image_dir alias : {task_config['image_dir']}")
    print(f"output_dir      : {task_config['output_dir']}")
    print(f"filter          : {task_config['filter']}")
    print(f"max_images      : {task_config['max_images']}")
    print("=" * 64)

    await connect(args.host, args.port)
    try:
        started = time.time()
        results = await distributed_map(process_images_on_device, [task_config])
        wall_time = time.time() - started

        print("\nResults")
        print("-" * 64)
        for idx, result in enumerate(results):
            print(f"Worker {idx}: {result}")
        print("-" * 64)
        print(f"Wall time: {wall_time:.2f}s")
    finally:
        await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
