import sys
import os
import asyncio
import time
import math
import argparse
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import connect, disconnect, crowdio, map as distributed_map
from developer_sdk.image_utils import reassemble_tiles


@crowdio.task(
    checkpoint=True,
    checkpoint_interval=3.0,
    checkpoint_state=[
        "processed_count",
        "total_tiles",
        "avg_time",
        "progress",
        "processed_results",
    ],
)
def process_image_worker(image_data):
    """
    Process all tiles of a single image inside a checkpoint-friendly loop.

    Each loop iteration processes one tile, so the framework can checkpoint
    after any tile and resume from where it left off.

    Args:
        image_data: dict with keys:
            - image_b64: base64-encoded full image
            - filter_type: filter to apply
            - tile_size: tile size in pixels
            - image_id: identifier for this image

    Returns:
        dict with original_size, filter_applied, and list of processed tiles
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import base64
    import io
    import time

    # ── Decode the full image ──────────────────────────────────────
    image_bytes = base64.b64decode(image_data["image_b64"])
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    tile_size = image_data.get("tile_size", 200)
    filter_type = image_data.get("filter_type", "sharpen")

    # ── Split into tiles locally ───────────────────────────────────
    tiles = []
    tile_id = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_w = min(tile_size, width - x)
            tile_h = min(tile_size, height - y)
            tiles.append(
                {
                    "tile_id": tile_id,
                    "box": (x, y, x + tile_w, y + tile_h),
                    "position": (x, y),
                    "size": (tile_w, tile_h),
                }
            )
            tile_id += 1

    # ── Checkpoint state variables ─────────────────────────────────
    total_tiles = len(tiles)
    processed_count = 0
    avg_time = 0.0
    progress = 0.0
    processed_results = []  # list of processed tile dicts

    cumulative_time = 0.0

    # ── Tile processing loop (checkpoint captures state each iter) ─
    for tile_info in tiles:
        start_time = time.time()

        # Crop the tile from the original image
        tile_img = img.crop(tile_info["box"])

        # Apply filter
        if filter_type == "sharpen":
            tile_img = tile_img.filter(ImageFilter.SHARPEN)
        elif filter_type == "blur":
            tile_img = tile_img.filter(ImageFilter.GaussianBlur(radius=2))
        elif filter_type == "edge":
            tile_img = tile_img.filter(ImageFilter.FIND_EDGES)
        elif filter_type == "enhance":
            enhancer = ImageEnhance.Contrast(tile_img)
            tile_img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Brightness(tile_img)
            tile_img = enhancer.enhance(1.2)
        elif filter_type == "grayscale":
            tile_img = tile_img.convert("L").convert("RGB")

        # Encode processed tile
        buf = io.BytesIO()
        tile_img.save(buf, format="PNG")
        tile_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        processing_time = time.time() - start_time
        cumulative_time += processing_time

        # Update checkpoint state
        processed_count += 1
        avg_time = cumulative_time / processed_count
        progress = (processed_count / total_tiles) * 100.0

        processed_results.append(
            {
                "tile_id": tile_info["tile_id"],
                "image": tile_b64,
                "position": tile_info["position"],
                "size": tile_info["size"],
                "processing_time": processing_time,
                "filter_applied": filter_type,
            }
        )

    return {
        "image_id": image_data.get("image_id", "unknown"),
        "original_size": (width, height),
        "filter_applied": filter_type,
        "total_tiles": total_tiles,
        "tiles": processed_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a percentage of dataset images."
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(os.path.dirname(__file__), "archive", "dataset"),
        help="Path to dataset root containing class folders.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=1.0,
        help="Percentage of images to process (1 means 1%%).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels.",
    )
    parser.add_argument(
        "--filter",
        default="sharpen",
        help="Filter type to apply: sharpen, enhance, blur, grayscale, edge.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "archive", "output"),
        help="Output directory for processed images.",
    )
    return parser.parse_args()


def collect_images(dataset_dir):
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            lower = filename.lower()
            if lower.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, filename))
    image_paths.sort()
    return image_paths


async def process_images(image_paths, tile_size, filter_type, output_dir):
    """Send all selected images as tasks and reassemble results."""
    import base64
    import json as _json
    import ast as _ast

    # Encode each image as base64 and prepare task inputs
    task_inputs = []
    for idx, image_path in enumerate(image_paths):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        task_inputs.append(
            {
                "image_b64": image_b64,
                "filter_type": filter_type,
                "tile_size": tile_size,
                "image_id": idx,
            }
        )

    start_time = time.time()
    raw_results = await distributed_map(process_image_worker, task_inputs)
    total_time = time.time() - start_time

    # Parse and save each result
    for idx, raw in enumerate(raw_results):
        if isinstance(raw, str):
            try:
                result = _json.loads(raw)
            except _json.JSONDecodeError:
                result = _ast.literal_eval(raw)
        else:
            result = raw

        image_path = image_paths[idx]
        original_size = tuple(result["original_size"])
        tiles = result["tiles"]

        result_image = reassemble_tiles(tiles, original_size)

        class_name = os.path.basename(os.path.dirname(image_path))
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(
            output_class_dir, f"processed_{filter_type}_{base_name}.png"
        )
        result_image.save(result_path)

        tile_times = [t["processing_time"] for t in tiles]
        avg_tile = sum(tile_times) / len(tile_times) if tile_times else 0

        print(
            f"  Processed: {image_path} -> {result_path} "
            f"({len(tiles)} tiles, avg_tile={avg_tile:.4f}s)"
        )

    print(f"\nTotal wall time: {total_time:.2f}s for {len(image_paths)} images")


async def main():
    args = parse_args()

    if args.percent <= 0 or args.percent > 100:
        raise ValueError("--percent must be in the range (0, 100].")

    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    image_paths = collect_images(args.dataset_dir)
    if not image_paths:
        raise ValueError("No images found in dataset directory.")

    total_images = len(image_paths)
    num_to_process = max(1, int(math.ceil(total_images * (args.percent / 100.0))))
    selected_images = image_paths[:num_to_process]

    await connect("localhost", 9000)
    try:
        print(
            f"Processing {len(selected_images)} of {total_images} images ({args.percent}%).\n"
        )
        await process_images(
            image_paths=selected_images,
            tile_size=args.tile_size,
            filter_type=args.filter,
            output_dir=args.output_dir,
        )
    finally:
        await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
