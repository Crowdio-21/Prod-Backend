import sys
import os
import asyncio
import time
import math
import argparse
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import crowdio_connect, crowdio_disconnect, CROWDio, crowdio_map
from developer_sdk.image_utils import split_image_into_tiles, reassemble_tiles


@CROWDio.task(
    checkpoint=False,
    checkpoint_interval=3.0,
    checkpoint_state=["processed_count", "total_tiles", "avg_time", "progress"],
)
def process_tile(tile_data):
    from PIL import Image, ImageFilter, ImageEnhance
    import base64
    import io
    import time

    start_time = time.time()

    # Checkpoint state variables - framework handles resume automatically
    processed_count = 1
    total_tiles = 1
    avg_time = 0.0
    progress = 100.0

    image_bytes = base64.b64decode(tile_data["image"])
    img = Image.open(io.BytesIO(image_bytes))

    filter_type = tile_data.get("filter_type", "sharpen")

    if filter_type == "sharpen":
        img = img.filter(ImageFilter.SHARPEN)
    elif filter_type == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_type == "edge":
        img = img.filter(ImageFilter.FIND_EDGES)
    elif filter_type == "enhance":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
    elif filter_type == "grayscale":
        img = img.convert("L").convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    processed_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    processing_time = time.time() - start_time
    avg_time = processing_time
    progress = 100.0

    return {
        "tile_id": tile_data["tile_id"],
        "image": processed_image,
        "processing_time": processing_time,
        "filter_applied": filter_type,
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


async def process_image(image_path, tile_size, filter_type, output_dir):
    original_image = Image.open(image_path)
    tiles = split_image_into_tiles(original_image, tile_size=tile_size)
    if not tiles:
        raise ValueError("No tiles produced from image. Check image size or tile_size.")

    tile_inputs = [
        {
            "image": tile["image"],
            "filter_type": filter_type,
            "tile_id": tile["tile_id"],
            "position": tile["position"],
            "size": tile["size"],
        }
        for tile in tiles
    ]

    start_time = time.time()
    raw_results = await crowdio_map(process_tile, tile_inputs)
    total_time = time.time() - start_time

    import json as _json
    import ast as _ast

    processed_tiles = []
    for i, raw in enumerate(raw_results):
        if isinstance(raw, str):
            try:
                processed = _json.loads(raw)
            except _json.JSONDecodeError:
                processed = _ast.literal_eval(raw)
        else:
            processed = raw
        processed["position"] = tile_inputs[i]["position"]
        processed["size"] = tile_inputs[i]["size"]
        processed_tiles.append(processed)

    processing_times = [t["processing_time"] for t in processed_tiles]
    avg_tile_time = sum(processing_times) / len(processing_times)

    result_image = reassemble_tiles(processed_tiles, original_image.size)

    class_name = os.path.basename(os.path.dirname(image_path))
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join(
        output_class_dir, f"processed_{filter_type}_{base_name}.png"
    )
    result_image.save(result_path)

    return {
        "image": image_path,
        "output": result_path,
        "total_time": total_time,
        "avg_tile_time": avg_tile_time,
    }


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

    await crowdio_connect("localhost", 9000)
    try:
        print(
            f"Processing {len(selected_images)} of {total_images} images ({args.percent}%)."
        )
        for image_path in selected_images:
            stats = await process_image(
                image_path=image_path,
                tile_size=args.tile_size,
                filter_type=args.filter,
                output_dir=args.output_dir,
            )
            print(
                "Processed: {image} -> {output} (total_time={total_time:.2f}s, avg_tile={avg_tile_time:.4f}s)".format(
                    **stats
                )
            )
    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
