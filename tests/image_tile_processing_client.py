import sys
import os
import asyncio
import time
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from developer_sdk import CrowdioClient

crowdio = CrowdioClient()


@crowdio.remote
def process_tile(tile_data):
    """
    Process an image tile remotely.
    Applies various image transformations to demonstrate distributed processing.

    Args:
        tile_data: Dictionary containing:
            - 'image': base64 encoded image data
            - 'filter_type': type of filter to apply
            - 'tile_id': identifier for this tile

    Returns:
        Dictionary with processed tile data and metadata
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import base64
    import io
    import time

    start_time = time.time()

    # Decode the image
    image_bytes = base64.b64decode(tile_data["image"])
    img = Image.open(io.BytesIO(image_bytes))

    # Apply the specified filter
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

    # Encode the processed image back to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    processed_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    processing_time = time.time() - start_time

    return {
        "tile_id": tile_data["tile_id"],
        "image": processed_image,
        "processing_time": processing_time,
        "filter_applied": filter_type,
    }


def split_image_into_tiles(image, tile_size=200):
    """
    Split an image into tiles of specified size

    Args:
        image: PIL Image object
        tile_size: Size of each tile (square)

    Returns:
        List of dictionaries containing tile data and position info
    """
    from PIL import Image
    import base64
    import io

    width, height = image.size
    tiles = []
    tile_id = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Calculate tile dimensions (handle edge cases)
            tile_width = min(tile_size, width - x)
            tile_height = min(tile_size, height - y)

            # Crop the tile
            tile = image.crop((x, y, x + tile_width, y + tile_height))

            # Encode to base64
            buffer = io.BytesIO()
            tile.save(buffer, format="PNG")
            tile_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            tiles.append(
                {
                    "tile_id": tile_id,
                    "image": tile_b64,
                    "position": (x, y),
                    "size": (tile_width, tile_height),
                }
            )
            tile_id += 1

    return tiles


def reassemble_tiles(processed_tiles, original_size):
    """
    Reassemble processed tiles back into a complete image

    Args:
        processed_tiles: List of processed tile dictionaries
        original_size: Tuple of (width, height) of original image

    Returns:
        PIL Image object
    """
    from PIL import Image
    import base64
    import io

    # Create a new blank image
    result = Image.new("RGB", original_size)

    # Sort tiles by tile_id to ensure correct ordering
    sorted_tiles = sorted(processed_tiles, key=lambda t: t["tile_id"])

    for tile_data in sorted_tiles:
        # Decode the tile
        image_bytes = base64.b64decode(tile_data["image"])
        tile_img = Image.open(io.BytesIO(image_bytes))

        # Paste it at the correct position
        position = tile_data["position"]
        result.paste(tile_img, position)

    return result


async def main():
    """Main function demonstrating distributed image tile processing"""

    try:

        foreman_host = "localhost"
        await crowdio.init(foreman_host, 9000)

        image_path = "image.png"  

        print(f"\n📷 Loading image from: {image_path}...")
        original_image = Image.open(image_path)
        print(
            f"✓ Image loaded: {original_image.size[0]}x{original_image.size[1]} pixels"
        )

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        original_path = os.path.join(output_dir, "original.png")
        original_image.save(original_path)
        print(f"✓ Original image saved to: {original_path}")

        tile_size = 200
        print(f"\n🔪 Splitting image into {tile_size}x{tile_size} tiles...")
        tiles = split_image_into_tiles(original_image, tile_size=tile_size)
        print(f"✓ Created {len(tiles)} tiles")

        # filter_types = ["sharpen", "enhance", "blur", "grayscale", "edge"]
        filter_types = ["sharpen"]

        for filter_type in filter_types:
            print(f"\n{'=' * 70}")
            print(f"Processing with filter: {filter_type.upper()}")
            print(f"{'=' * 70}")

            # Prepare tile data with filter type
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

            # Distribute processing across workers
            print(f"🚀 Distributing {len(tile_inputs)} tiles to workers...")
            start_time = time.time()

            processed_tiles = await process_tile.map(tile_inputs)

            end_time = time.time()
            total_time = end_time - start_time

            # Add position and size info back to processed tiles
            for i, processed in enumerate(processed_tiles):
                processed["position"] = tile_inputs[i]["position"]
                processed["size"] = tile_inputs[i]["size"]

            # Calculate statistics
            processing_times = [t["processing_time"] for t in processed_tiles]
            avg_tile_time = sum(processing_times) / len(processing_times)

            print(f"\n📊 Processing Statistics:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average time per tile: {avg_tile_time:.4f}s")
            print(f"   Speedup factor: {sum(processing_times) / total_time:.2f}x")

            print(f"\n🔧 Reassembling tiles...")
            result_image = reassemble_tiles(processed_tiles, original_image.size)

            result_path = os.path.join(output_dir, f"processed_{filter_type}.png")
            result_image.save(result_path)
            print(f"✓ Processed image saved to: {result_path}")


    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await crowdio.disconnect()
        print("\n👋 Disconnected from foreman")


if __name__ == "__main__":
    asyncio.run(main())
