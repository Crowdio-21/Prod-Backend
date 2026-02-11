import sys
import os
import asyncio
import time
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import connect, disconnect, crowdio
from developer_sdk.image_utils import (
    split_image_into_tiles,
    reassemble_tiles 
)


@crowdio.task(
    checkpoint=True,
    checkpoint_interval=3.0,
    checkpoint_state=["processed_count", "total_tiles", "avg_time", "progress"],
)
def process_tile(tile_data):
    """
    Process an image tile remotely with checkpointing support.
    Applies various image transformations to demonstrate distributed processing.

    Args:
        tile_data: Dictionary containing:
            - 'image': base64 encoded image data
            - 'filter_type': type of filter to apply
            - 'tile_id': identifier for this tile

    Returns:
        Dictionary with processed tile data and metadata

    Note:
        Checkpointing is enabled via @crowdio.task decorator.
        State variables (processed_count, total_tiles, avg_time, progress)
        are captured automatically and restored on resume.
    """
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
    avg_time = processing_time
    progress = 100.0

    return {
        "tile_id": tile_data["tile_id"],
        "image": processed_image,
        "processing_time": processing_time,
        "filter_applied": filter_type,
    }


async def main():
    """Main function demonstrating distributed image tile processing with checkpointing"""

    try:
        foreman_host = "localhost" 
        await connect(foreman_host, 9000) 

        image_path = "image.png" 
        
        original_image = Image.open(image_path)
        
        tile_size = 200
        tiles = split_image_into_tiles(original_image, tile_size=tile_size)
        
        # filter_types = ["sharpen", "enhance", "blur", "grayscale", "edge"]
        filter_types = ["sharpen"]

        for filter_type in filter_types:
        
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

            start_time = time.time()

            processed_tiles = await process_tile.map(tile_inputs)

            end_time = time.time()
            total_time = end_time - start_time
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

            result_path = os.path.join("output", f"processed_{filter_type}.png")
            result_image.save(result_path)
            print(f"✓ Processed image saved to: {result_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await disconnect()
        print("\n👋 Disconnected from foreman")


if __name__ == "__main__":
    asyncio.run(main())
