#!/usr/bin/env python3
"""
Example client demonstrating declarative checkpointing

This example shows how to use the @CROWDio.task decorator with
checkpoint_enabled=True to enable automatic state capture and recovery.

The task simulates a long-running computation (e.g., Monte Carlo simulation)
where intermediate state should be preserved across worker failures.
"""

import sys
import os
import asyncio
import time
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from developer_sdk import crowdio_connect, crowdio_map, crowdio_disconnect, CROWDio


def parse_result(result):
    """
    Parse a task result that may be a dict or a JSON string.
    
    Workers may return results as JSON strings, so we need to handle both cases.
    """
    if isinstance(result, dict):
        return result
    elif isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Not valid JSON, return as error dict
            return {"error": result, "status": "parse_error"}
    else:
        return {"error": str(result), "status": "unknown_type"}


# =============================================================================
# Example 1: Simple checkpoint-enabled task - PURE LOGIC, NO RESUME CODE!
# =============================================================================
@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=0.5,  # Checkpoint every 0.5 seconds
    checkpoint_state=["count", "total", "samples", "progress_percent"]  # Variables to capture
)
def monte_carlo_pi(num_samples):
    """
    Estimate Pi using Monte Carlo method with automatic checkpointing.
    
    The @crowdio.task decorator enables:
    - Automatic state capture every 0.5 seconds
    - Only specified variables are checkpointed
    - TRANSPARENT recovery - resume is handled automatically by the framework!
    
    DEVELOPER WRITES PURE LOGIC - NO RESUME CODE NEEDED!
    The framework automatically:
    1. Captures checkpoint_state variables during execution
    2. On resume, injects saved values into variables
    3. Adjusts loop ranges to continue from checkpoint position
    
    Args:
        num_samples: Number of random samples to generate
        
    Returns:
        Estimated value of Pi
    """
    import random
    import time
    
    # Just declare your variables normally - framework handles resume!
    count = 0            # Points inside the quarter circle
    total = 0            # Total points generated
    samples = []         # Recent sample coordinates (for visualization)
    progress_percent = 0.0  # Track progress for checkpointing
    
    # Simple loop - framework will automatically adjust range on resume!
    for i in range(num_samples):
        x = random.random()
        y = random.random()
        
        total += 1
        
        if x * x + y * y <= 1:
            count += 1
            samples.append((x, y, True))
        else:
            samples.append((x, y, False))
        
        # Keep only last 100 samples to limit checkpoint size
        if len(samples) > 100:
            samples = samples[-100:]
        
        # Update progress - this is captured automatically!
        progress_percent = (i + 1) / num_samples * 100
        
        # Simulate some work
        time.sleep(0.001)
        
        # Progress logging every 1000 iterations
        if (i + 1) % 1000 == 0:
            current_pi = 4 * count / total
            print(f"  Progress: {i + 1}/{num_samples} samples, "
                  f"Pi estimate: {current_pi:.6f}")
    
    # Final calculation
    pi_estimate = 4 * count / total
    return {
        "pi_estimate": pi_estimate,
        "total_samples": total,
        "inside_circle": count
    }


# =============================================================================
# Example 2: Task with retry logic
# =============================================================================
@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=0.5,
    checkpoint_state=["processed", "results", "progress_percent"],
    retry_on_failure=True,
    max_retries=3
)
def process_data_batch(batch):
    """
    Process a batch of data with checkpointing and retry support.
    
    If the worker fails mid-processing:
    1. Latest checkpoint is recovered
    2. Processing resumes AUTOMATICALLY from last checkpoint
    3. Automatic retry up to 3 times if persistent failures
    
    PURE LOGIC - NO RESUME CODE NEEDED!
    Just write your algorithm, the framework handles resume transparently.
    
    Args:
        batch: List of data items to process
        
    Returns:
        List of processed results
    """
    import time
    
    # Just declare variables normally - framework handles resume!
    processed = 0
    results = []
    progress_percent = 0.0
    
    # Simple loop - framework adjusts range on resume
    for i, item in enumerate(batch):
        # Simulate processing
        result = item ** 2 + 1
        results.append(result)
        processed += 1
        
        # Update progress - automatically captured!
        progress_percent = (i + 1) / len(batch) * 100
        
        time.sleep(0.1)  # Simulate work
    
    return {
        "processed_count": processed,
        "results": results,
        "batch_sum": sum(results)
    }


# =============================================================================
# Example 3: Task without checkpointing (for comparison)
# =============================================================================
@CROWDio.task(
    checkpoint=False  # Explicitly disabled
)
def quick_task(x):
    """
    A quick task that doesn't need checkpointing.
    
    For fast operations, checkpointing overhead may not be worth it.
    Use checkpoint_enabled=False (default) for sub-second tasks.
    """
    import time
    time.sleep(0.05)
    return x * x


# =============================================================================
# Main execution
# =============================================================================
async def main():
    """Main example function demonstrating declarative checkpointing"""
    if len(sys.argv) != 2:
        print("Usage: python example_checkpoint_client.py <foreman_host>")
        print("Example: python example_checkpoint_client.py 192.168.1.10")
        sys.exit(1)
    
    foreman_host = sys.argv[1]
    
    print("=" * 60)
    print("Declarative Checkpointing Demo")
    print("=" * 60)
    print()
    print("This demo shows automatic state capture using @CROWDio.task decorator")
    print()
    
    try:
        # Connect to foreman
        print(f"Connecting to foreman at {foreman_host}:9000...")
        await crowdio_connect(foreman_host, 9000)
        print("Connected!\n")
        
        # =====================================================================
        # Example 1: Monte Carlo Pi estimation with checkpointing
        # =====================================================================
        print("-" * 60)
        print("Example 1: Monte Carlo Pi Estimation (checkpoint_enabled=True)")
        print("-" * 60)
        print("Task Configuration:")
        print(f"  - checkpoint_enabled: True")
        print(f"  - checkpoint_interval: 5 seconds")
        print(f"  - checkpoint_state: ['count', 'total', 'samples']")
        print()
        
        # Run multiple Monte Carlo simulations in parallel
        sample_counts = [5000, 5000, 5000, 5000]
        
        print(f"Running {len(sample_counts)} Monte Carlo simulations...")
        print("(Each with automatic state checkpointing)")
        print()
        
        start_time = time.time()
        results = await crowdio_map(monte_carlo_pi, sample_counts)
        end_time = time.time()
        
        print("\nResults:")
        parsed_results = [parse_result(r) for r in results]
        for i, result in enumerate(parsed_results):
            if "error" in result:
                print(f"  Simulation {i+1}: ERROR - {result.get('error', 'Unknown error')}")
            else:
                print(f"  Simulation {i+1}: Pi ≈ {result['pi_estimate']:.6f} "
                      f"({result['total_samples']} samples)")
        
        # Filter successful results for averaging
        valid_results = [r for r in parsed_results if "pi_estimate" in r]
        if valid_results:
            avg_pi = sum(r['pi_estimate'] for r in valid_results) / len(valid_results)
            print(f"\n  Average Pi estimate: {avg_pi:.6f}")
            print(f"  Actual Pi:           3.141593")
            print(f"  Error:               {abs(avg_pi - 3.141593):.6f}")
        else:
            print(f"\n  No valid results to average")
        print(f"  Time taken:          {end_time - start_time:.2f} seconds")
        
        # =====================================================================
        # Example 2: Batch processing with checkpointing and retries
        # =====================================================================
        print("\n" + "-" * 60)
        print("Example 2: Batch Processing (checkpoint + retry)")
        print("-" * 60)
        print("Task Configuration:")
        print(f"  - checkpoint_enabled: True")
        print(f"  - checkpoint_interval: 3 seconds")
        print(f"  - checkpoint_state: ['processed', 'results']")
        print(f"  - retry_on_failure: True")
        print(f"  - max_retries: 3")
        print()
        
        # Create batches of data
        batches = [
            list(range(1, 11)),    # Batch 1: 1-10
            list(range(11, 21)),   # Batch 2: 11-20
            list(range(21, 31)),   # Batch 3: 21-30
        ]
        
        print(f"Processing {len(batches)} batches...")
        
        start_time = time.time()
        batch_results = await crowdio_map(process_data_batch, batches)
        end_time = time.time()
        
        print("\nResults:")
        parsed_batch_results = [parse_result(r) for r in batch_results]
        for i, result in enumerate(parsed_batch_results):
            if "error" in result:
                print(f"  Batch {i+1}: ERROR - {result.get('error', 'Unknown error')}")
            else:
                print(f"  Batch {i+1}: Processed {result['processed_count']} items, "
                      f"Sum = {result['batch_sum']}")
        
        valid_batch_results = [r for r in parsed_batch_results if "batch_sum" in r]
        if valid_batch_results:
            total_sum = sum(r['batch_sum'] for r in valid_batch_results)
            print(f"\n  Total sum across all batches: {total_sum}")
        else:
            print(f"\n  No valid batch results")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        
        # =====================================================================
        # Example 3: Quick tasks without checkpointing
        # =====================================================================
        print("\n" + "-" * 60)
        print("Example 3: Quick Tasks (checkpoint_enabled=False)")
        print("-" * 60)
        print("Task Configuration:")
        print(f"  - checkpoint_enabled: False")
        print(f"  (No checkpoint overhead for fast tasks)")
        print()
        
        numbers = list(range(1, 21))
        
        print(f"Running {len(numbers)} quick tasks...")
        
        start_time = time.time()
        quick_results = await crowdio_map(quick_task, numbers)
        end_time = time.time()
        
        print(f"\nResults: {quick_results[:5]}... (first 5 shown)")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("""
Declarative Checkpointing Benefits:
  1. Automatic state capture - no manual checkpoint code needed
  2. Configurable intervals - balance overhead vs. recovery granularity  
  3. Selective state - only checkpoint specified variables
  4. Transparent recovery - tasks resume from last checkpoint on failure
  5. Retry support - automatic retry with checkpoint restoration
  
Best Practices:
  - Use checkpoint_enabled=True for long-running tasks (>30 seconds)
  - Keep checkpoint_state minimal - only essential variables
  - Set checkpoint_interval based on failure probability
  - Disable checkpointing for quick tasks to avoid overhead
        """)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await crowdio_disconnect()
        print("\nDisconnected from foreman")


if __name__ == "__main__":
    asyncio.run(main())
