"""
Example: Distributing TensorFlow tensors among workers

This example demonstrates how to:
1. Create TensorFlow tensors
2. Serialize and distribute them to workers
3. Execute tensor operations on workers
4. Collect and aggregate results
"""

import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import uuid
import tensorflow as tf
import numpy as np

from common.tensorflow_utils import (
    TensorSerializer,
    TensorDistributor,
    TensorTaskExecutor,
    TensorResultHandler
)
from common.tensorflow_protocol import (
    create_distribute_tensor_message,
    create_execute_tensor_task_message,
    create_tensor_task_result_message
)


# ============================================================================
# Example 1: Basic Tensor Serialization
# ============================================================================

def example_tensor_serialization():
    """Example of serializing and deserializing tensors"""
    print("\n=== Example 1: Tensor Serialization ===")
    
    # Create a sample tensor
    tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"Original tensor:\n{tensor}")
    
    # Serialize it
    serialized = TensorSerializer.serialize_tensor(tensor)
    print(f"Serialized (truncated): {serialized[:100]}...")
    
    # Deserialize it
    deserialized = TensorSerializer.deserialize_tensor(serialized)
    print(f"Deserialized tensor:\n{deserialized}")
    print(f"Tensors equal: {tf.reduce_all(tensor == deserialized)}")


# ============================================================================
# Example 2: Batch Tensor Operations
# ============================================================================

def example_batch_tensors():
    """Example of working with batches of tensors"""
    print("\n=== Example 2: Batch Tensor Operations ===")
    
    # Create multiple tensors
    tensors = [
        tf.random.normal((3, 4)),
        tf.random.normal((3, 4)),
        tf.random.normal((3, 4))
    ]
    
    # Serialize batch
    serialized = TensorSerializer.serialize_batch_tensors(tensors)
    print(f"Batch serialized: {len(serialized)} bytes")
    
    # Deserialize batch
    deserialized = TensorSerializer.deserialize_batch_tensors(serialized)
    print(f"Deserialized {len(deserialized)} tensors")
    print(f"Shapes: {[t.shape for t in deserialized]}")


# ============================================================================
# Example 3: Tensor Distribution and Splitting
# ============================================================================

def example_tensor_distribution():
    """Example of distributing a tensor among workers"""
    print("\n=== Example 3: Tensor Distribution ===")
    
    # Create a large tensor (simulating data to process)
    data_tensor = tf.constant([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
    ], dtype=tf.float32)
    
    print(f"Original tensor shape: {data_tensor.shape}")
    
    # Create distributor
    distributor = TensorDistributor()
    
    # Split tensor for 3 workers
    num_workers = 3
    shards = distributor.split_tensor_for_workers(data_tensor, num_workers, axis=0)
    
    print(f"Split into {len(shards)} shards:")
    for i, shard in enumerate(shards):
        print(f"  Worker {i}: shape {shard.shape}")
    
    # Merge shards back
    merged = distributor.merge_tensor_shards(shards, axis=0)
    print(f"Merged shape: {merged.shape}")
    print(f"Reconstruction successful: {tf.reduce_all(data_tensor == merged)}")


# ============================================================================
# Example 4: Tensor Operations
# ============================================================================

def example_tensor_operations():
    """Example of applying tensor operations"""
    print("\n=== Example 4: Tensor Operations ===")
    
    tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"Original tensor:\n{tensor}")
    
    executor = TensorTaskExecutor()
    
    # Apply various operations
    operations = [
        ("reduce_mean", {"axis": 0}),
        ("reduce_sum", {"axis": 1}),
        ("square", {}),
        ("normalize", {"axis": 1}),
    ]
    
    for op_name, params in operations:
        result = executor.apply_tensor_operation(tensor, op_name, **params)
        print(f"\n{op_name}: shape {result.shape}")
        print(result)


# ============================================================================
# Example 5: Custom Tensor Function
# ============================================================================

def example_custom_tensor_function():
    """Example of executing a custom TensorFlow function"""
    print("\n=== Example 5: Custom Tensor Function ===")
    
    # Define a function as source code string
    func_code = """
def process_tensor(tensor, scale_factor=2.0):
    '''Process tensor by scaling and squaring'''
    scaled = tensor * scale_factor
    return tf.square(scaled)
"""
    
    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    print(f"Original tensor:\n{tensor}")
    
    executor = TensorTaskExecutor()
    result = executor.execute_tensor_function(func_code, tensor, scale_factor=2.0)
    print(f"Result after custom function:\n{result}")


# ============================================================================
# Example 6: Result Handling
# ============================================================================

def example_result_handling():
    """Example of serializing and deserializing results"""
    print("\n=== Example 6: Result Handling ===")
    
    handler = TensorResultHandler()
    
    # Different result types
    results = [
        tf.constant([1.0, 2.0, 3.0]),  # Tensor
        42.5,  # Scalar
        {"data": "value"},  # Dict
        [1, 2, 3],  # List
    ]
    
    for result in results:
        serialized = handler.serialize_result(result)
        deserialized = handler.deserialize_result(serialized)
        print(f"Original type: {type(result)}, Deserialized type: {type(deserialized)}")


# ============================================================================
# Example 7: Complete Workflow - Matrix Operations
# ============================================================================

def example_complete_workflow():
    """Example of a complete tensor workflow"""
    print("\n=== Example 7: Complete Workflow - Matrix Operations ===")
    
    # Step 1: Create input tensors
    print("\n1. Creating input tensors...")
    A = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    B = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    # Step 2: Serialize tensors for distribution
    print("\n2. Serializing tensors for distribution...")
    tensor_id_a = str(uuid.uuid4())
    tensor_id_b = str(uuid.uuid4())
    serialized_a = TensorSerializer.serialize_tensor(A)
    serialized_b = TensorSerializer.serialize_tensor(B)
    print(f"Tensor A ID: {tensor_id_a}")
    print(f"Tensor B ID: {tensor_id_b}")
    
    # Step 3: Simulate sending to workers and executing operations
    print("\n3. Distributing to workers and executing operations...")
    distributor = TensorDistributor()
    distributor.cache_tensor(tensor_id_a, A)
    distributor.cache_tensor(tensor_id_b, B)
    
    # Step 4: Execute matrix multiplication on worker
    print("\n4. Executing matrix multiplication...")
    executor = TensorTaskExecutor()
    B_cached = distributor.get_cached_tensor(tensor_id_b)
    result = executor.apply_tensor_operation(
        A,
        "matmul",
        other=B_cached
    )
    print(f"Result of A × B:\n{result}")
    
    # Step 5: Serialize and return results
    print("\n5. Serializing results for return...")
    handler = TensorResultHandler()
    serialized_result = handler.serialize_result(result)
    deserialized_result = handler.deserialize_result(serialized_result)
    print(f"Final result retrieved:\n{deserialized_result}")


# ============================================================================
# Example 8: Batch Processing with Tensor Shards
# ============================================================================

def example_batch_processing():
    """Example of batch processing with distributed tensor shards"""
    print("\n=== Example 8: Batch Processing with Tensor Shards ===")
    
    # Create a dataset (batch of samples)
    dataset = tf.constant(np.random.randn(100, 28, 28).astype(np.float32))
    print(f"Dataset shape: {dataset.shape}")
    
    # Distribute to 4 workers
    num_workers = 4
    distributor = TensorDistributor()
    shards = distributor.split_tensor_for_workers(dataset, num_workers, axis=0)
    
    print(f"\nDistributed to {num_workers} workers:")
    
    # Simulate processing on each worker
    executor = TensorTaskExecutor()
    results = []
    
    for i, shard in enumerate(shards):
        # Each worker normalizes its shard
        normalized = executor.apply_tensor_operation(shard, "normalize", axis=-1)
        # Compute mean for this shard
        mean = tf.reduce_mean(normalized, axis=0)
        results.append(mean)
        print(f"Worker {i}: processed shape {shard.shape} -> mean shape {mean.shape}")
    
    # Aggregate results
    aggregated = tf.stack(results)
    final_mean = tf.reduce_mean(aggregated, axis=0)
    print(f"\nAggregated result shape: {final_mean.shape}")
    print(f"Final computed mean shape: {final_mean.shape}")


if __name__ == "__main__":
    print("=" * 70)
    print("TensorFlow Tensor Distribution Examples")
    print("=" * 70)
    
    example_tensor_serialization()
    example_batch_tensors()
    example_tensor_distribution()
    example_tensor_operations()
    example_custom_tensor_function()
    example_result_handling()
    example_complete_workflow()
    example_batch_processing()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
