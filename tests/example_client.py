import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from developer_sdk import CrowdioClient

 
crowdio = CrowdioClient()

@crowdio.remote
def square(x):
    """Simple function to square a number"""
    import time

    time.sleep(0.1)
    return x**2

@crowdio.remote
def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

@crowdio.remote
def process_data(data):
    """Process some data"""
    import time

    result = sum(data) * 2
    time.sleep(0.05)
    return result


async def main():
    """Main example function"""
    
    try:
        foreman_host = "localhost"          
        await crowdio.init(foreman_host, 9000)

        print("\n1. Running square function on numbers 1-10...")
        numbers = list(range(1, 11))
        start_time = time.time()
        results = await square.map(numbers)
        end_time = time.time()
        print(f"Results: {results}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        print("\n2. Calculating fibonacci numbers...")
        fib_inputs = [10, 15, 20, 25, 30]
        start_time = time.time()
        fib_results = await fibonacci.map(fib_inputs)
        end_time = time.time()
        print(f"Fibonacci results: {fib_results}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        print("\n3. Processing data arrays...")
        data_arrays = [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [100, 200, 300, 400, 500],
            [1000, 2000, 3000, 4000, 5000],
        ]
        start_time = time.time()
        processed_results = await process_data.map(data_arrays)
        end_time = time.time()
        print(f"Processed results: {processed_results}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")


        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await crowdio.disconnect()
        print("Disconnected from foreman")


if __name__ == "__main__":
    asyncio.run(main())
