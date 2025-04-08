import numpy as np
import time
import random
from functools import lru_cache

# Large data collections
LARGE_ARRAY = np.random.rand(100000)
LARGE_LIST = [random.random() for _ in range(50000)]
LARGE_MATRIX = np.random.rand(1000, 1000)


# Recursive Fibonacci with large values
def fibonacci(n):
    """Compute Fibonacci numbers recursively"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Memoized version for comparison
@lru_cache(maxsize=None)
def fibonacci_memo(n):
    """Compute Fibonacci numbers with memoization"""
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)


# Recursive function with loop (combo)
def recursive_with_loop(n, data):
    """Recursive function that contains a loop"""
    if n <= 0:
        return 0

    result = 0
    for i in range(len(data) // n):
        result += data[i] * n

    return result + recursive_with_loop(n - 1, data)


# While loop containing for loop (combo)
def while_with_for(matrix):
    """While loop that contains a for loop"""
    result = 0
    i = 0
    while i < len(matrix):
        row_sum = 0
        for j in range(len(matrix[i])):
            row_sum += matrix[i][j] ** 2
        result += row_sum
        i += 1
    return result


# For loop with recursive function calls (combo)
def for_with_recursive(data, depth):
    """For loop that calls a recursive function"""
    result = 0
    for i in range(len(data)):
        if i % 100 == 0:  # Only process every 100th element to avoid excessive computation
            result += fibonacci(depth + (i % 10))
    return result


# Nested loops with varying depths
def nested_loops_varying_depth(matrix):
    """Nested loops with varying depths"""
    result = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i % 3 == 0:
                for k in range(min(10, j + 1)):
                    result += matrix[i][j] * k
            else:
                result += matrix[i][j]
    return result


# Function with loop that calls another function with loop (combo)
def function_with_loop_function(data):
    """Function with a loop that calls another function containing a loop"""
    result = 0
    for i in range(0, len(data), 100):  # Process in chunks of 100
        chunk = data[i:i + 100]
        result += process_chunk(chunk)
    return result


def process_chunk(chunk):
    """Process a chunk of data using a loop"""
    chunk_result = 0
    for item in chunk:
        chunk_result += item ** 2
    return chunk_result


if __name__ == "__main__":
    print("Starting performance tests...")

    # Test 1: Regular Fibonacci (will be slow)
    print("\nTest 1: Regular Fibonacci")
    start = time.time()
    result = fibonacci(30)
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    # Test 2: Memoized Fibonacci (for comparison)
    print("\nTest 2: Memoized Fibonacci")
    start = time.time()
    result = fibonacci_memo(35)
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    # Test 3: Recursive with loop
    print("\nTest 3: Recursive with loop")
    start = time.time()
    result = recursive_with_loop(20, LARGE_LIST[:1000])
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    # Test 4: While with for
    print("\nTest 4: While with for")
    start = time.time()
    result = while_with_for(LARGE_MATRIX[:500])
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    # Test 5: For with recursive
    print("\nTest 5: For with recursive")
    start = time.time()
    result = for_with_recursive(LARGE_LIST, 15)
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    # Test 6: Nested loops with varying depths
    print("\nTest 6: Nested loops with varying depths")
    start = time.time()
    result = nested_loops_varying_depth(LARGE_MATRIX[:200])
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    # Test 7: Function with loop function
    print("\nTest 7: Function with loop function")
    start = time.time()
    result = function_with_loop_function(LARGE_ARRAY)
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    print("\nAll tests completed.")
