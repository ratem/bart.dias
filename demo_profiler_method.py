"""
critical_path_test.py: Test suite for Bart.dIAs Critical Path Analysis

This file contains test cases specifically designed to stress test the DAG-based
critical path analysis algorithm. It includes examples of various parallel patterns
with known theoretical properties to validate the work-span model calculations.

The test cases cover:
1. Simple independent loops (embarrassingly parallel)
2. Nested loops with dependencies (limited parallelism)
3. Recursive algorithms with different branching factors
4. Pipeline patterns with sequential stages
5. Task graphs with complex dependencies
6. Combinations of patterns that create interesting critical paths
"""

import numpy as np
import time
import sys
from functools import reduce

def test_independent_loop(n=1000):
    """
    Embarrassingly parallel pattern with O(n) work and O(1) span.
    Expected parallelism: O(n)
    """
    result = []
    for i in range(n):
        # Independent computation with no dependencies between iterations
        result.append(i * i + 3 * i - 2)
    return sum(result)

def test_nested_independent_loops(n=100):
    """
    Nested loops with independent iterations.
    Work: O(n²), Span: O(1)
    Expected parallelism: O(n²)
    """
    result = 0
    for i in range(n):
        for j in range(n):
            # Each iteration is independent
            result += i * j
    return result

def test_loop_carried_dependency(n=1000):
    """
    Loop with carried dependency - forms a critical path.
    Work: O(n), Span: O(n)
    Expected parallelism: O(1)
    """
    result = 0
    for i in range(n):
        # Each iteration depends on the previous one
        result = result + i
    return result

def test_partial_dependency_loop(n=1000):
    """
    Loop where only part of the computation has dependencies.
    Work: O(n), Span: O(log n) with proper parallelization
    Expected parallelism: O(n/log n)
    """
    result = 0
    temp = []
    
    # First part: independent loop - parallelizable
    for i in range(n):
        temp.append(i * i)
    
    # Second part: reduction - limited parallelism
    for value in temp:
        result += value
        
    return result

def test_recursive_fibonacci(n=20):
    """
    Classic recursive Fibonacci with exponential work.
    Work: O(2^n), Span: O(n)
    Expected parallelism: O(2^n/n)
    """
    if n <= 1:
        return n
    else:
        return test_recursive_fibonacci(n-1) + test_recursive_fibonacci(n-2)

def test_divide_and_conquer(data):
    """
    Divide and conquer pattern (like mergesort).
    Work: O(n log n), Span: O(log n)
    Expected parallelism: O(n)
    """
    if len(data) <= 1:
        return data
    
    mid = len(data) // 2
    left = test_divide_and_conquer(data[:mid])
    right = test_divide_and_conquer(data[mid:])
    
    # Merge step
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def test_pipeline_pattern(data, n=1000):
    """
    Pipeline pattern with sequential stages.
    Work: O(n), Span: O(n)
    Expected parallelism: O(1) for fixed stages, O(s) for s stages
    """
    # Stage 1: Filter
    filtered = []
    for item in data:
        if item % 2 == 0:  # Filter even numbers
            filtered.append(item)
    
    # Stage 2: Map
    mapped = []
    for item in filtered:
        mapped.append(item * item)
    
    # Stage 3: Reduce
    result = 0
    for item in mapped:
        result += item
    
    return result

def test_dynamic_programming(n=20):
    """
    Dynamic programming with dependencies.
    Work: O(n²), Span: O(n)
    Expected parallelism: O(n)
    """
    # Calculate binomial coefficients
    dp = [[0 for _ in range(n+1)] for _ in range(n+1)]
    
    # Base cases
    for i in range(n+1):
        dp[i][0] = 1
    
    # Fill the dp table
    for i in range(1, n+1):
        for j in range(1, i+1):
            dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
    
    return dp[n][n//2]  # Return the middle value

def test_stencil_computation(n=100):
    """
    Stencil pattern (e.g., 2D grid computation).
    Work: O(n²), Span: O(n)
    Expected parallelism: O(n)
    """
    grid = np.zeros((n, n))
    new_grid = np.zeros((n, n))
    
    # Initialize grid
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i + j) % 10
    
    # Apply stencil (e.g., average of neighbors)
    for i in range(1, n-1):
        for j in range(1, n-1):
            new_grid[i, j] = (grid[i-1, j] + grid[i+1, j] + 
                             grid[i, j-1] + grid[i, j+1]) / 4
    
    return np.sum(new_grid)

def test_fork_join_pattern(n=1000):
    """
    Fork-join pattern with independent tasks.
    Work: O(n), Span: O(log n)
    Expected parallelism: O(n/log n)
    """
    data = list(range(n))
    
    # Split data into chunks (fork)
    chunk_size = 10
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process each chunk independently
    processed_chunks = []
    for chunk in chunks:
        chunk_sum = sum(x*x for x in chunk)
        processed_chunks.append(chunk_sum)
    
    # Join results
    final_result = sum(processed_chunks)
    return final_result

def test_wavefront_pattern(n=50):
    """
    Wavefront pattern with diagonal dependencies.
    Work: O(n²), Span: O(2n-1) = O(n)
    Expected parallelism: O(n)
    """
    # Create a matrix
    matrix = np.zeros((n, n))
    
    # Fill the matrix with initial values
    for i in range(n):
        for j in range(n):
            matrix[i, j] = i + j
    
    # Compute wavefront (each cell depends on cells to the left and above)
    dp = np.zeros((n, n))
    dp[0, 0] = matrix[0, 0]
    
    # Fill first row and column
    for i in range(1, n):
        dp[0, i] = dp[0, i-1] + matrix[0, i]
        dp[i, 0] = dp[i-1, 0] + matrix[i, 0]
    
    # Fill the rest of the matrix
    for i in range(1, n):
        for j in range(1, n):
            dp[i, j] = max(dp[i-1, j], dp[i, j-1]) + matrix[i, j]
    
    return dp[n-1, n-1]

def test_complex_task_graph(n=100):
    """
    Complex task graph with multiple dependencies.
    Creates a DAG with interesting critical path properties.
    """
    # Initialize values
    a = list(range(n))
    b = list(range(n, 2*n))
    c = []
    d = []
    e = []
    
    # Task 1: Process a
    for item in a:
        c.append(item * item)
    
    # Task 2: Process b (independent of Task 1)
    for item in b:
        d.append(item + 10)
    
    # Task 3: Combine results from Tasks 1 and 2
    for i in range(n):
        e.append(c[i] + d[i])
    
    # Task 4: Final reduction
    result = sum(e)
    
    return result

def test_master_worker_pattern(n=1000):
    """
    Master-worker pattern simulation.
    Work: O(n), Span: O(n/p) where p is number of workers
    """
    tasks = list(range(n))
    results = []
    
    # Simulate distribution to workers
    num_workers = 4
    chunks = [tasks[i:i+n//num_workers] for i in range(0, n, n//num_workers)]
    
    # Workers process their chunks
    for chunk in chunks:
        chunk_results = []
        for task in chunk:
            # Process each task
            chunk_results.append(task * task)
        results.extend(chunk_results)
    
    # Master aggregates results
    final_result = sum(results)
    return final_result

def test_amdahl_pattern(n=1000, sequential_fraction=0.1):
    """
    Pattern demonstrating Amdahl's Law limitations.
    Has a fixed sequential component that limits parallelism.
    """
    result = 0
    
    # Sequential part (cannot be parallelized)
    sequential_work = int(n * sequential_fraction)
    for i in range(sequential_work):
        result += i * i
        
    # Parallel part
    parallel_work = n - sequential_work
    for i in range(parallel_work):
        result += i
        
    return result

def test_critical_path_with_branches(n=100):
    """
    Creates a computation with multiple paths but one critical path.
    """
    # Path 1 (shorter work but sequential)
    path1_result = 0
    for i in range(n):
        path1_result = path1_result + i  # Sequential dependency
    
    # Path 2 (more work but parallelizable)
    path2_result = 0
    for i in range(n*n):
        path2_result += 1  # Independent iterations
    
    # Path 3 (medium work, partially parallelizable)
    path3_result = 0
    temp = []
    for i in range(n*10):
        temp.append(i % 10)  # Parallelizable
    for t in temp:
        path3_result += t  # Sequential reduction
    
    # Combine results (all paths must complete)
    return path1_result + path2_result + path3_result

def test_mixed_patterns():
    """
    Combines multiple patterns to create a complex dependency structure.
    """
    # First, generate some data with independent operations
    n = 100
    data = [i * i for i in range(n)]
    
    # Apply a stencil-like operation (with dependencies)
    for i in range(1, n-1):
        data[i] = (data[i-1] + data[i] + data[i+1]) / 3
    
    # Perform a reduction with loop-carried dependency
    result = 0
    for item in data:
        result = result * 0.99 + item
    
    # Apply a recursive operation
    def recursive_process(start, end):
        if end - start <= 2:
            return data[start] + (data[end-1] if start < end else 0)
        mid = (start + end) // 2
        return recursive_process(start, mid) + recursive_process(mid, end)
    
    final_result = recursive_process(0, n)
    return final_result + result

def test_nested_complex_patterns():
    """
    Creates deeply nested patterns with mixed dependencies.
    """
    n = 20
    result = 0
    
    # Outer loop with partial dependencies
    for i in range(n):
        # Independent computation
        temp = i * i
        
        # Nested loop with dependencies
        inner_result = 0
        for j in range(i+1):
            inner_result += j * temp
        
        # Recursive call with work proportional to i
        def recursive_sum(k):
            if k <= 1:
                return k
            return k + recursive_sum(k-1)
        
        # Combine results in a way that creates a critical path
        result = result + inner_result + recursive_sum(i)
    
    return result

def test_amdahl_with_nested_patterns(n=50, seq_fraction=0.2):
    """
    Demonstrates Amdahl's Law with nested patterns.
    """
    result = 0
    
    # Sequential part (cannot be parallelized)
    seq_work = int(n * seq_fraction)
    for i in range(seq_work):
        # Do some sequential work with dependencies
        for j in range(i+1):
            result = result + (i * j)
    
    # Parallel part
    parallel_work = n - seq_work
    partial_results = []
    
    # This loop could be parallelized
    for i in range(parallel_work):
        # Each iteration does independent work
        iter_result = 0
        for j in range(n):
            iter_result += i * j
        partial_results.append(iter_result)
    
    # Combine results
    for pr in partial_results:
        result += pr
    
    return result

def run_all_tests():
    """
    Run all test functions and print their execution times.
    """
    test_functions = [
        (test_independent_loop, [1000], "Independent Loop"),
        (test_nested_independent_loops, [50], "Nested Independent Loops"),
        (test_loop_carried_dependency, [1000], "Loop with Carried Dependency"),
        (test_partial_dependency_loop, [1000], "Partial Dependency Loop"),
        (test_recursive_fibonacci, [20], "Recursive Fibonacci"),
        (test_divide_and_conquer, [list(range(100))], "Divide and Conquer"),
        (test_pipeline_pattern, [list(range(1000))], "Pipeline Pattern"),
        (test_dynamic_programming, [20], "Dynamic Programming"),
        (test_stencil_computation, [50], "Stencil Computation"),
        (test_fork_join_pattern, [1000], "Fork-Join Pattern"),
        (test_wavefront_pattern, [30], "Wavefront Pattern"),
        (test_complex_task_graph, [100], "Complex Task Graph"),
        (test_master_worker_pattern, [1000], "Master-Worker Pattern"),
        (test_amdahl_pattern, [1000, 0.1], "Amdahl Pattern (10% sequential)"),
        (test_amdahl_pattern, [1000, 0.5], "Amdahl Pattern (50% sequential)"),
        (test_critical_path_with_branches, [50], "Critical Path with Branches"),
        (test_mixed_patterns, [], "Mixed Patterns"),
        (test_nested_complex_patterns, [], "Nested Complex Patterns"),
        (test_amdahl_with_nested_patterns, [50, 0.2], "Amdahl with Nested Patterns")
    ]
    
    print("=" * 80)
    print("CRITICAL PATH ANALYSIS TEST SUITE")
    print("=" * 80)
    print("\nRunning tests to evaluate critical path analysis algorithm...\n")
    
    results = []
    for func, args, name in test_functions:
        print(f"Running {name}...", end="")
        sys.stdout.flush()
        
        start_time = time.time()
        result = func(*args)
        elapsed = time.time() - start_time
        
        print(f" done in {elapsed:.4f} seconds")
        results.append((name, elapsed, result))
    
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Test Name':<40} {'Time (s)':<15} {'Result':<15}")
    print("-" * 80)
    
    for name, elapsed, result in results:
        result_str = str(result)
        if len(result_str) > 15:
            result_str = result_str[:12] + "..."
        print(f"{name:<40} {elapsed:<15.4f} {result_str:<15}")

if __name__ == "__main__":
    run_all_tests()

