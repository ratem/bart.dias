import unittest
import ast
from bdias_pattern_analyzer import BDiasPatternAnalyzer
from bdias_parser import BDiasParser


class TestPatternSpecificStructures(unittest.TestCase):
    """Test cases for specific parallel pattern structures."""

    def setUp(self):
        """Set up test cases."""
        self.parser = BDiasParser()
        self.analyzer = BDiasPatternAnalyzer(self.parser)

    def test_map_pattern(self):
        """Test Map pattern detection in various forms."""
        # Basic for loop
        code1 = """
def map_basic():
    result = []
    for i in range(100):
        result.append(i * i)
    return result
"""
        # List comprehension
        code2 = """
def map_listcomp():
    return [i * i for i in range(100)]
"""
        # Nested independent loops
        code3 = """
def map_nested():
    result = []
    for i in range(10):
        for j in range(10):
            result.append(i * j)
    return result
"""
        for code in [code1, code2, code3]:
            structured_code = self.parser.parse(code)
            identified_patterns = self.analyzer._identify_patterns(structured_code)
            self.assertGreater(len(identified_patterns["map"]), 0)

    def test_stencil_pattern(self):
        """Test Stencil pattern detection."""
        code = """
def stencil_2d(grid):
    n = len(grid)
    new_grid = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(1, n-1):
        for j in range(1, n-1):
            new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] + 
                             grid[i][j-1] + grid[i][j+1]) / 4
    return new_grid
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        self.assertGreater(len(identified_patterns["stencil"]), 0)

    def test_pipeline_pattern(self):
        """Test Pipeline pattern detection."""
        code = """
def pipeline():
    buffer1 = []
    buffer2 = []
    result = []

    # Stage 1: Generate data
    for i in range(100):
        buffer1.append(i * i)

    # Stage 2: Transform data
    for item in buffer1:
        buffer2.append(item + 10)

    # Stage 3: Process data
    for item in buffer2:
        result.append(item * 2)

    return result
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        self.assertGreater(len(identified_patterns["pipeline"]), 0)

    def test_reduction_pattern(self):
        """Test Reduction pattern detection."""
        code = """
def reduction(data):
    # Simple sum reduction
    result = 0
    for item in data:
        result += item
    return result
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        self.assertGreater(len(identified_patterns["reduction"]), 0)

    def test_fork_join_pattern(self):
        """Test Fork-Join pattern detection."""
        code = """
def fork_join(data):
    results = []

    # Fork: Process chunks independently
    chunks = [data[i:i+10] for i in range(0, len(data), 10)]
    for chunk in chunks:
        results.append(process_chunk(chunk))

    # Join: Combine results
    return sum(results)
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        self.assertGreater(len(identified_patterns["fork_join"]), 0)

    def test_divide_conquer_pattern(self):
        """Test Divide & Conquer pattern detection."""
        code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        self.assertGreater(len(identified_patterns["divide_conquer"]), 0)

    def test_master_worker_pattern(self):
        """Test Master-Worker pattern detection."""
        code = """
def master_worker(tasks):
    results = []

    # Master distributes tasks
    for task in tasks:
        results.append(worker_process(task))

    return results
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        self.assertGreater(len(identified_patterns["master_worker"]), 0)

    def test_scatter_gather_pattern(self):
        """Test Scatter-Gather pattern detection."""
        code = """
def scatter_gather(data):
    # Scatter phase
    chunks = [data[i:i+10] for i in range(0, len(data), 10)]
    processed = []

    # Process each chunk
    for chunk in chunks:
        processed.append(process_chunk(chunk))

    # Gather phase
    result = []
    for chunk in processed:
        result.extend(chunk)

    return result
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        self.assertGreater(len(identified_patterns["scatter_gather"]), 0)
