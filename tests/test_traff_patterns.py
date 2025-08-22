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
            identified_patterns = self.analyzer._identify_patterns(structured_code,code)
            self.assertGreater(len(identified_patterns["map_reduce"]), 0)

    def test_stencil_pattern_direct(self):
        """Test stencil pattern detection directly without using the parser."""
        code = """def stencil_function(grid):
        n = len(grid)
        new_grid = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(1, n-1):
            for j in range(1, n-1):
                new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] +
                                 grid[i][j-1] + grid[i][j+1]) / 4
        return new_grid
    """
        # Parse the code directly with ast
        tree = ast.parse(code)

        # Create a mock structured_code dictionary
        structured_code = {
            "loops": [
                {
                    "type": "nested_for",
                    "lineno": 4,
                    "source": """for i in range(1, n-1):
            for j in range(1, n-1):
                new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] +
                                 grid[i][j-1] + grid[i][j+1]) / 4"""
                }
            ]
        }

        # Test if the _has_neighbor_access method works correctly
        loop_node = ast.parse(structured_code["loops"][0]["source"])
        has_neighbor = self.analyzer._has_neighbor_access(loop_node)
        self.assertTrue(has_neighbor, "Stencil pattern should have neighbor access")

        # Test pattern identification with the mock structured_code
        identified_patterns = self.analyzer._identify_patterns(structured_code,code)
        self.assertGreater(len(identified_patterns["stencil"]), 0)

    def test_pipeline_pattern_direct(self):
        """Test pipeline pattern detection directly without using the parser."""
        code = """def pipeline_function():
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
        # Parse the code directly with ast
        tree = ast.parse(code)

        # Create a mock structured_code dictionary
        structured_code = {
            "functions": [
                {
                    "type": "function",
                    "lineno": 1,
                    "source": code
                }
            ]
        }

        # Test if the _has_producer_consumer_pattern method works correctly
        function_node = ast.parse(structured_code["functions"][0]["source"])
        has_producer_consumer = self.analyzer._has_producer_consumer_pattern(function_node)
        self.assertTrue(has_producer_consumer, "Pipeline pattern should be detected")

        # Test pattern identification with the mock structured_code
        identified_patterns = self.analyzer._identify_patterns(structured_code,code)
        self.assertGreater(len(identified_patterns["pipeline"]), 0)


    def test_fork_join_pattern_direct(self):
        """Test fork-join pattern detection directly without using the parser."""
        code = """def fork_join(data):
        results = []

        # Fork: Process chunks independently
        chunks = [data[i:i+10] for i in range(0, len(data), 10)]
        for chunk in chunks:
            results.append(process_chunk(chunk))

        # Join: Combine results
        return sum(results)
    """
        # Parse the code directly with ast
        tree = ast.parse(code)

        # Create a mock structured_code dictionary
        structured_code = {
            "functions": [
                {
                    "type": "function",
                    "lineno": 1,
                    "source": code
                }
            ]
        }

        # Test if the _has_independent_tasks method works correctly
        function_node = ast.parse(structured_code["functions"][0]["source"])
        has_independent_tasks = self.analyzer._has_independent_tasks(function_node)
        self.assertTrue(has_independent_tasks, "Fork-join pattern should be detected")

        # Test pattern identification with the mock structured_code
        identified_patterns = self.analyzer._identify_patterns(structured_code,code)
        self.assertGreater(len(identified_patterns["fork_join"]), 0)

    def test_divide_conquer_pattern_direct(self):
        """Test divide and conquer pattern detection directly without using the parser."""
        code = """def merge_sort(arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        return merge(left, right)
    """
        # Parse the code directly with ast
        tree = ast.parse(code)

        # Create a mock structured_code dictionary
        structured_code = {
            "functions": [
                {
                    "type": "recursive_function",
                    "lineno": 2,
                    "source": """def merge_sort(arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        return merge(left, right)"""
                }
            ]
        }

        # Test if the _has_divide_combine_pattern method works correctly
        function_node = ast.parse(structured_code["functions"][0]["source"])
        has_divide_combine = self.analyzer._has_divide_combine_pattern(function_node)
        self.assertTrue(has_divide_combine, "Divide and conquer pattern should be detected")

        # Test pattern identification with the mock structured_code
        identified_patterns = self.analyzer._identify_patterns(structured_code,code)
        self.assertGreater(len(identified_patterns["divide_conquer"]), 0)

    def test_master_worker_pattern_direct(self):
        """Test master-worker pattern detection directly without using the parser."""
        code = """def master_worker(tasks):
        results = []

        # Master distributes tasks
        for task in tasks:
            results.append(worker_process(task))

        return results
    """
        # Parse the code directly with ast
        tree = ast.parse(code)

        # Create a mock structured_code dictionary
        structured_code = {
            "functions": [
                {
                    "type": "function",
                    "lineno": 1,
                    "source": code
                }
            ]
        }

        # Test if the _has_task_distribution method works correctly
        function_node = ast.parse(structured_code["functions"][0]["source"])
        has_task_distribution = self.analyzer._has_task_distribution(function_node)
        self.assertTrue(has_task_distribution, "Master-worker pattern should be detected")

        # Test pattern identification with the mock structured_code
        identified_patterns = self.analyzer._identify_patterns(structured_code,code)
        self.assertGreater(len(identified_patterns["master_worker"]), 0)

    def test_scatter_gather_pattern_direct(self):
        """Test scatter-gather pattern detection directly without using the parser."""
        code = """def scatter_gather(data):
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
        # Parse the code directly with ast
        tree = ast.parse(code)

        # Create a mock structured_code dictionary
        structured_code = {
            "functions": [
                {
                    "type": "function",
                    "lineno": 1,
                    "source": code
                }
            ]
        }

        # Test if the _has_distribution_collection_pattern method works correctly
        function_node = ast.parse(structured_code["functions"][0]["source"])
        has_distribution_collection = self.analyzer._has_distribution_collection_pattern(function_node)
        self.assertTrue(has_distribution_collection, "Scatter-gather pattern should be detected")

        # Test pattern identification with the mock structured_code
        identified_patterns = self.analyzer._identify_patterns(structured_code,code)
        self.assertGreater(len(identified_patterns["scatter_gather"]), 0)
