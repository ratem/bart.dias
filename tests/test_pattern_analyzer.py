import unittest
import ast
from bdias_pattern_analyzer import BDiasPatternAnalyzer
from bdias_parser import BDiasParser


class TestPatternAnalyzer(unittest.TestCase):
    """Test cases for the BDiasPatternAnalyzer class."""

    def setUp(self):
        """Set up test cases."""
        self.parser = BDiasParser()
        self.analyzer = BDiasPatternAnalyzer(self.parser)

    def test_pattern_matrix_initialization(self):
        """Test that the pattern matrix is correctly initialized."""
        expected_patterns = [
            "map_reduce", "pipeline", "stencil", "pool_worker",
            "fork_join", "divide_conquer", "scatter_gather"
        ]
        for pattern in expected_patterns:
            self.assertIn(pattern, self.analyzer.pattern_matrix)

        for pattern, characteristics in self.analyzer.pattern_matrix.items():
            self.assertIn("structure", characteristics)
            self.assertIn("data_access", characteristics)
            self.assertIn("communication", characteristics)
            self.assertIn("synchronization", characteristics)
            self.assertIn("parallelism", characteristics)
            self.assertIn("suitable_partitioning", characteristics)
            self.assertIn("performance", characteristics)

    def test_map_reduce_pattern_detection(self):
        """Test detection of Map-Reduce pattern."""
        code = """
def map_reduce_function():
    # Map phase
    result = []
    for i in range(100):
        result.append(i * i)

    # Reduce phase
    total = 0
    for item in result:
        total += item
    return total
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code, code)
        self.assertGreater(len(identified_patterns["map_reduce"]), 0)
        self.assertGreaterEqual(identified_patterns["map_reduce"][0]["confidence"], 0.7)

    def test_stencil_pattern_direct(self):
        """Test stencil pattern detection directly without using the parser."""
        code = """
def stencil_function(grid):
    n = len(grid)
    new_grid = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(1, n-1):
        for j in range(1, n-1):
            new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] +
                             grid[i][j-1] + grid[i][j+1]) / 4
    return new_grid
"""
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
        loop_node = ast.parse(structured_code["loops"][0]["source"])
        has_neighbor = self.analyzer._has_neighbor_access(loop_node)
        self.assertTrue(has_neighbor, "Stencil pattern should have neighbor access")
        identified_patterns = self.analyzer._identify_patterns(structured_code, code)
        self.assertGreater(len(identified_patterns["stencil"]), 0)

    def test_pipeline_pattern_direct(self):
        """Test pipeline pattern detection directly without using the parser."""
        code = """
def pipeline_function():
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
        structured_code = {
            "functions": [
                {
                    "type": "function",
                    "lineno": 1,
                    "source": code
                }
            ]
        }
        function_node = ast.parse(structured_code["functions"][0]["source"])
        has_producer_consumer = self.analyzer._has_producer_consumer_pattern(function_node)
        self.assertTrue(has_producer_consumer, "Pipeline pattern should be detected")
        identified_patterns = self.analyzer._identify_patterns(structured_code, code)
        self.assertGreater(len(identified_patterns["pipeline"]), 0)

    def test_pattern_characteristics_analysis(self):
        """Test analysis of pattern characteristics."""
        code = """
def test_function():
    # Map pattern
    result1 = []
    for i in range(100):
        result1.append(i * i)

    # Reduction pattern
    result2 = 0
    for i in range(100):
        result2 += i

    return result1, result2
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code, code)
        pattern_analysis = self.analyzer._analyze_pattern_characteristics(identified_patterns, structured_code)
        self.assertIn("identified_patterns", pattern_analysis)
        self.assertIn("recommended_partitioning", pattern_analysis)
        self.assertIn("performance_characteristics", pattern_analysis)
        if "map_reduce" in pattern_analysis["recommended_partitioning"]:
            map_partitioning = pattern_analysis["recommended_partitioning"]["map_reduce"]
            self.assertIn("strategies", map_partitioning)
            self.assertIn("SDP", map_partitioning["strategies"])
            self.assertIn("SIP", map_partitioning["strategies"])
            self.assertEqual(len(map_partitioning["strategies"]), 2)

    def test_analyze_full_code(self):
        """Test full analysis of a code sample."""
        code = """
def compute_map_reduce(data):
    # Map phase
    mapped = []
    for item in data:
        mapped.append(item * item)

    # Reduce phase
    result = 0
    for item in mapped:
        result += item
    return result
"""
        analysis_results = self.analyzer.analyze(code)
        self.assertIn("identified_patterns", analysis_results)
        self.assertIn("recommended_partitioning", analysis_results)
        self.assertIn("performance_characteristics", analysis_results)
        self.assertGreater(len(analysis_results["identified_patterns"]), 0)
