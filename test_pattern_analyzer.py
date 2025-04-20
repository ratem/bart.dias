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
        # Check that all expected patterns are in the matrix
        expected_patterns = ["map", "pipeline", "stencil", "master_worker",
                             "reduction", "fork_join", "divide_conquer", "scatter_gather"]
        for pattern in expected_patterns:
            self.assertIn(pattern, self.analyzer.pattern_matrix)

        # Check that each pattern has the required characteristics
        for pattern, characteristics in self.analyzer.pattern_matrix.items():
            self.assertIn("structure", characteristics)
            self.assertIn("data_access", characteristics)
            self.assertIn("communication", characteristics)
            self.assertIn("synchronization", characteristics)
            self.assertIn("parallelism", characteristics)
            self.assertIn("suitable_partitioning", characteristics)
            self.assertIn("performance", characteristics)

    def test_map_pattern_detection(self):
        """Test detection of Map pattern."""
        code = """
def map_function():
    result = []
    for i in range(100):
        result.append(i * i)
    return result
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)

        # Check that Map pattern is identified
        self.assertGreater(len(identified_patterns["map"]), 0)
        # Check that the confidence is high
        self.assertGreaterEqual(identified_patterns["map"][0]["confidence"], 0.7)

    def test_reduction_pattern_detection(self):
        """Test detection of Reduction pattern."""
        code = """
def reduction_function():
    result = 0
    for i in range(100):
        result += i
    return result
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)

        # Check that Reduction pattern is identified
        self.assertGreater(len(identified_patterns["reduction"]), 0)
        # Check that the confidence is high
        self.assertGreaterEqual(identified_patterns["reduction"][0]["confidence"], 0.7)

    def test_stencil_pattern_detection(self):
        """Test detection of Stencil pattern."""
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
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)

        # Check that Stencil pattern is identified
        self.assertGreater(len(identified_patterns["stencil"]), 0)
        # Check that the confidence is high
        self.assertGreaterEqual(identified_patterns["stencil"][0]["confidence"], 0.7)

    def test_pipeline_pattern_detection(self):
        """Test detection of Pipeline pattern."""
        code = """
def pipeline_function():
    result = []
    buffer = []
    while not done:
        # Producer stage
        for i in range(10):
            buffer.append(produce_item())

        # Consumer stage
        for item in buffer:
            result.append(process_item(item))
        buffer = []
    return result
"""
        structured_code = self.parser.parse(code)
        identified_patterns = self.analyzer._identify_patterns(structured_code)

        # Check that Pipeline pattern is identified
        self.assertGreater(len(identified_patterns["pipeline"]), 0)
        # Check that the confidence is high
        self.assertGreaterEqual(identified_patterns["pipeline"][0]["confidence"], 0.7)

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
        identified_patterns = self.analyzer._identify_patterns(structured_code)
        pattern_analysis = self.analyzer._analyze_pattern_characteristics(identified_patterns, structured_code)

        # Check that analysis contains expected sections
        self.assertIn("identified_patterns", pattern_analysis)
        self.assertIn("recommended_partitioning", pattern_analysis)
        self.assertIn("performance_characteristics", pattern_analysis)

        # Check that Map pattern has appropriate partitioning recommendations
        if "map" in pattern_analysis["recommended_partitioning"]:
            map_partitioning = pattern_analysis["recommended_partitioning"]["map"]
            self.assertIn("strategies", map_partitioning)
            self.assertIn("SDP", map_partitioning["strategies"])
            self.assertIn("SIP", map_partitioning["strategies"])
            self.assertIn("horizontal", map_partitioning["strategies"])

        # Check that Reduction pattern has appropriate performance characteristics
        if "reduction" in pattern_analysis["performance_characteristics"]:
            reduction_perf = pattern_analysis["performance_characteristics"]["reduction"]
            self.assertIn("work", reduction_perf)
            self.assertIn("span", reduction_perf)
            self.assertIn("parallelism", reduction_perf)

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

        # Check that analysis contains expected sections
        self.assertIn("identified_patterns", analysis_results)
        self.assertIn("recommended_partitioning", analysis_results)
        self.assertIn("performance_characteristics", analysis_results)

        # Check that at least one pattern is identified
        self.assertGreater(len(analysis_results["identified_patterns"]), 0)
