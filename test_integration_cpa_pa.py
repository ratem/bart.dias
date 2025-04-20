import unittest
import ast

import networkx as nx

from bdias_critical_path import BDiasCriticalPathAnalyzer
from bdias_pattern_analyzer import BDiasPatternAnalyzer
from bdias_parser import BDiasParser
from bdias_assist import BDiasAssist
from bdias_code_gen import BDiasCodeGen


class MockCodeGen:
    """Mock code generator for testing."""

    def __init__(self):
        self.EXPLANATIONS = {}
        self.PARTITIONING_SUGGESTIONS = {}


class TestIntegration(unittest.TestCase):
    """Test cases for the integration between critical path and pattern analyzers."""

    def setUp(self):
        """Set up test cases."""
        self.parser = BDiasParser()
        self.code_gen = MockCodeGen()
        self.assist = BDiasAssist(self.parser, self.code_gen)
        self.critical_path_analyzer = BDiasCriticalPathAnalyzer()
        self.pattern_analyzer = BDiasPatternAnalyzer(self.parser)

    def test_critical_path_bottleneck_pattern_analysis(self):
        """Test that bottlenecks from critical path analysis can be analyzed for patterns."""
        code = """
def compute_sum(n):
    result = 0
    for i in range(n):
        result += i
    return result

def main():
    n = 1000
    result = compute_sum(n)
    print(result)
"""
        # Parse the code
        tree = ast.parse(code)
        mock_parser = type('', (), {'tree': tree})()

        # Perform critical path analysis
        cp_results = self.critical_path_analyzer.analyze(mock_parser, code)

        # Check that bottlenecks were identified
        self.assertGreater(len(cp_results["bottlenecks"]), 0)

        # For each bottleneck, perform pattern analysis
        for bottleneck in cp_results["bottlenecks"]:
            # Extract the source code of the bottleneck
            bottleneck_code = bottleneck["source"]

            # Analyze the bottleneck for patterns
            bottleneck_tree = ast.parse(bottleneck_code)
            mock_bottleneck_parser = type('', (), {'tree': bottleneck_tree})()

            # Use the pattern analyzer to analyze the bottleneck
            pattern_results = self.pattern_analyzer.analyze(bottleneck_code)

            # Check that pattern analysis produced results
            self.assertIsNotNone(pattern_results)
            self.assertIn("identified_patterns", pattern_results)

    def test_integrated_analysis_workflow(self):
        """Test the integrated analysis workflow."""
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
        # Parse the code
        structured_code = self.parser.parse(code)

        # Perform critical path analysis
        cp_results = self.critical_path_analyzer.analyze(self.parser, code)

        # Check that bottlenecks were identified
        self.assertGreater(len(cp_results["bottlenecks"]), 0)

        # For each bottleneck, perform pattern analysis
        for bottleneck in cp_results["bottlenecks"]:
            # Analyze the bottleneck for patterns
            pattern_results = self.pattern_analyzer.analyze(bottleneck["source"])

            # Check that pattern analysis produced results
            self.assertIsNotNone(pattern_results)
            self.assertIn("identified_patterns", pattern_results)

            # Store pattern suggestions in the bottleneck
            bottleneck["suggested_patterns"] = pattern_results.get("identified_patterns", {})

        # Check that the workflow produced meaningful results
        self.assertGreater(len(cp_results["bottlenecks"]), 0)
        for bottleneck in cp_results["bottlenecks"]:
            self.assertIn("suggested_patterns", bottleneck)

    def test_theoretical_metrics_validation(self):
        """Test validation of theoretical metrics from Träff's book."""
        # Create a DAG with known work and span
        self.critical_path_analyzer.dag = nx.DiGraph()
        entry_node = self.critical_path_analyzer.entry_node
        exit_node = self.critical_path_analyzer.exit_node

        self.critical_path_analyzer.dag.add_node(entry_node, work=0, span=0)
        self.critical_path_analyzer.dag.add_node(exit_node, work=0, span=0)
        self.critical_path_analyzer.dag.add_node("A", work=100, span=10)
        self.critical_path_analyzer.dag.add_node("B", work=200, span=20)
        self.critical_path_analyzer.dag.add_node("C", work=150, span=15)

        # Create edges for a fork-join pattern
        self.critical_path_analyzer.dag.add_edge(entry_node, "A")
        self.critical_path_analyzer.dag.add_edge("A", "B")
        self.critical_path_analyzer.dag.add_edge("A", "C")
        self.critical_path_analyzer.dag.add_edge("B", exit_node)
        self.critical_path_analyzer.dag.add_edge("C", exit_node)

        # Calculate metrics
        total_work = self.critical_path_analyzer._calculate_total_work()
        critical_path, span = self.critical_path_analyzer._find_critical_path()
        parallelism = total_work / span if span > 0 else float('inf')

        # Validate against theoretical expectations
        # Work Law: T_p >= T_1/p
        for p in [1, 2, 4, 8, 16]:
            min_time_p = total_work / p
            self.assertGreaterEqual(min_time_p, span)  # T_p >= T_∞

        # Depth Law: T_p >= T_∞
        self.assertEqual(span, 30)  # 0 + 10 + 20 + 0

        # Parallelism: T_1/T_∞
        expected_parallelism = 450 / 30  # (100 + 200 + 150) / 30
        self.assertEqual(parallelism, expected_parallelism)

        # Amdahl's Law: Maximum speedup <= 1/s where s is sequential fraction
        sequential_fraction = span / total_work
        max_speedup = 1 / sequential_fraction
        self.assertEqual(max_speedup, parallelism)  # Should be equal by definition
