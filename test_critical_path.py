import unittest
import ast
import networkx as nx
from bdias_critical_path import BDiasCriticalPathAnalyzer
from bdias_parser import BDiasParser


class MockParser:
    """Mock parser for testing critical path analyzer."""

    def __init__(self, tree):
        self.tree = tree


class TestCriticalPathAnalyzer(unittest.TestCase):
    """Test cases for the BDiasCriticalPathAnalyzer class."""

    def setUp(self):
        """Set up test cases."""
        self.analyzer = BDiasCriticalPathAnalyzer()
        self.parser = BDiasParser()

    def test_dag_construction(self):
        """Test DAG construction from simple code."""
        code = """
def simple_function():
    a = 1
    b = 2
    c = a + b
    return c
"""
        tree = ast.parse(code)
        mock_parser = MockParser(tree)
        self.analyzer._build_dag(tree, code)

        # Check that the DAG has been constructed
        self.assertTrue(isinstance(self.analyzer.dag, nx.DiGraph))
        # Check entry and exit nodes
        self.assertIn(self.analyzer.entry_node, self.analyzer.dag.nodes())
        self.assertIn(self.analyzer.exit_node, self.analyzer.dag.nodes())
        # Check that at least one function node exists
        function_nodes = [n for n in self.analyzer.dag.nodes() if 'function_' in str(n)]
        self.assertGreater(len(function_nodes), 0)

    def test_work_span_calculation(self):
        """Test work and span calculation for a simple node."""
        code = """
for i in range(10):
    a = i * i
"""
        node = ast.parse(code)
        work, span = self.analyzer._calculate_work_span(node)

        # Check that work and span are positive
        self.assertGreater(work, 0)
        self.assertGreater(span, 0)
        # Check that work is at least as large as span
        self.assertGreaterEqual(work, span)

    def test_critical_path_identification(self):
        """Test critical path identification in a simple DAG."""
        # Create a simple DAG with a known critical path
        self.analyzer.dag = nx.DiGraph()
        self.analyzer.dag.add_node(self.analyzer.entry_node, span=0)
        self.analyzer.dag.add_node(self.analyzer.exit_node, span=0)
        self.analyzer.dag.add_node("A", span=5)
        self.analyzer.dag.add_node("B", span=10)
        self.analyzer.dag.add_node("C", span=3)

        # Create edges: entry -> A -> B -> exit and entry -> C -> exit
        self.analyzer.dag.add_edge(self.analyzer.entry_node, "A")
        self.analyzer.dag.add_edge("A", "B")
        self.analyzer.dag.add_edge("B", self.analyzer.exit_node)
        self.analyzer.dag.add_edge(self.analyzer.entry_node, "C")
        self.analyzer.dag.add_edge("C", self.analyzer.exit_node)

        path, span = self.analyzer._find_critical_path()

        # The critical path should be entry -> A -> B -> exit with span 15
        self.assertIn("A", path)
        self.assertIn("B", path)
        self.assertNotIn("C", path)
        self.assertEqual(span, 15)

    def test_bottleneck_identification(self):
        """Test bottleneck identification from a critical path."""
        # Create a critical path with known bottlenecks
        critical_path = [self.analyzer.entry_node, "function_10", "for_loop_20", "while_loop_30",
                         self.analyzer.exit_node]

        # Set up node attributes in the DAG
        self.analyzer.dag = nx.DiGraph()
        self.analyzer.dag.add_node(self.analyzer.entry_node, type="control", span=0, work=0)
        self.analyzer.dag.add_node(self.analyzer.exit_node, type="control", span=0, work=0)
        self.analyzer.dag.add_node("function_10", type="function", name="test_func", lineno=10, span=5, work=20,
                                   source="def test_func():\n    pass")
        self.analyzer.dag.add_node("for_loop_20", type="for_loop", name="for_loop_line_20", lineno=20, span=10, work=30,
                                   source="for i in range(10):\n    pass")
        self.analyzer.dag.add_node("while_loop_30", type="while_loop", name="while_loop_line_30", lineno=30, span=8,
                                   work=25, source="while condition:\n    pass")

        bottlenecks = self.analyzer._identify_bottlenecks(critical_path)

        # Check that bottlenecks are identified correctly
        self.assertEqual(len(bottlenecks), 3)  # Should return top 3 bottlenecks
        self.assertEqual(bottlenecks[0]['type'], 'for_loop')  # Highest span should be first
        self.assertEqual(bottlenecks[0]['span'], 10)
        self.assertEqual(bottlenecks[1]['type'], 'while_loop')
        self.assertEqual(bottlenecks[2]['type'], 'function')

    def test_analyze_full_code(self):
        """Test full analysis of a code sample."""
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
        mock_parser = MockParser(ast.parse(code))
        results = self.analyzer.analyze(mock_parser, code)

        # Check that all expected keys are in the results
        expected_keys = ["dag", "total_work", "critical_path", "critical_path_length",
                         "parallelism", "sequential_fraction", "amdahl_max_speedup", "bottlenecks"]
        for key in expected_keys:
            self.assertIn(key, results)

        # Check that metrics are reasonable
        self.assertGreater(results["total_work"], 0)
        self.assertGreater(results["critical_path_length"], 0)
        self.assertGreater(results["parallelism"], 0)
        self.assertGreaterEqual(results["sequential_fraction"], 0)
        self.assertLessEqual(results["sequential_fraction"], 1)
        self.assertGreaterEqual(results["amdahl_max_speedup"], 1)
        self.assertGreater(len(results["bottlenecks"]), 0)

    def test_theoretical_metrics(self):
        """Test theoretical metrics calculation for a known DAG."""
        # Create a DAG with known work and span
        self.analyzer.dag = nx.DiGraph()
        self.analyzer.dag.add_node(self.analyzer.entry_node, work=0, span=0)
        self.analyzer.dag.add_node(self.analyzer.exit_node, work=0, span=0)
        self.analyzer.dag.add_node("A", work=10, span=5)
        self.analyzer.dag.add_node("B", work=20, span=10)
        self.analyzer.dag.add_node("C", work=15, span=3)

        # Create edges
        self.analyzer.dag.add_edge(self.analyzer.entry_node, "A")
        self.analyzer.dag.add_edge("A", "B")
        self.analyzer.dag.add_edge("B", self.analyzer.exit_node)
        self.analyzer.dag.add_edge(self.analyzer.entry_node, "C")
        self.analyzer.dag.add_edge("C", self.analyzer.exit_node)

        # Calculate metrics
        total_work = self.analyzer._calculate_total_work()
        critical_path, span = self.analyzer._find_critical_path()
        parallelism = total_work / span if span > 0 else float('inf')
        sequential_fraction = span / total_work if total_work > 0 else 1.0

        # Check metrics against expected values
        self.assertEqual(total_work, 45)  # 10 + 20 + 15
        self.assertEqual(span, 15)  # 0 + 5 + 10 + 0
        self.assertEqual(parallelism, 3.0)  # 45/15
        self.assertEqual(sequential_fraction, 1 / 3)  # 15/45
