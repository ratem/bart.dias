import unittest
from test_critical_path import TestCriticalPathAnalyzer
from test_pattern_analyzer import TestPatternAnalyzer
from test_integration_cpa_pa import TestIntegration
from test_traff_patterns import TestPatternSpecificStructures
def run_tests():
    """Run all tests for critical path and pattern analyzer modules."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCriticalPathAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestPatternAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    test_suite.addTest(unittest.makeSuite(TestPatternSpecificStructures))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result


if __name__ == "__main__":
    run_tests()
