import unittest


def run_tests():
    """Run all tests for critical path and pattern analyzer modules."""
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_tests()

'''
def run_tests():
    """Run all tests for Bart.dIAs."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases from test discovery
    loader = unittest.TestLoader()
    start_dir = 'tests'
    discovered_tests = loader.discover(start_dir)
    test_suite.addTest(discovered_tests)
    
    # Add pattern code generation tests
    from test_pattern_codegen import TestMapPatternCodegen
    test_suite.addTest(unittest.makeSuite(TestMapPatternCodegen))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

'''