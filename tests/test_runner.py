import unittest

loader = unittest.TestLoader()
start_dir = 'tests'
suite = loader.discover(start_dir)


def run_tests():
    """Run all tests for critical path and pattern analyzer modules."""
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_tests()
