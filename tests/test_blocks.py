import unittest
import ast
from bdias_parser import BDiasParser
from bdias_code_gen import BDiasCodeGen
from bdias_assist import BDiasAssist


class BDiasBlocksTest(unittest.TestCase):
    def setUp(self):
        # Initialize the parser for each test
        self.parser = BDiasParser()

        # Sample explanations and partitioning suggestions for testing
        self.explanations = {
            'loop': "Loop explanation",
            'nested loop': "Nested loop explanation",
            'function': "Function explanation",
            'recursive function': "Recursive function explanation",
            'function call': "Function call explanation",
            'list comprehension': "List comprehension explanation",
            'loop and function': "Loop and function explanation",
            'for_with_recursive_call': "For with recursive call explanation",
            'while_with_for': "While with for explanation",
            'for_in_while': "For in while explanation",
            'for_with_loop_functions': "For with loop functions explanation",
            'while_with_loop_functions': "While with loop functions explanation"
        }

        self.partitioning_suggestions = {
            'loop': "Loop partitioning",
            'nested loop': "Nested loop partitioning",
            'function': "Function partitioning",
            'recursive function': "Recursive function partitioning",
            'function call': "Function call partitioning",
            'list comprehension': "List comprehension partitioning",
            'loop and function': "Loop and function partitioning",
            'for_with_recursive_call': "For with recursive call partitioning",
            'while_with_for': "While with for partitioning",
            'for_in_while': "For in while partitioning",
            'for_with_loop_functions': "For with loop functions partitioning",
            'while_with_loop_functions': "While with loop functions partitioning"
        }

        self.code_generator = BDiasCodeGen(self.explanations, self.partitioning_suggestions)
        self.assistant = BDiasAssist(self.parser, self.code_generator)

    # Tests for Enhanced Dependency Analysis

    def test_build_dependency_graph(self):
        """Test the build_dependency_graph method."""
        code = """
def test_function():
    a = 5
    b = a + 2
    c = b * 3
    return c
        """
        tree = ast.parse(code)
        function_node = tree.body[0]

        dependency_graph = self.parser.build_dependency_graph(function_node)

        # Check that dependencies are correctly identified
        self.assertIn('a', dependency_graph)
        self.assertIn('b', dependency_graph)
        self.assertIn('c', dependency_graph)

        # Check specific dependencies
        self.assertTrue(len(dependency_graph['a']) == 0 or 'b' in dependency_graph['a'])
        self.assertTrue(len(dependency_graph['b']) == 0 or 'c' in dependency_graph['b'])
        self.assertEqual(len(dependency_graph['c']), 0)  # c has no dependencies

    def test_analyze_data_flow(self):
        """Test the analyze_data_flow method."""
        code = """
def test_function():
    a = 5
    b = a + 2
    a = b * 3
    return a
        """
        tree = ast.parse(code)
        function_node = tree.body[0]

        data_flow_deps = self.parser.analyze_data_flow(function_node)

        # Check that read-after-write dependencies are detected
        self.assertTrue(any(var == 'a' for _, _, var in data_flow_deps))
        self.assertTrue(any(var == 'b' for _, _, var in data_flow_deps))

    def test_track_cross_function_dependencies(self):
        """Test the track_cross_function_dependencies method."""
        code = """
global_var = 0

def modify_global():
    global global_var
    global_var += 1

def test_function():
    modify_global()
    return global_var
        """
        self.parser.tree = ast.parse(code)
        function_node = self.parser.tree.body[2]  # test_function

        modified_globals = self.parser.track_cross_function_dependencies(function_node)

        # Check that global variable modification is detected
        self.assertIn('global_var', modified_globals)

    def test_analyze_recursive_calls(self):
        """Test the analyze_recursive_calls method."""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
        """
        tree = ast.parse(code)
        function_node = tree.body[0]

        recursive_analysis = self.parser.analyze_recursive_calls(function_node)

        # Check that recursive calls are detected
        self.assertTrue(len(recursive_analysis["calls"]) > 0)
        # Check parallelizability analysis
        self.assertIn("parallelizable", recursive_analysis)

    # Tests for Combo Pattern Detection

    def test_for_with_recursive_call(self):
        """Test detection of for loops with recursive function calls."""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def test_function():
    for i in range(10):
        result = fibonacci(i)
        print(result)
        """
        structured_code = self.parser.parse(code)

        # Check that the combo pattern is detected
        self.assertIn("combos", structured_code)
        self.assertTrue(any("for_with_recursive_call" in combo.get("type", "")
                            for combo in structured_code["combos"]))

    def test_while_with_for(self):
        """Test detection of while loops containing for loops."""
        code = """
def test_function():
    i = 0
    while i < 5:
        for j in range(i):
            print(j)
        i += 1
        """
        structured_code = self.parser.parse(code)

        # Check that the combo pattern is detected
        self.assertIn("combos", structured_code)
        self.assertTrue(any("while_with_for" in combo.get("type", "")
                            for combo in structured_code["combos"]))

    def test_for_in_while(self):
        """Test detection of for loops inside while loops."""
        code = """
def test_function():
    i = 0
    while i < 5:
        for j in range(i):
            print(j)
        i += 1
        """
        structured_code = self.parser.parse(code)

        # Check that the combo pattern is detected
        self.assertIn("combos", structured_code)
        self.assertTrue(any("for_in_while" in combo.get("type", "")
                            for combo in structured_code["combos"]) or
                        any("while_with_for" in combo.get("type", "")
                            for combo in structured_code["combos"]))

    def test_nested_loops_varying_depth(self):
        """Test detection of nested loops with varying depths."""
        code = """
def test_function():
    for i in range(5):
        print(i)
        for j in range(i):
            print(j)
            for k in range(j):
                print(k)
        """
        structured_code = self.parser.parse(code)

        # Check that nested loops are detected
        self.assertTrue(
            any(combo.get("nesting_depth", 0) > 1 for combo in structured_code.get("combos", [])) or
            any(loop.get("nesting_depth", 0) > 1 for loop in structured_code.get("loops", []))
        )

    def test_for_with_loop_functions(self):
        """Test detection of for loops calling functions that contain loops."""
        code = """
def function_with_loop(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result

def test_function():
    for i in range(5):
        result = function_with_loop(i)
        print(result)
        """
        structured_code = self.parser.parse(code)

        # Check that the combo pattern is detected
        self.assertIn("combos", structured_code)
        self.assertTrue(any("for_with_loop_functions" in combo.get("type", "")
                            for combo in structured_code["combos"]))

    def test_while_with_loop_functions(self):
        """Test detection of while loops calling functions that contain loops."""
        code = """
def function_with_loop(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result

def test_function():
    i = 0
    while i < 5:
        result = function_with_loop(i)
        print(result)
        i += 1
        """
        structured_code = self.parser.parse(code)

        # Check that the combo pattern is detected
        self.assertIn("combos", structured_code)
        self.assertTrue(any("while_with_loop_functions" in combo.get("type", "")
                            for combo in structured_code["combos"]))

    # Tests for Code Generation for Combo Patterns

    def test_code_generation_for_with_recursive_call(self):
        """Test code generation for loops with recursive function calls."""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def test_function():
    for i in range(10):
        result = fibonacci(i)
        print(result)
        """
        structured_code = self.parser.parse(code)
        suggestions = self.code_generator.generate_suggestions(structured_code)

        # Check that a suggestion is generated for the combo pattern
        self.assertTrue(any(suggestion.get("opportunity_type") == "for_with_recursive_call"
                            for suggestion in suggestions))

    def test_code_generation_while_with_for(self):
        """Test code generation for while loops containing for loops."""
        code = """
def test_function():
    i = 0
    while i < 5:
        for j in range(i):
            print(j)
        i += 1
        """
        structured_code = self.parser.parse(code)
        suggestions = self.code_generator.generate_suggestions(structured_code)

        # Check that a suggestion is generated for the combo pattern
        self.assertTrue(any(suggestion.get("opportunity_type") == "while_with_for"
                            for suggestion in suggestions))


if __name__ == '__main__':
    unittest.main()
