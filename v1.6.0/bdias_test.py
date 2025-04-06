import ast
from bdias_parser import BDiasParser
from bdias_code_gen import BDiasCodeGen  # Import the codegen class


def test_bdias_parser():
    """Tests that the core methods are not None"""
    parser = BDiasParser()

    # Basic test code
    code = """
def my_function(x):
    return x * 2

for i in range(10):
    print(i)

[x**2 for x in range(5)]
"""

    structured_code = parser.parse(code)

    # Check if code was parsed
    assert structured_code is not None, "Code parsing failed"

    # Check structure
    assert isinstance(structured_code, dict), "Structured code is not a dictionary"
    assert "loops" in structured_code, "Loops key is missing"
    assert "functions" in structured_code, "Functions key is missing"
    assert "list_comprehensions" in structured_code, "List comprehensions key is missing"

    # Check number of elements
    assert len(structured_code["loops"]) == 1, "Incorrect number of loops"
    assert len(structured_code["functions"]) == 1, "Incorrect number of functions"
    assert len(structured_code["list_comprehensions"]) == 1, "Incorrect number of list comprehensions"

    print("Basic BDiasParser tests passed!")

def test_bdias_codegen_generate_suggestions():
    """Test some very base generations"""
    #Sample from what we expect to get from the code.
    EXPLANATIONS = {
        'loop': "Basic for loops",
        'function': "For Functions",
        'recursive function': "For a specific Recursive Call",
        'function call': "For generic calls",
        'list comprehension': "For list comprehensions"
    }

    # Partitioning suggestions corresponding to the indices used in find_parallelization_opportunities
    PARTITIONING_SUGGESTIONS = {
        'loop': "basic partitioning",
        'function': "split",
        'recursive function': "recursive",
        'function call': "simple",
        'list comprehension': "hash"
    }
    code_generator = BDiasCodeGen(EXPLANATIONS, PARTITIONING_SUGGESTIONS)
    code = """
def my_function(x):
    return x * 2

for i in range(10):
    print(i)

[x**2 for x in range(5)]
"""
    parser = BDiasParser()
    structured_code = parser.parse(code)
    suggestions = code_generator.generate_suggestions(structured_code)
    assert suggestions is not None
    print ("Tests on some code functions all passed!")
if __name__ == '__main__':
    # Run the tests
    test_bdias_parser()
    test_bdias_codegen_generate_suggestions()