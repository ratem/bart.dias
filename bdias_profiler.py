import ast
from typing import List, Dict, Any


class BDiasProfiler:
    """
    Statically profiles Python code to identify computationally intensive sections
    without executing the code.
    """

    def __init__(self, max_results: int = 5):
        """
        Initialize the profiler.

        Args:
            max_results: Maximum number of time-consuming sections to display
        """
        self.max_results = max_results

    def profile_code(self, parser, code: str) -> List[Dict[str, Any]]:
        """
        Statically profile the given code and identify computationally intensive sections.

        Args:
            parser: The BDiasParser instance that has already parsed the code
            code: The Python code to analyze

        Returns:
            List of dictionaries containing profiling information for the most computationally
            intensive sections
        """
        # Get the AST from the parser
        tree = parser.tree

        # Extract code blocks (functions, loops, etc.)
        code_blocks = self._extract_code_blocks(tree)

        # Analyze each code block for computational intensity
        for block in code_blocks:
            block["intensity"] = self._estimate_computational_intensity(block["node"])

            # Add source code for the block
            block["source"] = ast.unparse(block["node"])

            # Remove the AST node to make the result more readable
            del block["node"]

        # Sort blocks by computational intensity (descending)
        code_blocks.sort(key=lambda x: x["intensity"], reverse=True)

        return code_blocks[:min(self.max_results, len(code_blocks))]

    def _extract_code_blocks(self, tree):
        """Extract code blocks (functions, loops) from AST."""
        code_blocks = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                code_blocks.append({
                    "type": "function",
                    "name": node.name,
                    "lineno": node.lineno,
                    "end_lineno": self._get_end_line(node),
                    "node": node
                })
            elif isinstance(node, ast.For):
                code_blocks.append({
                    "type": "for_loop",
                    "name": f"for_loop_line_{node.lineno}",
                    "lineno": node.lineno,
                    "end_lineno": self._get_end_line(node),
                    "node": node
                })
            elif isinstance(node, ast.While):
                code_blocks.append({
                    "type": "while_loop",
                    "name": f"while_loop_line_{node.lineno}",
                    "lineno": node.lineno,
                    "end_lineno": self._get_end_line(node),
                    "node": node
                })
            elif isinstance(node, ast.ListComp):
                code_blocks.append({
                    "type": "list_comprehension",
                    "name": f"list_comp_line_{node.lineno}",
                    "lineno": node.lineno,
                    "end_lineno": self._get_end_line(node),
                    "node": node
                })

        return code_blocks

    def _get_end_line(self, node):
        """Get the end line number of a node."""
        if hasattr(node, 'end_lineno'):
            return node.end_lineno

        # Estimate end line for older Python versions
        max_line = node.lineno
        for child in ast.iter_child_nodes(node):
            if hasattr(child, 'lineno'):
                child_end = self._get_end_line(child)
                max_line = max(max_line, child_end)

        return max_line

    def _estimate_computational_intensity(self, node):
        """
        Estimate the computational intensity of a code block based on static analysis.

        This method analyzes various factors that contribute to computational complexity:
        - Loop nesting depth
        - Number of function calls
        - Recursion
        - Number of operations
        - Complexity of data structures

        Returns:
            A numerical score representing the estimated computational intensity
        """
        intensity = 1.0

        # Adjust based on code size
        node_size = len(ast.unparse(node).splitlines())
        intensity *= (1 + 0.1 * node_size)

        # Adjust based on loop nesting
        if isinstance(node, (ast.For, ast.While)):
            # Base score for a loop
            intensity *= 10

            # Check for nested loops
            nested_loops = sum(1 for _ in ast.walk(node)
                               if isinstance(_, (ast.For, ast.While)) and _ != node)
            if nested_loops > 0:
                intensity *= (5 ** nested_loops)

        # Adjust based on function calls
        function_calls = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Call))
        if function_calls > 0:
            intensity *= (2 * function_calls)

        # Adjust for recursive calls
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            recursive_calls = sum(1 for _ in ast.walk(node)
                                  if isinstance(_, ast.Call) and
                                  isinstance(_.func, ast.Name) and
                                  _.func.id == function_name)
            if recursive_calls > 0:
                intensity *= (10 * recursive_calls)

        # Adjust for operations that are typically computationally intensive
        for subnode in ast.walk(node):
            # Math operations
            if isinstance(subnode, ast.BinOp):
                intensity *= 1.2

            # Function calls to known intensive operations
            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Attribute):
                if hasattr(subnode.func, 'attr'):
                    # Check for numpy operations
                    if subnode.func.attr in ['dot', 'matmul', 'multiply', 'exp', 'log']:
                        intensity *= 5

            # List/dict comprehensions
            if isinstance(subnode, (ast.ListComp, ast.DictComp, ast.SetComp)):
                intensity *= 3

            # Exception handling (try/except blocks)
            if isinstance(subnode, ast.Try):
                intensity *= 1.5

        return intensity

    def get_user_selection(self, ranked_blocks, code_lines):
        """
        Present the ranked code blocks to the user and get their selection.

        Args:
            ranked_blocks: List of ranked code blocks
            code_lines: List of code lines for displaying context

        Returns:
            The selected code block
        """
        print("\nTop computationally intensive sections in your code:")
        for i, block in enumerate(ranked_blocks):
            block_type = block["type"].replace("_", " ").title()
            block_name = block["name"]
            start_line = block["lineno"]
            end_line = block["end_lineno"]

            # Show code snippet
            code_snippet = "\n".join(code_lines[start_line - 1:end_line])

            print(f"{i + 1}. {block_type}: {block_name} (Lines {start_line}-{end_line})")
            print(f"   Estimated computational intensity: {block['intensity']:.2f}")
            print(f"   Code snippet:")
            for line in code_snippet.splitlines()[:3]:  # Show first 3 lines
                print(f"      {line}")
            if len(code_snippet.splitlines()) > 3:
                print("      ...")

        while True:
            try:
                selection = int(input(f"\nSelect a section to optimize (1-{len(ranked_blocks)}): "))
                if 1 <= selection <= len(ranked_blocks):
                    return ranked_blocks[selection - 1]
                else:
                    print(f"Please enter a number between 1 and {len(ranked_blocks)}")
            except ValueError:
                print("Please enter a valid number")
