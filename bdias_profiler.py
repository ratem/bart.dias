"""
BDiasProfiler: Static Code Profiling Module for Bart.dIAs

This module provides static profiling capabilities for Python code, identifying
computationally intensive sections without executing the code.

Features:
- Performs static analysis to estimate computational intensity of code blocks
- Identifies and ranks computationally intensive functions, loops, and list comprehensions
- Uses heuristics to estimate intensity based on code structure and complexity
- Analyzes nested loops, function calls, and recursive patterns
- Provides context-aware adjustments for different types of operations
- Performs basic data flow analysis to detect early exits and redundant operations

Classes:
- BDiasProfiler: Main class for static code profiling

Dependencies:
- ast (Python standard library)
- typing

Limitations:
- Cannot accurately predict runtime performance for all code patterns
- Does not account for data-dependent performance characteristics
- Heuristic-based estimates may not perfectly align with actual execution costs
"""


import ast
from typing import List, Dict, Any


class BDiasProfiler:
    """
    Statically profiles Python code to identify computationally intensive sections
    without executing the code. Uses heuristic-based analysis only.
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

        Args:
            node: AST node to analyze

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

        # Apply context-specific adjustments
        intensity *= self._adjust_for_context(node)

        # Apply data flow analysis
        intensity *= self._analyze_data_flow(node)

        return intensity

    def _adjust_for_context(self, node):
        """
        Adjust computational intensity based on domain-specific context.

        Args:
            node: AST node to analyze

        Returns:
            Adjustment factor for the intensity
        """
        adjustment = 1.0

        # Check for operations on different data structures
        for subnode in ast.walk(node):
            # Math operations
            if isinstance(subnode, ast.BinOp):
                adjustment *= 1.2

            # Function calls to known intensive operations
            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Attribute):
                if hasattr(subnode.func, 'attr'):
                    # Check for numpy operations (potentially GPU-accelerated)
                    if subnode.func.attr in ['dot', 'matmul', 'multiply', 'exp', 'log']:
                        adjustment *= 5.0
                    # Check for other intensive operations
                    elif subnode.func.attr in ['sort', 'filter', 'map']:
                        adjustment *= 2.0

            # List/dict comprehensions
            if isinstance(subnode, (ast.ListComp, ast.DictComp, ast.SetComp)):
                adjustment *= 3.0

            # Exception handling (try/except blocks)
            if isinstance(subnode, ast.Try):
                adjustment *= 1.5

        return adjustment

    def _analyze_data_flow(self, node):
        """
        Analyze data flow to detect early exits and redundant operations.

        Args:
            node: AST node to analyze

        Returns:
            Adjustment factor for the intensity based on data flow analysis
        """
        adjustment = 1.0

        # Check for early exits (break, return, continue)
        early_exits = 0
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.Break, ast.Return, ast.Continue)):
                early_exits += 1

        # If there are early exits, reduce intensity based on their position
        if early_exits > 0:
            # Simple heuristic: more early exits = more potential for early termination
            adjustment *= max(0.5, 1.0 - (0.1 * early_exits))

        # Check for redundant operations
        redundant_ops = self._detect_redundant_operations(node)
        if redundant_ops > 0:
            adjustment *= max(0.5, 1.0 - (0.1 * redundant_ops))

        # Check for constant conditions in loops
        if isinstance(node, (ast.For, ast.While)) and self._has_constant_condition(node):
            # Loop with constant condition might be optimized away or run very few times
            adjustment *= 0.5

        return adjustment

    def _detect_redundant_operations(self, node):
        """
        Detect potentially redundant operations in the code.

        Args:
            node: AST node to analyze

        Returns:
            Number of potentially redundant operations
        """
        redundant_ops = 0

        # Simple check for repeated calculations
        # This is a very basic implementation - a real one would track variable assignments
        calculations = {}

        for subnode in ast.walk(node):
            if isinstance(subnode, ast.BinOp):
                # Convert the operation to a string representation
                op_str = ast.unparse(subnode)
                if op_str in calculations:
                    redundant_ops += 1
                else:
                    calculations[op_str] = True

        return redundant_ops

    def _has_constant_condition(self, node):
        """
        Check if a loop has a constant condition that might lead to few iterations.

        Args:
            node: AST node to analyze

        Returns:
            True if the loop has a constant condition, False otherwise
        """
        if isinstance(node, ast.While) and hasattr(node, 'test'):
            # Check if the condition is a constant (True/False/Number)
            return isinstance(node.test, (ast.Constant, ast.Num, ast.NameConstant))

        return False
