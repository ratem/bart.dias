"""
BDiasPatternAnalyzer: Parallel Pattern Recognition Module for Bart.dIAs

This module implements a Pattern Characteristic Matrix that maps computational
structures to specific parallel patterns such as Pipeline, Stencil, Master-Worker,
etc. It focuses exclusively on pattern recognition and matching, without
overlapping with critical path analysis responsibilities.

Features:
- Maps computational structures to known parallel patterns
- Identifies data access patterns and communication requirements
- Analyzes synchronization points and dependencies
- Recommends appropriate partitioning strategies for each pattern
- Provides theoretical performance characteristics for identified patterns

Dependencies:
- BDiasParser for basic code structure identification
- ast (Python standard library)
"""

import ast
from typing import Dict, List, Set, Tuple, Any, Optional


class BDiasPatternAnalyzer:
    """
    Analyzes code structures to recognize higher-level parallel programming patterns.

    The Pattern Characteristic Matrix maps computational structures to specific
    parallel patterns and provides information about their properties.

    This class focuses exclusively on pattern recognition and matching, without
    overlapping with critical path analysis responsibilities.
    """

    def __init__(self, parser):
        """
        Initialize the pattern analyzer with a parser instance.

        Args:
            parser: BDiasParser instance that has already parsed the code
        """
        self.parser = parser

        # Define the Pattern Characteristic Matrix
        # This maps pattern names to their characteristics
        self.pattern_matrix = {
            "map": {
                "structure": ["independent_loop", "list_comprehension"],
                "data_access": "independent",
                "communication": "minimal",
                "synchronization": "start_end",
                "parallelism": "data_parallel",
                "suitable_partitioning": ["SDP", "SIP", "horizontal"],
                "performance": {
                    "work": "O(n)",
                    "span": "O(1)",
                    "parallelism": "O(n)"
                }
            },
            "pipeline": {
                "structure": ["sequential_loops_with_dependencies", "producer_consumer_pattern"],
                "data_access": "streaming",
                "communication": "neighbor_only",
                "synchronization": "pipelined",
                "parallelism": "task_parallel",
                "suitable_partitioning": ["TDP", "TIP"],
                "performance": {
                    "work": "O(n)",
                    "span": "O(s + n)",  # s = number of stages
                    "parallelism": "O(min(s, n/s))"  # s = number of stages
                }
            },
            "stencil": {
                "structure": ["nested_loops_with_neighbor_access"],
                "data_access": "neighbor_dependent",
                "communication": "neighbor_only",
                "synchronization": "per_iteration",
                "parallelism": "data_parallel",
                "suitable_partitioning": ["SDP", "horizontal"],
                "performance": {
                    "work": "O(n^d)",  # d = dimensions
                    "span": "O(t)",  # t = time steps
                    "parallelism": "O(n^d/t)"
                }
            },
            "master_worker": {
                "structure": ["task_distribution_loop", "work_queue_pattern"],
                "data_access": "centralized_distribution",
                "communication": "master_to_workers",
                "synchronization": "task_completion",
                "parallelism": "task_parallel",
                "suitable_partitioning": ["SIP", "horizontal", "hash"],
                "performance": {
                    "work": "O(n)",
                    "span": "O(n/p + c)",  # c = coordination overhead
                    "parallelism": "O(p)"
                }
            },
            "reduction": {
                "structure": ["accumulation_loop", "tree_reduction"],
                "data_access": "converging",
                "communication": "tree_based",
                "synchronization": "tree_levels",
                "parallelism": "logarithmic",
                "suitable_partitioning": ["SDP", "SIP"],
                "performance": {
                    "work": "O(n)",
                    "span": "O(log n)",
                    "parallelism": "O(n/log n)"
                }
            },
            "fork_join": {
                "structure": ["recursive_function", "parallel_sections"],
                "data_access": "independent_tasks",
                "communication": "at_fork_join_points",
                "synchronization": "join_points",
                "parallelism": "task_parallel",
                "suitable_partitioning": ["SIP", "horizontal"],
                "performance": {
                    "work": "O(n)",
                    "span": "O(d)",  # d = maximum recursion depth
                    "parallelism": "O(n/d)"
                }
            },
            "divide_conquer": {
                "structure": ["recursive_divide_function", "tree_recursion"],
                "data_access": "hierarchical",
                "communication": "tree_based",
                "synchronization": "tree_levels",
                "parallelism": "task_parallel",
                "suitable_partitioning": ["SIP", "horizontal"],
                "performance": {
                    "work": "O(n log n)",  # For typical cases like mergesort
                    "span": "O(log^2 n)",
                    "parallelism": "O(n)"
                }
            },
            "scatter_gather": {
                "structure": ["distribution_collection_pattern"],
                "data_access": "distributed_then_centralized",
                "communication": "all_to_one_one_to_all",
                "synchronization": "barrier",
                "parallelism": "data_parallel",
                "suitable_partitioning": ["SDP", "horizontal", "hash"],
                "performance": {
                    "work": "O(n)",
                    "span": "O(log p)",  # p = number of processors
                    "parallelism": "O(n/log p)"
                }
            }
        }

        # Pattern detection rules - map code structures to pattern indicators
        self.pattern_detection_rules = self._initialize_detection_rules()

    def _initialize_detection_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize the pattern detection rules that map code structures
        to pattern indicators.

        Returns:
            Dictionary mapping pattern names to detection criteria
        """
        return {
            "map": {
                "indicators": [
                    {"type": "loop", "has_dependencies": False},
                    {"type": "list_comprehension"}
                ],
                "confidence_threshold": 0.8
            },
            "pipeline": {
                "indicators": [
                    {"type": "sequential_loops", "has_producer_consumer": True},
                    {"type": "loop", "has_stage_dependencies": True}
                ],
                "confidence_threshold": 0.7
            },
            "stencil": {
                "indicators": [
                    {"type": "nested_loop", "has_neighbor_access": True},
                    {"type": "loop", "has_grid_dependencies": True}
                ],
                "confidence_threshold": 0.7
            },
            "master_worker": {
                "indicators": [
                    {"type": "loop", "has_task_distribution": True},
                    {"type": "function", "has_work_queue": True}
                ],
                "confidence_threshold": 0.6
            },
            "reduction": {
                "indicators": [
                    {"type": "loop", "has_accumulation": True},
                    {"type": "recursive_function", "has_combining_pattern": True}
                ],
                "confidence_threshold": 0.8
            },
            "fork_join": {
                "indicators": [
                    {"type": "recursive_function", "has_independent_tasks": True},
                    {"type": "function", "has_parallel_sections": True}
                ],
                "confidence_threshold": 0.7
            },
            "divide_conquer": {
                "indicators": [
                    {"type": "recursive_function", "has_divide_combine": True},
                    {"type": "function", "has_tree_recursion": True}
                ],
                "confidence_threshold": 0.8
            },
            "scatter_gather": {
                "indicators": [
                    {"type": "combo", "has_distribution_collection": True},
                    {"type": "function", "has_all_to_one_one_to_all": True}
                ],
                "confidence_threshold": 0.6
            }
        }

    def analyze(self, code: str) -> Dict[str, Any]:
        """
        Analyze the code to identify parallel patterns based on the
        Pattern Characteristic Matrix.

        Args:
            code: The Python code to analyze

        Returns:
            Dictionary containing identified patterns and their characteristics
        """
        # Get the structured code from the parser
        structured_code = self.parser.parse(code)

        # Identify patterns in the structured code
        identified_patterns = self._identify_patterns(structured_code)

        # Analyze pattern characteristics
        pattern_analysis = self._analyze_pattern_characteristics(identified_patterns, structured_code)

        return pattern_analysis

    def _identify_patterns(self, structured_code: Dict[str, List]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify parallel patterns in the structured code based on the detection rules.

        Args:
            structured_code: Dictionary containing structured code from the parser

        Returns:
            Dictionary mapping pattern names to lists of identified instances
        """
        identified_patterns = {pattern: [] for pattern in self.pattern_matrix}

        # Check for Map pattern (independent loops and list comprehensions)
        for loop in structured_code.get("loops", []):
            if self._is_independent_loop(loop):
                identified_patterns["map"].append({
                    "type": "loop",
                    "lineno": loop["lineno"],
                    "confidence": 0.9,
                    "details": loop
                })

        for list_comp in structured_code.get("list_comprehensions", []):
            identified_patterns["map"].append({
                "type": "list_comprehension",
                "lineno": list_comp["lineno"],
                "confidence": 0.95,
                "details": list_comp
            })

        # Check for Pipeline pattern
        for combo in structured_code.get("combos", []):
            if "for_in_while" in combo.get("type", "") or "while_with_for" in combo.get("type", ""):
                if self._has_producer_consumer_pattern(combo):
                    identified_patterns["pipeline"].append({
                        "type": "combo",
                        "lineno": combo["lineno"],
                        "confidence": 0.8,
                        "details": combo
                    })

        # Check for Stencil pattern
        for loop in structured_code.get("loops", []):
            if loop.get("type") == "nested_for" and self._has_neighbor_access(loop):
                identified_patterns["stencil"].append({
                    "type": "nested_loop",
                    "lineno": loop["lineno"],
                    "confidence": 0.85,
                    "details": loop
                })

        # Check for Master-Worker pattern
        for function in structured_code.get("functions", []):
            if self._has_task_distribution(function):
                identified_patterns["master_worker"].append({
                    "type": "function",
                    "lineno": function["lineno"],
                    "confidence": 0.7,
                    "details": function
                })

        # Check for Reduction pattern
        for loop in structured_code.get("loops", []):
            if self._has_accumulation_pattern(loop):
                identified_patterns["reduction"].append({
                    "type": "loop",
                    "lineno": loop["lineno"],
                    "confidence": 0.85,
                    "details": loop
                })

        # Check for Fork-Join pattern
        for function in structured_code.get("functions", []):
            if "recursive" in function.get("type", "") and self._has_independent_tasks(function):
                identified_patterns["fork_join"].append({
                    "type": "recursive_function",
                    "lineno": function["lineno"],
                    "confidence": 0.75,
                    "details": function
                })

        # Check for Divide & Conquer pattern
        for function in structured_code.get("functions", []):
            if "recursive" in function.get("type", "") and self._has_divide_combine_pattern(function):
                identified_patterns["divide_conquer"].append({
                    "type": "recursive_function",
                    "lineno": function["lineno"],
                    "confidence": 0.85,
                    "details": function
                })

        # Check for Scatter-Gather pattern
        for combo in structured_code.get("combos", []):
            if self._has_distribution_collection_pattern(combo):
                identified_patterns["scatter_gather"].append({
                    "type": "combo",
                    "lineno": combo["lineno"],
                    "confidence": 0.7,
                    "details": combo
                })

        return identified_patterns

    def _analyze_pattern_characteristics(self, identified_patterns: Dict[str, List],
                                         structured_code: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze the characteristics of identified patterns.

        Args:
            identified_patterns: Dictionary of identified pattern instances
            structured_code: Dictionary containing structured code from the parser

        Returns:
            Dictionary containing pattern analysis results
        """
        results = {
            "identified_patterns": {},
            "recommended_partitioning": {},
            "performance_characteristics": {}
        }

        # Process each pattern type
        for pattern_name, instances in identified_patterns.items():
            if not instances:
                continue

            # Filter instances by confidence threshold
            threshold = self.pattern_detection_rules[pattern_name]["confidence_threshold"]
            confident_instances = [inst for inst in instances if inst["confidence"] >= threshold]

            if not confident_instances:
                continue

            # Add to results
            results["identified_patterns"][pattern_name] = confident_instances

            # Get pattern characteristics from the matrix
            pattern_info = self.pattern_matrix[pattern_name]

            # Add partitioning recommendations
            results["recommended_partitioning"][pattern_name] = {
                "strategies": pattern_info["suitable_partitioning"],
                "rationale": f"The {pattern_name} pattern works best with {', '.join(pattern_info['suitable_partitioning'])} partitioning strategies due to its {pattern_info['data_access']} data access pattern and {pattern_info['communication']} communication requirements."
            }

            # Add performance characteristics
            results["performance_characteristics"][pattern_name] = {
                "work": pattern_info["performance"]["work"],
                "span": pattern_info["performance"]["span"],
                "parallelism": pattern_info["performance"]["parallelism"],
                "communication_overhead": pattern_info["communication"],
                "synchronization_points": pattern_info["synchronization"]
            }

        return results

    def suggest_patterns_for_code_block(self, code_block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a specific code block and suggest appropriate parallel patterns.

        This method is designed to be used by the integration layer to get pattern
        suggestions for a specific code block, such as a bottleneck identified
        in critical path analysis.

        Args:
            code_block: Dictionary containing code block information

        Returns:
            List of suggested patterns with confidence scores and rationales
        """
        suggested_patterns = []

        # Extract code block characteristics
        node_type = code_block.get('type', '')
        source_code = code_block.get('source', '')

        # Parse the code block to analyze its structure
        try:
            block_node = ast.parse(source_code)

            # Analyze computational structure
            has_nested_loops = self._has_nested_loops(block_node)
            has_neighbor_access = self._has_neighbor_access(block_node)
            has_reduction_pattern = self._has_reduction_pattern(block_node)
            has_producer_consumer_pattern = self._has_producer_consumer_pattern(block_node)
            has_independent_tasks = self._has_independent_tasks(block_node)
            has_divide_combine_pattern = self._has_divide_combine_pattern(block_node)
            has_distribution_collection_pattern = self._has_distribution_collection_pattern(block_node)
            has_task_distribution = self._has_task_distribution(block_node)
            has_accumulation_pattern = self._has_accumulation_pattern(block_node)
            is_independent_loop = self._is_independent_loop(block_node)

            # Match with patterns from the Pattern Characteristic Matrix
            # Map pattern
            if is_independent_loop or (node_type == 'for_loop' and not has_nested_loops and not has_reduction_pattern):
                suggested_patterns.append({
                    "pattern": "map",
                    "confidence": 0.9 if is_independent_loop else 0.7,
                    "rationale": "The code block contains independent operations that can be executed in parallel.",
                    "partitioning": ["SDP", "SIP", "horizontal"],
                    "description": "Apply the same operation independently to each element in a dataset.",
                    "speedup_potential": "Linear (O(p)) with sufficient data parallelism."
                })

            # Stencil pattern
            if has_nested_loops and has_neighbor_access:
                suggested_patterns.append({
                    "pattern": "stencil",
                    "confidence": 0.85,
                    "rationale": "The code block contains nested loops with neighbor access patterns.",
                    "partitioning": ["SDP", "horizontal"],
                    "description": "Update array elements based on neighboring elements.",
                    "speedup_potential": "O(n) for 2D problems with n² elements."
                })

            # Pipeline pattern
            if has_producer_consumer_pattern:
                suggested_patterns.append({
                    "pattern": "pipeline",
                    "confidence": 0.8,
                    "rationale": "The code block shows a producer-consumer pattern with data flowing between stages.",
                    "partitioning": ["TDP", "TIP"],
                    "description": "Divide a task into a series of stages, with data flowing through stages.",
                    "speedup_potential": "Limited by the slowest stage, up to O(s) for s stages."
                })

            # Reduction pattern
            if has_reduction_pattern or has_accumulation_pattern:
                suggested_patterns.append({
                    "pattern": "reduction",
                    "confidence": 0.85,
                    "rationale": "The code block accumulates results, suitable for parallel reduction.",
                    "partitioning": ["SDP", "SIP"],
                    "description": "Combine multiple elements into a single result using an associative operation.",
                    "speedup_potential": "O(log n) critical path with O(n/log n) parallelism."
                })

            # Divide and Conquer pattern
            if has_divide_combine_pattern:
                suggested_patterns.append({
                    "pattern": "divide_conquer",
                    "confidence": 0.85,
                    "rationale": "The code block recursively divides work and combines results.",
                    "partitioning": ["SIP", "horizontal"],
                    "description": "Recursively break down a problem into smaller subproblems.",
                    "speedup_potential": "O(n) for many problems with O(n log n) work."
                })

            # Fork-Join pattern
            if has_independent_tasks:
                suggested_patterns.append({
                    "pattern": "fork_join",
                    "confidence": 0.75,
                    "rationale": "The code block creates independent tasks that can be executed in parallel.",
                    "partitioning": ["SIP", "horizontal"],
                    "description": "Split a task into subtasks, execute them in parallel, then join results.",
                    "speedup_potential": "Limited by the critical path length."
                })

            # Master-Worker pattern
            if has_task_distribution:
                suggested_patterns.append({
                    "pattern": "master_worker",
                    "confidence": 0.7,
                    "rationale": "The code block distributes independent tasks to workers.",
                    "partitioning": ["SIP", "horizontal", "hash"],
                    "description": "A master process distributes tasks to worker processes.",
                    "speedup_potential": "Near-linear with good load balancing."
                })

            # Scatter-Gather pattern
            if has_distribution_collection_pattern:
                suggested_patterns.append({
                    "pattern": "scatter_gather",
                    "confidence": 0.7,
                    "rationale": "The code block distributes data for parallel processing and then collects results.",
                    "partitioning": ["SDP", "horizontal", "hash"],
                    "description": "Distribute data across processes, process independently, then collect results.",
                    "speedup_potential": "Near-linear with minimal communication overhead."
                })

            # If no specific pattern was identified, suggest a generic approach
            if not suggested_patterns:
                # Check if it might be a sequential bottleneck
                if "while" in source_code.lower() and "for" in source_code.lower():
                    suggested_patterns.append({
                        "pattern": "pipeline",
                        "confidence": 0.5,
                        "rationale": "The code block contains nested loops that might benefit from pipelining.",
                        "partitioning": ["TDP", "TIP"],
                        "description": "Transform sequential stages into a pipeline.",
                        "speedup_potential": "Limited by dependencies between iterations."
                    })
                else:
                    suggested_patterns.append({
                        "pattern": "task_parallelism",
                        "confidence": 0.4,
                        "rationale": "Consider restructuring the code to expose more parallelism.",
                        "partitioning": ["SIP"],
                        "description": "Identify independent tasks that can be executed concurrently.",
                        "speedup_potential": "Depends on the amount of parallelism exposed."
                    })

            # Sort by confidence
            suggested_patterns.sort(key=lambda x: x["confidence"], reverse=True)

        except SyntaxError:
            # If parsing fails, return a generic suggestion
            suggested_patterns.append({
                "pattern": "unknown",
                "confidence": 0.3,
                "rationale": "Unable to parse code structure for detailed analysis.",
                "partitioning": ["SDP"]
            })

        return suggested_patterns

    def get_pattern_characteristics(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get the characteristics of a specific pattern from the matrix.

        Args:
            pattern_name: Name of the pattern

        Returns:
            Dictionary containing pattern characteristics
        """
        return self.pattern_matrix.get(pattern_name, {})

    def suggest_partitioning_strategy(self, pattern_name: str) -> List[str]:
        """
        Suggest appropriate partitioning strategies for a specific pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            List of recommended partitioning strategies
        """
        pattern_info = self.pattern_matrix.get(pattern_name, {})
        return pattern_info.get("suitable_partitioning", [])

    # Pattern detection helper methods
    def _is_independent_loop(self, loop: Dict[str, Any]) -> bool:
        """Check if a loop has independent iterations."""
        # Implementation would analyze loop body for dependencies
        # This is a simplified placeholder
        return not any(dep in loop.get("type", "") for dep in ["nested", "recursive"])

    def _has_producer_consumer_pattern(self, combo: Dict[str, Any]) -> bool:
        """Check if a combo has a producer-consumer pattern."""
        # Implementation would look for data flow between loop iterations
        # This is a simplified placeholder
        return "for_in_while" in combo.get("type", "") or "while_with_for" in combo.get("type", "")

    def _has_neighbor_access(self, loop: Dict[str, Any]) -> bool:
        """Check if a loop accesses neighboring elements (stencil pattern)."""
        # Implementation would analyze array access patterns
        # This is a simplified placeholder
        return loop.get("type") == "nested_for"

    def _has_task_distribution(self, function: Dict[str, Any]) -> bool:
        """Check if a function distributes tasks (master-worker pattern)."""
        # Implementation would look for task distribution patterns
        # This is a simplified placeholder
        return "function" in function.get("type", "")

    def _has_accumulation_pattern(self, loop: Dict[str, Any]) -> bool:
        """Check if a loop accumulates results (reduction pattern)."""
        # Implementation would look for accumulation variables
        # This is a simplified placeholder
        return "loop" in loop.get("type", "")

    def _has_independent_tasks(self, function: Dict[str, Any]) -> bool:
        """Check if a function creates independent tasks (fork-join pattern)."""
        # Implementation would analyze task dependencies
        # This is a simplified placeholder
        return "recursive" in function.get("type", "")

    def _has_divide_combine_pattern(self, function: Dict[str, Any]) -> bool:
        """Check if a function has divide-and-conquer pattern."""
        # Implementation would look for recursive division and combination
        # This is a simplified placeholder
        return "recursive" in function.get("type", "")

    def _has_distribution_collection_pattern(self, combo: Dict[str, Any]) -> bool:
        """Check if a combo has distribution-collection pattern (scatter-gather)."""
        # Implementation would look for data distribution followed by collection
        # This is a simplified placeholder
        return "combo" in combo.get("type", "")

    # Additional pattern detection helpers for AST-based analysis
    def _has_nested_loops(self, node):
        """Check if a node contains nested loops."""
        outer_loops = []
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.For, ast.While)):
                outer_loops.append(subnode)

        for loop in outer_loops:
            for subnode in ast.walk(loop):
                if isinstance(subnode, (ast.For, ast.While)) and subnode != loop:
                    return True
        return False

    def _has_reduction_pattern(self, node):
        """Check if a node contains a reduction pattern (accumulation)."""
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.AugAssign):  # +=, *=, etc.
                return True
            elif isinstance(subnode, ast.Assign):
                if isinstance(subnode.targets[0], ast.Name) and \
                        isinstance(subnode.value, ast.BinOp) and \
                        isinstance(subnode.value.left, ast.Name) and \
                        subnode.targets[0].id == subnode.value.left.id:
                    return True  # x = x + ...
        return False
