"""
BDiasCriticalPathAnalyzer: DAG-based Critical Path Analysis Module for Bart.dIAs

This module implements explicit Directed Acyclic Graph (DAG) modeling and analysis
for Python code, based on theoretical foundations from parallel computing theory.
It constructs a computational DAG representation of code and analyzes its inherent
parallelism characteristics.

Theoretical Foundation:
- Work-span model (Section 2.3.1 in Träff): Represents computation as a DAG where
  nodes are tasks and edges are dependencies
- Critical path analysis (Theorem 2.5): Identifies the longest path of dependent
  operations that limits parallel execution
- Amdahl's Law applications (Section 2.2.7): Calculates maximum theoretical speedup
  based on sequential fraction
- Brent's Theorem principles (Section 2.2.4): Relates work, span, and processor count

Key Metrics Calculated:
- T₁: Total work - sum of computational costs across all operations (Definition 2.7)
- T∞: Critical path length/Span - longest chain of dependent operations (Depth Law)
- P = T₁/T∞: Maximum parallelism - theoretical upper bound on speedup
- Sequential fraction: Proportion of work that must be done sequentially
- Amdahl's maximum speedup: 1/(sequential fraction)

Features:
- Constructs a DAG representation of Python code using AST analysis
- Calculates work and span metrics using operation counting
- Identifies the critical path that limits parallelization
- Calculates theoretical parallelism and Amdahl's Law bounds
- Identifies sequential bottlenecks for targeted optimization
- Provides visualization of the DAG with critical path highlighting

Classes:
- BDiasCriticalPathAnalyzer: Main class for DAG construction and critical path analysis

Dependencies:
- ast: For parsing and analyzing Python code
- networkx: For graph operations and critical path algorithms
"""

import ast
import networkx as nx
from typing import Dict, List, Tuple, Any, Set, Optional


class BDiasCriticalPathAnalyzer:
    """
    Analyzes Python code to construct a DAG representation and identify
    the critical path that limits parallel execution.

    Based on concepts from Träff's "Lectures on Parallel Computing":
    - Work-span model (Section 2.3.1)
    - Critical path analysis (Theorem 2.5)
    - Amdahl's Law applications (Section 2.2.7)
    """

    def __init__(self):
        """Initialize the critical path analyzer."""
        self.dag = nx.DiGraph()
        self.entry_node = "__ENTRY__"
        self.exit_node = "__EXIT__"

        # Operation costs based on type (from Träff's PRAM-like cost model)
        self.OP_COSTS = {
            ast.BinOp: 1,  # Binary operations
            ast.UnaryOp: 1,  # Unary operations
            ast.Compare: 1,  # Comparisons
            ast.Call: 5,  # Function calls (higher cost)
            ast.Subscript: 1,  # Array/list access
            ast.Attribute: 1,  # Object attribute access
            ast.For: 2,  # Loop overhead
            ast.While: 2,  # Loop overhead
            ast.If: 1,  # Conditional overhead
            ast.ListComp: 3,  # List comprehension
            ast.DictComp: 4  # Dictionary comprehension
        }

    def analyze(self, parser, code: str) -> Dict[str, Any]:
        """
        Analyze the given code to construct a DAG and calculate parallelism metrics.

        This is the main entry point for critical path analysis. It builds a DAG
        representation of the code, calculates work and span metrics, identifies
        the critical path, and computes theoretical parallelism bounds.

        Args:
            parser: The BDiasParser instance that has already parsed the code
            code: The Python code to analyze as a string

        Returns:
            Dictionary containing DAG analysis results including:
            - total_work: Sum of computational costs across all operations (T₁)
            - critical_path_length: Length of the critical path (T∞)
            - parallelism: Ratio of total work to critical path length (T₁/T∞)
            - sequential_fraction: Proportion of work that must be done sequentially
            - amdahl_max_speedup: Maximum theoretical speedup (1/sequential_fraction)
            - bottlenecks: List of operations on the critical path
            - critical_path: Sequence of nodes forming the critical path
            - dag: The constructed DAG for visualization or further analysis
        """

        # Get the AST from the parser
        tree = parser.tree

        # Extract code blocks and their dependencies
        self._build_dag(tree, code)

        # Calculate metrics
        total_work = self._calculate_total_work()
        critical_path, span = self._find_critical_path()
        parallelism = total_work / span if span > 0 else float('inf')

        # Estimate sequential fraction for Amdahl's Law
        sequential_fraction = span / total_work if total_work > 0 else 1.0
        amdahl_max_speedup = 1 / sequential_fraction if sequential_fraction > 0 else float('inf')

        # Identify bottlenecks (nodes on the critical path)
        bottlenecks = self._identify_bottlenecks(critical_path)

        return {
            "dag": self.dag,
            "total_work": total_work,
            "critical_path": critical_path,
            "critical_path_length": span,
            "parallelism": parallelism,
            "sequential_fraction": sequential_fraction,
            "amdahl_max_speedup": amdahl_max_speedup,
            "bottlenecks": bottlenecks
        }

    def _build_dag(self, tree, code: str) -> None:
        """
        Construct a Directed Acyclic Graph (DAG) from the AST.

        Builds a computational DAG where nodes represent code blocks (functions, loops)
        and edges represent dependencies between them. Each node is annotated with
        work and span metrics calculated from the corresponding AST subtree.

        Args:
            tree: The AST of the code
            code: The original code string for reference
        """
        # Add entry and exit nodes
        self.dag.add_node(self.entry_node, type="control", work=0, span=0)
        self.dag.add_node(self.exit_node, type="control", work=0, span=0)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(tree)

        # Add nodes for each code block
        for block in code_blocks:
            node_id = f"{block['type']}_{block['lineno']}"
            work, span = self._calculate_work_span(block['node'])

            self.dag.add_node(node_id,
                              type=block['type'],
                              name=block.get('name', ''),
                              lineno=block['lineno'],
                              end_lineno=block['end_lineno'],
                              work=work,
                              span=span,
                              source=ast.unparse(block['node']))

        # Analyze dependencies between blocks
        self._analyze_dependencies(code_blocks)

        # Connect orphan nodes (no incoming edges) to entry
        for node in self.dag.nodes():
            if node != self.entry_node and node != self.exit_node:
                if self.dag.in_degree(node) == 0:
                    self.dag.add_edge(self.entry_node, node)

                # Connect nodes with no outgoing edges to exit
                if self.dag.out_degree(node) == 0:
                    self.dag.add_edge(node, self.exit_node)

    def _extract_code_blocks(self, tree):
        """
        Extract code blocks (functions, loops) from AST.

        Identifies significant computational units in the code that will become
        nodes in the DAG. Currently extracts functions, for loops, while loops,
        and list comprehensions.

        Args:
            tree: The AST to analyze

        Returns:
            List of dictionaries containing information about each code block
        """
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

    def _calculate_work_span(self, node):
        """
        Calculate work and span using operation counting and DAG analysis.

        Implements the work-span model from Section 2.3.1 of Träff's book.
        Work is the total computational cost of all operations, while span
        is the longest chain of dependent operations.

        Args:
            node: AST node to analyze

        Returns:
            Tuple of (work, span) where:
            - work: Total computational cost of all operations in the node
            - span: Length of the longest chain of dependent operations
        """
        # Create a DAG for this node
        local_dag = nx.DiGraph()
        node_counter = 0
        parent_map = {}  # Maps AST nodes to DAG nodes

        # Add entry node
        entry_id = f"entry_{node_counter}"
        local_dag.add_node(entry_id, cost=0)
        node_counter += 1

        # Process the AST to build the local DAG
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.stmt, ast.expr)):
                # Skip the root node (already processed as entry)
                if subnode == node:
                    parent_map[subnode] = entry_id
                    continue

                # Create a node for this AST node
                node_id = f"node_{node_counter}"
                node_counter += 1

                # Determine the cost of this operation
                cost = self.OP_COSTS.get(type(subnode), 0.5)
                local_dag.add_node(node_id, cost=cost)

                # Find parent in the AST
                for parent_ast in ast.walk(node):
                    if any(child is subnode for child in ast.iter_child_nodes(parent_ast)):
                        if parent_ast in parent_map:
                            # Add edge from parent to this node
                            local_dag.add_edge(parent_map[parent_ast], node_id)
                            break

                # Store mapping from AST node to DAG node
                parent_map[subnode] = node_id

        # Calculate work (sum of all costs)
        work = sum(data['cost'] for _, data in local_dag.nodes(data=True))

        # Calculate span (longest path)
        span = 0
        if local_dag.nodes:
            try:
                # Find longest path using topological sort
                longest_paths = {n: 0 for n in local_dag.nodes()}

                for n in nx.topological_sort(local_dag):
                    for succ in local_dag.successors(n):
                        longest_paths[succ] = max(
                            longest_paths[succ],
                            longest_paths[n] + local_dag.nodes[succ]['cost']
                        )

                span = max(longest_paths.values())
            except nx.NetworkXUnfeasible:
                # Handle cycles (shouldn't happen in valid Python AST)
                span = work

        return work, span

    def _analyze_dependencies(self, code_blocks):
        """
        Analyze dependencies between code blocks and add edges to the DAG.

        Identifies data and control dependencies between code blocks based on
        variable definitions and uses, following Bernstein's conditions for
        parallelism (Section 2.3.3 in Träff).

        Args:
            code_blocks: List of code blocks extracted from the AST
        """
        # Sort blocks by line number for deterministic processing
        sorted_blocks = sorted(code_blocks, key=lambda x: x['lineno'])

        # Build a mapping of variable definitions and uses
        var_defs = {}  # Maps variables to the nodes that define them
        var_uses = {}  # Maps variables to the nodes that use them

        # First pass: collect variable definitions and uses
        for block in sorted_blocks:
            node_id = f"{block['type']}_{block['lineno']}"

            # Analyze variable definitions and uses in this block
            defs, uses = self._extract_vars(block['node'])

            # Record variable definitions
            for var in defs:
                if var not in var_defs:
                    var_defs[var] = []
                var_defs[var].append(node_id)

            # Record variable uses
            for var in uses:
                if var not in var_uses:
                    var_uses[var] = []
                var_uses[var].append(node_id)

        # Second pass: create edges based on dependencies
        for var, use_nodes in var_uses.items():
            if var in var_defs:
                for def_node in var_defs[var]:
                    for use_node in use_nodes:
                        # Only add edge if definition comes before use
                        def_lineno = int(def_node.split('_')[-1])
                        use_lineno = int(use_node.split('_')[-1])
                        if def_lineno < use_lineno and def_node != use_node:
                            self.dag.add_edge(def_node, use_node, type="data_dependency", var=var)

        # Add control dependencies (e.g., if a block is nested within another)
        for i, outer in enumerate(sorted_blocks):
            outer_id = f"{outer['type']}_{outer['lineno']}"
            outer_start = outer['lineno']
            outer_end = outer['end_lineno']

            for j, inner in enumerate(sorted_blocks):
                if i == j:
                    continue

                inner_id = f"{inner['type']}_{inner['lineno']}"
                inner_start = inner['lineno']

                # If inner block starts within outer block's range, add control dependency
                if outer_start < inner_start < outer_end:
                    self.dag.add_edge(outer_id, inner_id, type="control_dependency")

    def _extract_vars(self, node) -> Tuple[Set[str], Set[str]]:
        """
        Extract variables defined and used in an AST node.

        Analyzes an AST node to identify variables that are defined (written to)
        and used (read from) within the node, which is essential for dependency analysis.

        Args:
            node: AST node to analyze

        Returns:
            Tuple of (defined_variables, used_variables) as sets of variable names
        """
        defined_vars = set()
        used_vars = set()

        for subnode in ast.walk(node):
            # Variable definitions
            if isinstance(subnode, ast.Assign):
                for target in subnode.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                defined_vars.add(elt.id)

            # Variable uses
            elif isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                used_vars.add(subnode.id)

        return defined_vars, used_vars

    def _calculate_total_work(self) -> float:
        """
        Calculate the total work (T₁) as the sum of work of all nodes.

        Implements the Work Law from Section 2.2.4 of Träff's book.

        Returns:
            Total work value as a float
        """
        return sum(data.get('work', 0) for _, data in self.dag.nodes(data=True))

    def _find_critical_path(self) -> Tuple[List[str], float]:
        """
        Find the critical path (path with highest total span) from entry to exit.

        Implements the Depth Law from Section 2.2.4 of Träff's book.
        The critical path is the longest chain of dependent operations that
        limits the potential parallelism of the code.

        Returns:
            Tuple of (critical_path_node_list, total_path_span) where:
            - critical_path_node_list: List of node IDs forming the critical path
            - total_path_span: Sum of span values along the critical path
        """
        # Use networkx to find the critical path
        # We negate the span values to use shortest_path with negative weights
        for u, v in self.dag.edges():
            self.dag.edges[u, v]['weight'] = -self.dag.nodes[v].get('span', 0)

        try:
            path = nx.shortest_path(self.dag, source=self.entry_node, target=self.exit_node, weight='weight')
            path_span = sum(self.dag.nodes[node].get('span', 0) for node in path)
            return path, path_span
        except nx.NetworkXNoPath:
            return [], 0

    def _identify_bottlenecks(self, critical_path: List[str]) -> List[Dict[str, Any]]:
        """
        Identify bottlenecks in the critical path (nodes with high span).

        Bottlenecks are operations on the critical path that contribute significantly
        to the sequential fraction of the computation, as described in Section 2.2.7
        of Träff's book on sequential bottlenecks.

        Args:
            critical_path: List of node IDs in the critical path

        Returns:
            List of dictionaries containing bottleneck information, sorted by span
        """
        bottlenecks = []

        for node in critical_path:
            if node == self.entry_node or node == self.exit_node:
                continue

            data = self.dag.nodes[node]
            span = data.get('span', 0)
            work = data.get('work', 0)

            # Consider nodes with high span as bottlenecks
            if span > 0:
                bottlenecks.append({
                    'node_id': node,
                    'type': data.get('type', ''),
                    'name': data.get('name', ''),
                    'lineno': data.get('lineno', 0),
                    'span': span,
                    'work': work,
                    'source': data.get('source', '')
                })

        # Sort bottlenecks by span (descending)
        bottlenecks.sort(key=lambda x: x['span'], reverse=True)

        return bottlenecks[:3]  # Return top 3 bottlenecks

    def visualize_dag(self, output_file: str = None) -> None:
        """
        Visualize the DAG using matplotlib and networkx.

        Args:
            output_file: Optional file path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Create position layout
            pos = nx.spring_layout(self.dag)

            # Draw nodes with different colors based on type
            node_colors = []
            for node in self.dag.nodes():
                if node == self.entry_node or node == self.exit_node:
                    node_colors.append('lightgray')
                elif self.dag.nodes[node].get('type') == 'function':
                    node_colors.append('lightblue')
                elif 'loop' in self.dag.nodes[node].get('type', ''):
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightsalmon')

            # Draw the graph
            nx.draw(self.dag, pos, with_labels=True, node_color=node_colors,
                    node_size=500, font_size=8, arrows=True)

            # Add node labels with work/span values
            node_labels = {node: f"{node}\nW:{data.get('work', 0):.1f} S:{data.get('span', 0):.1f}"
                           for node, data in self.dag.nodes(data=True)}
            nx.draw_networkx_labels(self.dag, pos, labels=node_labels, font_size=6)

            plt.title("Code Dependency Graph with Work/Span Analysis")

            if output_file:
                plt.savefig(output_file)
            else:
                plt.show()

        except ImportError:
            print("Matplotlib or networkx not available for visualization.")

    def generate_report(self, analysis: Dict) -> str:
        """
        Generate a report of the critical path analysis.

        Args:
            analysis: The analysis results from the analyze method

        Returns:
            A formatted string report
        """
        report = [
            "=== Critical Path Analysis Results ===",
            "",
            f"Total Work (T₁): {analysis['total_work']:.2f}",
            f"Critical Path Length (T∞): {analysis['critical_path_length']:.2f}",
            f"Inherent Parallelism (T₁/T∞): {analysis['parallelism']:.2f}",
            "",
            f"Sequential Fraction (Amdahl's Law): {analysis['sequential_fraction']:.4f} ({analysis['sequential_fraction'] * 100:.2f}%)",
            f"Maximum Theoretical Speedup: {analysis['amdahl_max_speedup']:.2f}x",
            "",
            "Sequential Bottlenecks (Critical Path):"
        ]

        for i, bottleneck in enumerate(analysis['bottlenecks'], 1):
            report.append(
                f"{i}. {bottleneck['type'].replace('_', ' ').title()}: {bottleneck['name']} (Line {bottleneck['lineno']})")
            report.append(f"   Work: {bottleneck['work']:.2f}, Span: {bottleneck['span']:.2f}")
            report.append(f"   Source: {bottleneck['source'][:50]}..." if len(
                bottleneck['source']) > 50 else f"   Source: {bottleneck['source']}")
            report.append("")

        return "\n".join(report)

    def suggest_patterns_for_bottleneck(self, bottleneck):
        """
        Analyze a bottleneck on the critical path and suggest appropriate parallel patterns.

        This method examines the code structure of a bottleneck identified in the critical path
        analysis and suggests suitable parallel patterns based on the code's characteristics.
        It uses a comprehensive set of pattern detection helpers to identify computational
        structures that match known parallel patterns.

        Args:
            bottleneck: Dictionary containing bottleneck information including:
                - type: The type of code block (function, loop, etc.)
                - source: The source code of the bottleneck
                - lineno: The line number where the bottleneck appears
                - span: The critical path length of the bottleneck
                - work: The total work of the bottleneck

        Returns:
            List of suggested patterns with confidence scores, rationales, and partitioning strategies
        """
        # Extract bottleneck characteristics
        node_type = bottleneck.get('type', '')
        source_code = bottleneck.get('source', '')

        # Parse the bottleneck code to analyze its structure
        try:
            bottleneck_node = ast.parse(source_code)
        except SyntaxError:
            # If parsing fails, return a generic suggestion
            return [{
                "pattern": "unknown",
                "confidence": 0.3,
                "rationale": "Unable to parse code structure for detailed analysis.",
                "partitioning": ["SDP"]
            }]

        # Analyze computational structure using all available helpers
        has_nested_loops = self._has_nested_loops(bottleneck_node)
        has_neighbor_access = self._has_neighbor_access(bottleneck_node)
        has_reduction_pattern = self._has_reduction_pattern(bottleneck_node)
        has_producer_consumer_pattern = self._has_producer_consumer_pattern(bottleneck_node)
        has_independent_tasks = self._has_independent_tasks(bottleneck_node)
        has_divide_combine_pattern = self._has_divide_combine_pattern(bottleneck_node)
        has_distribution_collection_pattern = self._has_distribution_collection_pattern(bottleneck_node)
        has_task_distribution = self._has_task_distribution(bottleneck_node)
        has_accumulation_pattern = self._has_accumulation_pattern(bottleneck_node)
        is_independent_loop = self._is_independent_loop(bottleneck_node)

        # Match with patterns from the Pattern Characteristic Matrix
        suggested_patterns = []

        # Map pattern - checks for independent loops and operations
        if is_independent_loop or (node_type == 'for_loop' and not has_nested_loops and not has_reduction_pattern):
            suggested_patterns.append({
                "pattern": "map",
                "confidence": 0.9 if is_independent_loop else 0.7,
                "rationale": "The bottleneck contains independent operations that can be executed in parallel.",
                "partitioning": ["SDP", "SIP", "horizontal"],
                "description": "Apply the same operation independently to each element in a dataset.",
                "speedup_potential": "Linear (O(p)) with sufficient data parallelism."
            })

        # Stencil pattern - checks for nested loops with neighbor access
        if has_nested_loops and has_neighbor_access:
            suggested_patterns.append({
                "pattern": "stencil",
                "confidence": 0.85,
                "rationale": "The bottleneck contains nested loops with neighbor access patterns.",
                "partitioning": ["SDP", "horizontal"],
                "description": "Update array elements based on neighboring elements.",
                "speedup_potential": "O(n) for 2D problems with n² elements."
            })

        # Pipeline pattern - checks for producer-consumer relationships
        if has_producer_consumer_pattern:
            suggested_patterns.append({
                "pattern": "pipeline",
                "confidence": 0.8,
                "rationale": "The bottleneck shows a producer-consumer pattern with data flowing between stages.",
                "partitioning": ["TDP", "TIP"],
                "description": "Divide a task into a series of stages, with data flowing through stages.",
                "speedup_potential": "Limited by the slowest stage, up to O(s) for s stages."
            })

        # Reduction pattern - checks for accumulation operations
        if has_reduction_pattern or has_accumulation_pattern:
            suggested_patterns.append({
                "pattern": "reduction",
                "confidence": 0.85,
                "rationale": "The bottleneck accumulates results, suitable for parallel reduction.",
                "partitioning": ["SDP", "SIP"],
                "description": "Combine multiple elements into a single result using an associative operation.",
                "speedup_potential": "O(log n) critical path with O(n/log n) parallelism."
            })

        # Divide and Conquer pattern - checks for recursive division and combination
        if has_divide_combine_pattern:
            suggested_patterns.append({
                "pattern": "divide_conquer",
                "confidence": 0.85,
                "rationale": "The bottleneck recursively divides work and combines results.",
                "partitioning": ["SIP", "horizontal"],
                "description": "Recursively break down a problem into smaller subproblems.",
                "speedup_potential": "O(n) for many problems with O(n log n) work."
            })

        # Fork-Join pattern - checks for independent task creation
        if has_independent_tasks:
            suggested_patterns.append({
                "pattern": "fork_join",
                "confidence": 0.75,
                "rationale": "The bottleneck creates independent tasks that can be executed in parallel.",
                "partitioning": ["SIP", "horizontal"],
                "description": "Split a task into subtasks, execute them in parallel, then join results.",
                "speedup_potential": "Limited by the critical path length."
            })

        # Master-Worker pattern - checks for task distribution
        if has_task_distribution:
            suggested_patterns.append({
                "pattern": "master_worker",
                "confidence": 0.7,
                "rationale": "The bottleneck distributes independent tasks to workers.",
                "partitioning": ["SIP", "horizontal", "hash"],
                "description": "A master process distributes tasks to worker processes.",
                "speedup_potential": "Near-linear with good load balancing."
            })

        # Scatter-Gather pattern - checks for distribution followed by collection
        if has_distribution_collection_pattern:
            suggested_patterns.append({
                "pattern": "scatter_gather",
                "confidence": 0.7,
                "rationale": "The bottleneck distributes data for parallel processing and then collects results.",
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
                    "rationale": "The bottleneck contains nested loops that might benefit from pipelining.",
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

        # Add Amdahl's Law analysis for the top pattern
        if suggested_patterns:
            top_pattern = suggested_patterns[0]
            span_ratio = bottleneck.get('span', 0) / bottleneck.get('work', 1)
            top_pattern["amdahl_analysis"] = {
                "sequential_fraction": span_ratio,
                "max_theoretical_speedup": 1 / span_ratio if span_ratio > 0 else float('inf'),
                "recommendation": "Focus on this pattern to reduce the critical path length."
            }

        return suggested_patterns


    def _has_nested_loops(self, node):
        """
        Check if a node contains nested loops.

        This method identifies if the AST node contains loops nested within other loops,
        which is a key characteristic of patterns like stencil computations.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if nested loops are present
        """
        outer_loops = []
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.For, ast.While)):
                outer_loops.append(subnode)

        for loop in outer_loops:
            for subnode in ast.walk(loop):
                if isinstance(subnode, (ast.For, ast.While)) and subnode != loop:
                    return True
        return False

    def _has_neighbor_access(self, node):
        """
        Check if a node accesses neighboring array elements.

        This method identifies array access patterns where elements adjacent to
        the current index are accessed, which is characteristic of stencil patterns.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if neighboring element access is present
        """
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Subscript):
                # Look for patterns like a[i+1], a[i-1], etc.
                if self._is_neighbor_index(subnode):
                    return True
        return False

    def _is_neighbor_index(self, subscript_node):
        """
        Check if a subscript node accesses neighboring elements.

        This helper method examines array subscript expressions to determine
        if they access elements at offset +1 or -1 from an index variable.

        Args:
            subscript_node: The AST subscript node to analyze

        Returns:
            Boolean indicating if the subscript accesses a neighboring element
        """
        if isinstance(subscript_node.slice, ast.BinOp):
            # Check for i+1, i-1 patterns
            if isinstance(subscript_node.slice, ast.BinOp) and \
                    isinstance(subscript_node.slice.op, (ast.Add, ast.Sub)) and \
                    isinstance(subscript_node.slice.right, ast.Constant) and \
                    subscript_node.slice.right.value == 1:
                return True
        return False

    def _has_reduction_pattern(self, node):
        """
        Check if a node contains a reduction pattern (accumulation).

        This method identifies patterns where values are accumulated into a variable,
        which is characteristic of reduction patterns.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if a reduction pattern is present
        """
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

    def _has_producer_consumer_pattern(self, node):
        """
        Check if a node exhibits a producer-consumer pattern.

        This method identifies patterns where data is produced in one section
        and consumed in another, which is characteristic of pipeline patterns.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if a producer-consumer pattern is present
        """
        # Look for patterns where one loop produces data and another consumes it
        # This is often seen in for-in-while or while-with-for patterns
        if hasattr(node, 'type') and (
                'for_in_while' in node.get('type', '') or 'while_with_for' in node.get('type', '')):
            return True

        # Look for queue or buffer operations
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if hasattr(subnode.func, 'attr') and subnode.func.attr in ['put', 'get', 'append', 'pop']:
                    return True
                elif hasattr(subnode.func, 'id') and subnode.func.id in ['put', 'get', 'append', 'pop']:
                    return True

        return False

    def _has_independent_tasks(self, node):
        """
        Check if a node creates independent tasks.

        This method identifies patterns where independent computations are created,
        which is characteristic of fork-join patterns.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if independent tasks are created
        """
        # Check if this is a recursive function with independent recursive calls
        if isinstance(node, ast.FunctionDef):
            recursive_calls = []
            function_name = node.name

            # Find recursive calls
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call) and hasattr(subnode.func, 'id') and subnode.func.id == function_name:
                    recursive_calls.append(subnode)

            # If there are multiple recursive calls, check if they're independent
            if len(recursive_calls) > 1:
                # Simple heuristic: if recursive calls are in different branches, they're likely independent
                return True

        # Check for thread/process creation or parallel constructs
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if hasattr(subnode.func, 'attr') and subnode.func.attr in ['Thread', 'Process', 'Pool', 'map']:
                    return True
                elif hasattr(subnode.func, 'id') and subnode.func.id in ['Thread', 'Process', 'Pool', 'map']:
                    return True

        return False

    def _has_divide_combine_pattern(self, node):
        """
        Check if a node has a divide-and-conquer pattern.

        This method identifies patterns where a problem is recursively divided
        and results are combined, which is characteristic of divide-and-conquer patterns.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if a divide-and-conquer pattern is present
        """
        if not isinstance(node, ast.FunctionDef):
            return False

        # Check for recursive function with a divide-and-combine structure
        function_name = node.name
        has_recursive_call = False
        has_divide = False
        has_combine = False

        # Look for recursive calls
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call) and hasattr(subnode.func, 'id') and subnode.func.id == function_name:
                has_recursive_call = True

                # Check if the recursive call uses a divided input (e.g., array slicing)
                for arg in subnode.args:
                    if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                        has_divide = True

        # Look for result combination (e.g., binary operations between recursive calls)
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.BinOp) and isinstance(subnode.left, ast.Call) and isinstance(subnode.right,
                                                                                                    ast.Call):
                if (hasattr(subnode.left.func, 'id') and subnode.left.func.id == function_name and
                        hasattr(subnode.right.func, 'id') and subnode.right.func.id == function_name):
                    has_combine = True

        return has_recursive_call and (has_divide or has_combine)

    def _has_distribution_collection_pattern(self, node):
        """
        Check if a node has a distribution-collection pattern.

        This method identifies patterns where data is distributed across multiple
        processors and then collected, which is characteristic of scatter-gather patterns.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if a distribution-collection pattern is present
        """
        has_distribution = False
        has_collection = False

        # Look for distribution patterns (scatter, broadcast, etc.)
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if hasattr(subnode.func, 'attr') and subnode.func.attr in ['scatter', 'broadcast', 'map']:
                    has_distribution = True
                elif hasattr(subnode.func, 'id') and subnode.func.id in ['scatter', 'broadcast', 'map']:
                    has_distribution = True

        # Look for collection patterns (gather, reduce, etc.)
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if hasattr(subnode.func, 'attr') and subnode.func.attr in ['gather', 'reduce', 'collect', 'join']:
                    has_collection = True
                elif hasattr(subnode.func, 'id') and subnode.func.id in ['gather', 'reduce', 'collect', 'join']:
                    has_collection = True

        return has_distribution and has_collection

    def _has_task_distribution(self, node):
        """
        Check if a function distributes tasks.

        This method identifies patterns where tasks are distributed to workers,
        which is characteristic of master-worker patterns.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if task distribution is present
        """
        # Look for task queue operations or worker pool usage
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if hasattr(subnode.func, 'attr') and subnode.func.attr in ['map', 'apply', 'submit', 'starmap']:
                    return True
                elif hasattr(subnode.func, 'id') and subnode.func.id in ['map', 'apply', 'submit', 'starmap']:
                    return True

        # Look for loop that assigns work to different workers
        has_loop = False
        has_worker_assignment = False

        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.For, ast.While)):
                has_loop = True

                # Check if inside the loop there's worker assignment
                for inner in ast.walk(subnode):
                    if isinstance(inner, ast.Call):
                        if hasattr(inner.func, 'attr') and inner.func.attr in ['put', 'send', 'assign']:
                            has_worker_assignment = True
                        elif hasattr(inner.func, 'id') and inner.func.id in ['put', 'send', 'assign']:
                            has_worker_assignment = True

        return has_loop and has_worker_assignment

    def _has_accumulation_pattern(self, node):
        """
        Check if a loop has an accumulation pattern.

        This is a more specific version of reduction pattern detection,
        focusing on loops that accumulate results.

        Args:
            node: The AST node to analyze

        Returns:
            Boolean indicating if an accumulation pattern is present
        """
        # Check if this is a loop
        if not isinstance(node, (ast.For, ast.While)):
            return False

        # Look for accumulation variables that are updated in each iteration
        accumulation_vars = set()

        for subnode in ast.walk(node):
            if isinstance(subnode, ast.AugAssign):  # +=, *=, etc.
                if isinstance(subnode.target, ast.Name):
                    accumulation_vars.add(subnode.target.id)
            elif isinstance(subnode, ast.Assign):
                if isinstance(subnode.targets[0], ast.Name) and isinstance(subnode.value, ast.BinOp):
                    if isinstance(subnode.value.left, ast.Name) and subnode.targets[0].id == subnode.value.left.id:
                        accumulation_vars.add(subnode.targets[0].id)

        return len(accumulation_vars) > 0

    def _is_independent_loop(self, loop):
        """
        Check if a loop has independent iterations.

        This method analyzes a loop to determine if its iterations can be
        executed independently (no loop-carried dependencies).

        Args:
            loop: The loop node to analyze

        Returns:
            Boolean indicating if the loop has independent iterations
        """
        if not isinstance(loop, (ast.For, ast.While)):
            return False

        # Get loop variable (for 'for' loops)
        loop_var = None
        if isinstance(loop, ast.For) and isinstance(loop.target, ast.Name):
            loop_var = loop.target.id

        # Check for loop-carried dependencies
        for subnode in ast.walk(loop):
            # Check for assignments to array elements using the loop variable
            if isinstance(subnode, ast.Assign) and isinstance(subnode.targets[0], ast.Subscript):
                if self._uses_variable_in_subscript(subnode.targets[0], loop_var):
                    # Check if the same array is read with a different index
                    if self._reads_array_with_offset(loop, subnode.targets[0].value.id, loop_var):
                        return False

            # Check for augmented assignments with dependencies
            if isinstance(subnode, ast.AugAssign) and isinstance(subnode.target, ast.Name):
                # If we're updating a variable that's not the loop variable itself
                # and it's used elsewhere in the loop, this might indicate a dependency
                if subnode.target.id != loop_var and self._variable_used_elsewhere(loop, subnode.target.id, subnode):
                    return False

        return True

    def _uses_variable_in_subscript(self, subscript_node, var_name):
        """
        Check if a subscript uses a specific variable in its index.

        Args:
            subscript_node: The subscript node to check
            var_name: The variable name to look for

        Returns:
            Boolean indicating if the variable is used in the subscript
        """
        if not var_name:
            return False

        for subnode in ast.walk(subscript_node):
            if isinstance(subnode, ast.Name) and subnode.id == var_name:
                return True

        return False

    def _reads_array_with_offset(self, loop_node, array_name, loop_var):
        """
        Check if an array is read with an offset from the loop variable.

        This helps detect loop-carried dependencies where an iteration
        depends on values computed in previous iterations.

        Args:
            loop_node: The loop node to check
            array_name: The name of the array to check
            loop_var: The loop variable name

        Returns:
            Boolean indicating if the array is read with an offset
        """
        if not loop_var:
            return False

        for subnode in ast.walk(loop_node):
            if isinstance(subnode, ast.Subscript) and isinstance(subnode.value,
                                                                 ast.Name) and subnode.value.id == array_name:
                # Check if this is a read operation (not on the left side of an assignment)
                if not self._is_assignment_target(loop_node, subnode):
                    # Check if the subscript uses the loop variable with an offset
                    if isinstance(subnode.slice, ast.BinOp):
                        if (isinstance(subnode.slice.left, ast.Name) and subnode.slice.left.id == loop_var and
                                isinstance(subnode.slice.op, (ast.Add, ast.Sub))):
                            return True

        return False

    def _is_assignment_target(self, root_node, node_to_check):
        """
        Check if a node is the target of an assignment.

        Args:
            root_node: The root node to search in
            node_to_check: The node to check if it's an assignment target

        Returns:
            Boolean indicating if the node is an assignment target
        """
        for subnode in ast.walk(root_node):
            if isinstance(subnode, ast.Assign) and node_to_check in subnode.targets:
                return True
            if isinstance(subnode, ast.AugAssign) and node_to_check == subnode.target:
                return True

        return False

    def _variable_used_elsewhere(self, root_node, var_name, exclude_node):
        """
        Check if a variable is used elsewhere in a node.

        Args:
            root_node: The root node to search in
            var_name: The variable name to look for
            exclude_node: A node to exclude from the search

        Returns:
            Boolean indicating if the variable is used elsewhere
        """
        for subnode in ast.walk(root_node):
            if subnode != exclude_node:
                for inner in ast.walk(subnode):
                    if isinstance(inner, ast.Name) and inner.id == var_name:
                        return True

        return False

