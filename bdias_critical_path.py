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

        # Extract code bl   ocks
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

