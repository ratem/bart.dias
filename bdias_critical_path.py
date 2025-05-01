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

    '''
    def visualize_dag(self, output_file: str = None, mode: str = "2d",
                      critical_path_nodes: Optional[set] = None,
                      critical_path_edges: Optional[set] = None) -> None:
        """
        Visualize the DAG, coloring critical path nodes and edges in red, others in blue/gray.
        Node labels are only the line numbers (or range) of the code block.
        Parameters:
            output_file (str, optional): Path to save the 2D plot (if mode="2d").
            mode (str): "2d" (default, static plot) or "3d" (interactive, requires matplotlib 3D).
            critical_path_nodes (set, optional): Set of node IDs to highlight as critical path.
            critical_path_edges (set, optional): Set of edge tuples (u, v) on the critical path.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = self.dag

        # Compute critical path nodes and edges if not provided
        if critical_path_nodes is None or critical_path_edges is None:
            try:
                critical_path, _ = self._find_critical_path()
                critical_path_nodes = set(critical_path)
                critical_path_edges = set(zip(critical_path, critical_path[1:]))
            except Exception:
                critical_path_nodes = set()
                critical_path_edges = set()

        # Prepare node colors and labels
        node_colors = []
        node_labels = {}
        for n, data in G.nodes(data=True):
            # Label is the line number(s)
            if "lineno" in data and "end_lineno" in data:
                if data["lineno"] == data["end_lineno"]:
                    label = f"{data['lineno']}"
                else:
                    label = f"{data['lineno']}-{data['end_lineno']}"
            elif "lineno" in data:
                label = f"{data['lineno']}"
            else:
                label = str(n)
            node_labels[n] = label
            # Color: red for critical path, blue otherwise
            node_colors.append("red" if n in critical_path_nodes else "blue")

        # Prepare edge colors
        edge_colors = []
        for u, v in G.edges():
            if (u, v) in critical_path_edges:
                edge_colors.append("red")
            else:
                edge_colors.append("gray")

        if mode == "3d":
            try:
                from mpl_toolkits.mplot3d import Axes3D
                import numpy as np
                pos_2d = nx.spring_layout(G, seed=42, dim=2)
                # Assign a z coordinate based on topological order
                topo_order = list(nx.topological_sort(G))
                z_coords = {n: i for i, n in enumerate(topo_order)}
                pos_3d = {n: (x, y, z_coords[n]) for n, (x, y) in pos_2d.items()}

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Draw edges
                for (u, v), color in zip(G.edges(), edge_colors):
                    x = [pos_3d[u][0], pos_3d[v][0]]
                    y = [pos_3d[u][1], pos_3d[v][1]]
                    z = [pos_3d[u][2], pos_3d[v][2]]
                    ax.plot(x, y, z, color=color, alpha=0.7, linewidth=2 if color == "red" else 1)

                # Draw nodes
                xs = [pos_3d[n][0] for n in G.nodes()]
                ys = [pos_3d[n][1] for n in G.nodes()]
                zs = [pos_3d[n][2] for n in G.nodes()]
                ax.scatter(xs, ys, zs, c=node_colors, s=200, depthshade=True)

                # Add labels
                for n in G.nodes():
                    x, y, z = pos_3d[n]
                    ax.text(x, y, z, node_labels[n], fontsize=10, ha='center', va='center')

                ax.set_title("DAG with Critical Path (3D)")
                plt.show()
            except ImportError:
                print("mpl_toolkits.mplot3d is required for 3D visualization.")
        else:
            # 2D plot
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=[2 if c == "red" else 1 for c in edge_colors],
                                   alpha=0.7)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='white')
            plt.title("DAG with Critical Path (2D)")
            plt.axis('off')
            if output_file:
                plt.savefig(output_file, bbox_inches='tight')
            plt.show()

    '''

    def visualize_dag(
            self,
            output_file: Optional[str] = None,
            mode: str = "2d",
            critical_path_nodes: Optional[set] = None,
            critical_path_edges: Optional[set] = None,
    ) -> None:
        """
        Visualize the DAG, coloring critical path nodes and edges in red, others in blue/gray.
        START/END nodes are always shown and not faded. Node labels are only the line numbers (or range).
        Non-critical-path, non-START/END nodes are faded.
        In 3D, nodes in the same group are spaced out.
        Parameters:
            output_file (str, optional): Path to save the 2D plot (if mode="2d").
            mode (str): "2d" (default, static plot) or "3d" (interactive, requires matplotlib 3D).
            critical_path_nodes (set, optional): Set of node IDs to highlight as critical path.
            critical_path_edges (set, optional): Set of edge tuples (u, v) on the critical path.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = self.dag

        # Compute critical path nodes and edges if not provided
        if critical_path_nodes is None or critical_path_edges is None:
            try:
                critical_path, _ = self._find_critical_path()
                critical_path_nodes = set(critical_path)
                critical_path_edges = set(zip(critical_path, critical_path[1:]))
            except Exception:
                critical_path_nodes = set()
                critical_path_edges = set()

        # Prepare node colors, alphas, and labels
        node_colors = []
        node_alphas = []
        node_labels = {}
        node_groups = {}
        group_counter = 0
        group_map = {}

        for n, data in G.nodes(data=True):
            # Label is the line number(s)
            if n == self.entry_node:
                label = "START"
            elif n == self.exit_node:
                label = "END"
            elif "lineno" in data and "end_lineno" in data:
                if data["lineno"] == data["end_lineno"]:
                    label = f"{data['lineno']}"
                else:
                    label = f"{data['lineno']}-{data['end_lineno']}"
            elif "lineno" in data:
                label = f"{data['lineno']}"
            else:
                label = str(n)
            node_labels[n] = label

            # Grouping for 3D: group by function name if present, else by type
            group_key = data.get("name") or data.get("type") or "other"
            if group_key not in group_map:
                group_map[group_key] = group_counter
                group_counter += 1
            node_groups[n] = group_map[group_key]

            # Color and alpha: critical path = red/1.0, START/END = black/1.0, others blue/0.2
            if n == self.entry_node or n == self.exit_node:
                node_colors.append("black")
                node_alphas.append(1.0)
            elif n in critical_path_nodes:
                node_colors.append("red")
                node_alphas.append(1.0)
            else:
                node_colors.append("blue")
                node_alphas.append(0.2)

        # Prepare edge colors
        edge_colors = []
        for u, v in G.edges():
            if (u, v) in critical_path_edges:
                edge_colors.append("red")
            else:
                edge_colors.append("gray")

        if mode == "3d":
            try:
                from mpl_toolkits.mplot3d import Axes3D

                # Axes: X=node size, Y=start line, Z=group (with jitter for spacing)
                x_vals, y_vals, z_vals = {}, {}, {}
                group_jitter = {}

                for n, data in G.nodes(data=True):
                    # X: node size (lines spanned)
                    lines = (data.get("lineno", 0), data.get("end_lineno", data.get("lineno", 0)))
                    x = max(1, lines[1] - lines[0] + 1)
                    x_vals[n] = x

                    # Y: starting line number
                    y = data.get("lineno", 0)
                    y_vals[n] = y

                    # Z: group with jitter for spacing
                    group = node_groups[n]
                    if group not in group_jitter:
                        group_jitter[group] = 0
                    z = group + 0.2 * group_jitter[group]
                    group_jitter[group] += 1
                    z_vals[n] = z

                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection="3d")

                # Draw edges
                for (u, v), color in zip(G.edges(), edge_colors):
                    ax.plot(
                        [x_vals[u], x_vals[v]],
                        [y_vals[u], y_vals[v]],
                        [z_vals[u], z_vals[v]],
                        color=color,
                        alpha=0.8,
                        linewidth=2 if color == "red" else 1,
                    )

                # Draw nodes with alpha
                xs = [x_vals[n] for n in G.nodes()]
                ys = [y_vals[n] for n in G.nodes()]
                zs = [z_vals[n] for n in G.nodes()]
                for i, n in enumerate(G.nodes()):
                    ax.scatter(
                        xs[i], ys[i], zs[i],
                        c=node_colors[i],
                        s=250,
                        alpha=node_alphas[i],
                        depthshade=True
                    )

                # Add labels
                for n in G.nodes():
                    ax.text(
                        x_vals[n], y_vals[n], z_vals[n], node_labels[n],
                        fontsize=10, ha="center", va="center"
                    )

                ax.set_xlabel("Node Size (lines)")
                ax.set_ylabel("Start Line")
                ax.set_zlabel("Group (function/type)")
                ax.set_title("DAG with Critical Path (3D, semantic axes)")
                plt.show()
            except ImportError:
                print("mpl_toolkits.mplot3d is required for 3D visualization.")
        else:
            # 2D: project onto (line size, start line)
            x_vals = {}
            y_vals = {}
            jitter = {}
            for n, data in G.nodes(data=True):
                lines = (data.get("lineno", 0), data.get("end_lineno", data.get("lineno", 0)))
                x = max(1, lines[1] - lines[0] + 1)
                group = node_groups[n]
                if group not in jitter:
                    jitter[group] = 0
                # Spread nodes in the same group along y
                y = data.get("lineno", 0) + 0.5 * jitter[group]
                jitter[group] += 1
                x_vals[n] = x
                y_vals[n] = y

            pos = {n: (x_vals[n], y_vals[n]) for n in G.nodes()}
            plt.figure(figsize=(12, 8))
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edge_color=edge_colors,
                width=[2 if c == "red" else 1 for c in edge_colors],
                alpha=0.8,
            )
            # Draw nodes with fading
            for i, n in enumerate(G.nodes()):
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=[n],
                    node_color=node_colors[i],
                    node_size=700,
                    alpha=node_alphas[i]
                )
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="white")
            plt.xlabel("Node Size (lines)")
            plt.ylabel("Start Line (with group jitter)")
            plt.title("DAG with Critical Path (2D, semantic axes)")
            plt.axis("on")
            if output_file:
                plt.savefig(output_file, bbox_inches="tight")
            plt.show()

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

