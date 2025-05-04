"""
BDiasPatternCodegen: Pattern-Based Code Generation Module for Bart.dIAs

This module implements pattern-specific code generators that transform identified
parallel patterns into optimized parallel implementations. It uses AST transformation
combined with Jinja2 templates to generate readable, efficient parallel code.

Features:
- Pattern-specific AST transformers for each parallel pattern
- Integration with the pattern analyzer to get pattern characteristics
- Template-based code generation for readability and maintainability
- Support for different partitioning strategies
- Hardware-aware code generation that adapts to available system resources

Currently implemented patterns:
- Map-Reduce: Transforms independent operations followed by associative reduction
  into parallel implementations using Python's multiprocessing module

Each pattern transformer:
1. Analyzes the AST to extract key components of the pattern
2. Selects appropriate templates based on the pattern and partitioning strategy
3. Generates parallelized code that adapts to the available hardware
4. Provides hardware-specific recommendations for optimal performance

Classes:
- BDiasPatternTransformer: Base class for pattern-specific AST transformers
- MapReducePatternTransformer: Transforms Map-Reduce patterns into parallel implementations

Dependencies:
- ast: For AST manipulation
- jinja2: For template-based code generation
"""

import ast
import jinja2
from typing import Dict, List, Any, Optional, Tuple


class BDiasPatternTransformer(ast.NodeTransformer):
    """Base class for pattern-specific AST transformers."""

    def __init__(self, bottleneck: Dict[str, Any], partitioning_strategy: List[str]):
        """
        Initialize the pattern transformer.

        Args:
            bottleneck: Dictionary containing bottleneck information
            partitioning_strategy: List of recommended partitioning strategies
        """
        self.bottleneck = bottleneck
        self.partitioning_strategy = partitioning_strategy
        self.imports_to_add = []
        self.context = {}  # For template rendering

        # Initialize Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def add_import(self, module: str, name: Optional[str] = None, alias: Optional[str] = None):
        """
        Add an import statement to be inserted at the top of the module.

        Args:
            module: Module to import
            name: Specific name to import from module (for from ... import ...)
            alias: Alias for the import
        """
        self.imports_to_add.append((module, name, alias))

    def finalize(self, tree: ast.AST) -> ast.AST:
        """
        Add necessary imports and finalize the transformed AST.

        Args:
            tree: The AST to finalize

        Returns:
            The finalized AST
        """
        # Add imports at the top of the file
        for module, name, alias in self.imports_to_add:
            import_node = self._create_import_node(module, name, alias)
            if isinstance(tree, ast.Module):
                tree.body.insert(0, import_node)

        return tree

    def _create_import_node(self, module: str, name: Optional[str] = None, alias: Optional[str] = None) -> ast.stmt:
        """
        Create an import AST node.

        Args:
            module: Module to import
            name: Specific name to import from module (for from ... import ...)
            alias: Alias for the import

        Returns:
            Import AST node
        """
        if name:
            # from module import name as alias
            return ast.ImportFrom(
                module=module,
                names=[ast.alias(name=name, asname=alias)],
                level=0
            )
        else:
            # import module as alias
            return ast.Import(
                names=[ast.alias(name=module, asname=alias)]
            )

    def get_template_context(self) -> Dict[str, Any]:
        """
        Return the context for template rendering.

        Returns:
            Dictionary containing context for template rendering
        """
        return self.context

    def generate_code(self) -> str:
        """
        Generate code using the template and context.

        Returns:
            Generated code as a string
        """
        raise NotImplementedError("Subclasses must implement generate_code()")


class MapReducePatternTransformer(BDiasPatternTransformer):
    """
    Transforms Map-Reduce patterns into parallel implementations.

    This class analyzes code structures that exhibit the Map-Reduce pattern
    and transforms them into parallel implementations using Python's
    multiprocessing module. It handles both function definitions and for loops,
    and supports different partitioning strategies.

    The Map-Reduce pattern involves:
    1. Map phase: Applying the same operation independently to each element in a dataset
    2. Reduce phase: Combining the results using an associative operation

    The transformation process:
    1. Identifies the map and reduce components in the original code
    2. Creates parallel implementations for both phases
    3. Adapts the implementation to the available hardware
    4. Generates code with appropriate error handling and synchronization

    Supported partitioning strategies:
    - SDP (Spatial Domain Partitioning): Divides data into chunks processed by separate workers
    - SIP (Spatial Instruction Partitioning): Applies the same operation to different data elements

    The generated code automatically adapts to the number of available processors
    and includes hardware-specific optimizations.
    """


    def visit_FunctionDef(self, node):
        """Transform function definitions into parallel map-reduce implementations."""
        # Skip if this isn't the target bottleneck
        if not self._is_target_node(node):
            return self.generic_visit(node)

        # Extract function components
        func_name = node.name
        func_args = [arg.arg for arg in node.args.args]
        func_body = ast.unparse(node.body)

        # Store context for template rendering
        self.context.update({
            'func_name': func_name,
            'func_args': func_args,
            'func_body': func_body,
            'body': func_body,  # For compatibility
            'partitioning_strategy': self.partitioning_strategy,
            'is_class_method': func_args and func_args[0] == 'self'
        })

        # Add necessary imports
        self.add_import('multiprocessing', 'Pool')
        self.add_import('functools', 'reduce')

        # Return the original node (transformation happens in template)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        """Transform for loops into parallel map-reduce implementations."""
        # Skip if this isn't the target bottleneck
        if not self._is_target_node(node):
            return self.generic_visit(node)

        # Extract loop components
        loop_var = ast.unparse(node.target)
        iter_expr = ast.unparse(node.iter)
        body = ast.unparse(node.body)

        # Store context for template rendering
        self.context.update({
            'loop_var': loop_var,
            'iter_expr': iter_expr,
            'body': body,  # This is the key line - storing the loop body
            'partitioning_strategy': self.partitioning_strategy
        })

        # Add necessary imports
        self.add_import('multiprocessing', 'Pool')
        self.add_import('functools', 'reduce')

        # Return the original node (transformation happens in template)
        return node

    def _is_target_node(self, node: ast.AST) -> bool:
        """
        Check if this node is the target bottleneck.

        Args:
            node: AST node to check

        Returns:
            True if this is the target bottleneck, False otherwise
        """
        # First check if the node has a line number
        if not hasattr(node, 'lineno'):
            return False

        # Get the bottleneck line number
        bottleneck_lineno = self.bottleneck.get('lineno', -1)

        # For function definitions, also check the function name
        if isinstance(node, ast.FunctionDef):
            # If the bottleneck is a function, check both line number and name
            if self.bottleneck.get('type') == 'function':
                # Extract function name from source if available
                source = self.bottleneck.get('source', '')
                if source.startswith('def '):
                    func_name = source.split('(')[0].replace('def ', '').strip()
                    if node.name == func_name:
                        return True

            # Allow for small differences in line numbers for functions
            return abs(node.lineno - bottleneck_lineno) <= 2

        # For loop nodes, allow for small differences in line numbers
        if isinstance(node, ast.For):
            # If the bottleneck is a for loop, check both line number and iteration expression
            if self.bottleneck.get('type') == 'for_loop':
                # Extract iteration expression from source if available
                source = self.bottleneck.get('source', '')
                if source.startswith('for '):
                    # Try to match the iteration expression
                    iter_expr = ast.unparse(node.iter)
                    if iter_expr in source:
                        return True

            # Allow for small differences in line numbers for loops
            return abs(node.lineno - bottleneck_lineno) <= 2

        # For other node types, use exact line number matching with a small tolerance
        return abs(node.lineno - bottleneck_lineno) <= 1

    def generate_code(self) -> str:
        """
        Generate parallel code for the Map-Reduce pattern.

        Returns:
            Generated code as a string
        """
        # Add precomputed values to the context - hardware specific values
        import multiprocessing
        processor_count = multiprocessing.cpu_count()
        data_size = 10  # Default value

        # Estimate data size from context
        if 'func_body' in self.context:
            func_body = self.context['func_body']
            # Look for array/list declarations with size information
            import re
            size_patterns = [
                r'shape=\((\d+),\)',  # numpy array shape
                r'range\((\d+)\)',  # range function
                r'len\((\w+)\)',  # length of something
            ]

            for pattern in size_patterns:
                matches = re.findall(pattern, func_body)
                if matches:
                    try:
                        data_size = max(data_size, int(matches[0]))
                    except (ValueError, IndexError):
                        pass
        elif 'iter_expr' in self.context:
            iter_expr = self.context['iter_expr']
            if 'range' in iter_expr:
                # Try to extract the range parameters
                try:
                    range_params = iter_expr.replace('range(', '').replace(')', '').split(',')
                    if len(range_params) == 1:
                        data_size = int(range_params[0])
                    elif len(range_params) == 2:
                        data_size = int(range_params[1]) - int(range_params[0])
                    elif len(range_params) == 3:
                        data_size = (int(range_params[1]) - int(range_params[0])) // int(range_params[2])
                except (ValueError, IndexError):
                    # Keep default if parsing fails
                    pass
            elif 'len(' in iter_expr:
                # This is a common pattern in the for loop at line 197
                # For example: for t in range(1, len(self.accelerations_y)):
                data_size = 100  # Use a reasonable default for array lengths

        elements_per_processor = max(1, data_size // processor_count)

        # Precompute processor ranges
        processor_ranges = []
        for p in range(processor_count):
            start = p * elements_per_processor
            end = min((p + 1) * elements_per_processor, data_size)
            processor_ranges.append((p, start, end))

        # Add to context
        self.context['processor_ranges'] = processor_ranges
        self.context['processor_count'] = processor_count
        self.context['data_size'] = data_size
        self.context['elements_per_processor'] = elements_per_processor

        # Determine if we're dealing with a function or a loop
        is_function = 'func_name' in self.context

        # Choose template based on node type and partitioning strategy
        if is_function:
            # Function templates
            if "SDP" in self.partitioning_strategy:
                template_name = "map_reduce/function_sdp_multiprocessing.j2"
            elif "SIP" in self.partitioning_strategy:
                template_name = "map_reduce/function_sip_multiprocessing.j2"
            else:
                template_name = "map_reduce/function_default_multiprocessing.j2"
        else:
            # Loop templates
            if "SDP" in self.partitioning_strategy:
                template_name = "map_reduce/sdp_multiprocessing.j2"
            elif "SIP" in self.partitioning_strategy:
                template_name = "map_reduce/sip_multiprocessing.j2"
            else:
                template_name = "map_reduce/default_multiprocessing.j2"

        # Load and render the template
        try:
            template = self.env.get_template(template_name)
            return template.render(**self.context)
        except jinja2.exceptions.TemplateNotFound:
            # Fallback to default template if specific template not found
            fallback_template_name = "map_reduce/default_multiprocessing.j2" if not is_function else "map_reduce/function_default_multiprocessing.j2"
            template = self.env.get_template(fallback_template_name)
            return template.render(**self.context)


class PipelinePatternTransformer(BDiasPatternTransformer):
    """
    Transformer for the Pipeline pattern.  It picks out the top-level
    function whose name matches the bottleneck, extracts the append-expressions
    from each stage, and populates context with:
      - func_name, func_args, input_data
      - stage_count, stage_exprs
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Match by function name (bottleneck['name']), not by lineno/details.
        if (self.bottleneck.get('type') != 'function'
            or node.name != self.bottleneck.get('name')):
            return node

        # Extract all for-loops in the function body
        loops = [stmt for stmt in node.body if isinstance(stmt, ast.For)]
        stage_exprs = []
        for loop in loops:
            # find the .append(...) call in this loop
            for stmt in loop.body:
                if (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, 'attr', '') == 'append'):
                    # unparse the single argument to append(...)
                    expr = ast.unparse(stmt.value.args[0]).strip()
                    stage_exprs.append(expr)
                    break

        # function arguments and input data
        func_args = [arg.arg for arg in node.args.args]
        input_data = func_args[0] if func_args else 'data'

        # Populate context for templates
        self.context.update({
            'func_name':   node.name,
            'func_args':   func_args,
            'input_data':  input_data,
            'stage_count': len(stage_exprs),
            'stage_exprs': stage_exprs
        })

        # Common imports for pipeline code
        self.add_import('multiprocessing', None, None)
        self.add_import('queue', 'Queue', None)

        return node

def generate_hardware_recommendations(context: Dict[str, Any]) -> str:
    """
    Generate hardware-specific recommendations based on the context.

    Args:
        context: Dictionary containing context information

    Returns:
        Hardware recommendations as a string
    """
    recommendations = []

    processor_count = context.get('processor_count', 0)
    data_size = context.get('data_size', 0)
    elements_per_processor = context.get('elements_per_processor', 0)
    is_memory_bound = context.get('is_memory_bound', False)

    if processor_count > 0:
        recommendations.append(f"This code will utilize {processor_count} processors.")

    if elements_per_processor > 0:
        recommendations.append(f"Each processor will handle approximately {elements_per_processor} elements.")

    if is_memory_bound:
        recommendations.append(
            "This operation is memory-bound. Consider reducing the data size or increasing available memory.")
    else:
        recommendations.append("This operation is CPU-bound and should scale well with additional processors.")

    if data_size < processor_count:
        recommendations.append(
            "The data size is smaller than the number of processors. Consider using fewer processors to avoid overhead.")

    return "\n".join(recommendations)


def generate_parallel_code(
    bottleneck: Dict[str, Any],
    pattern: str,
    partitioning_strategy: List[str]
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Dispatch to the correct transformer, run the AST pass, finalize imports,
    then render the appropriate Jinja2 template with a fully-populated context.
    """
    source_code = bottleneck['source']
    tree = ast.parse(source_code)

    # 1. Instantiate the right transformer
    if pattern == 'pipeline':
        transformer = PipelinePatternTransformer(bottleneck, partitioning_strategy)
    elif pattern == 'map_reduce':
        transformer = MapReducePatternTransformer(bottleneck, partitioning_strategy)
    else:
        raise NotImplementedError(f"Pattern '{pattern}' not supported for code generation.")

    # 2. Run the AST transformer to build transformer.context
    transformer.visit(tree)
    transformer.finalize(tree)

    # 3. Select the template based on pattern, function status, and strategy
    # Unified logic for all patterns
    is_function = 'func_name' in transformer.context
    strat = partitioning_strategy[0].lower()

    if is_function:
        template_name = f"{pattern}/function_{strat}_multiprocessing.j2"
    else:
        template_name = f"{pattern}/{strat}_multiprocessing.j2"

    try:
        tpl = transformer.env.get_template(template_name)
    except jinja2.exceptions.TemplateNotFound:
        # Fallback to appropriate default template
        if is_function:
            tpl = transformer.env.get_template(f"{pattern}/function_default_multiprocessing.j2")
        else:
            tpl = transformer.env.get_template(f"{pattern}/default_multiprocessing.j2")

    # 4. Render the template with the populated context
    transformed_code = tpl.render(**transformer.context)

    # 5. Optionally add hardware recommendations into context
    # (if your templates or presenter use them)
    if hasattr(transformer, 'context'):
        from multiprocessing import cpu_count
        transformer.context.setdefault('processor_count', cpu_count())

    return source_code, transformed_code, transformer.context

