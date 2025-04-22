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

Classes:
- BDiasPatternTransformer: Base class for pattern-specific AST transformers
- MapPatternTransformer: Transforms Map patterns into parallel implementations
- (Additional pattern transformers to be added)

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


class MapPatternTransformer(BDiasPatternTransformer):
    """Transforms Map patterns into parallel implementations."""

    def visit_For(self, node: ast.For) -> ast.AST:
        """
        Transform independent loops into parallel map operations.

        Args:
            node: The for loop node to transform

        Returns:
            Transformed AST node
        """
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
            'body': body,
            'partitioning_strategy': self.partitioning_strategy
        })

        # Add necessary imports
        self.add_import('multiprocessing', 'Pool')

        # Return the original node (transformation happens in template)
        return node


    def visit_FunctionDef(self, node):
        """Transform function definitions into parallel implementations."""
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
            'func_body': func_body,  # Store as func_body
            'body': func_body,  # Also store as body for compatibility
            'partitioning_strategy': self.partitioning_strategy
        })

        # Add necessary imports
        self.add_import('multiprocessing', 'Pool')

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

        # For other node types, use exact line number matching
        return node.lineno == bottleneck_lineno

    def generate_code(self) -> str:
        """
        Generate parallel code for the Map pattern.

        Returns:
            Generated code as a string
        """
        # Add precomputed values to the context - draft hdw specific values
        import multiprocessing
        processor_count = multiprocessing.cpu_count()
        data_size = 10  # Default value

        # For loops, try to determine the size of the iterable
        if 'iter_expr' in self.context:
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

        # Check if this is a class method (first argument is 'self')
        is_class_method = False
        if is_function and 'func_args' in self.context and len(self.context['func_args']) > 0:
            is_class_method = self.context['func_args'][0] == 'self'

            # If it's a class method, modify the context to handle it properly
            if is_class_method:
                # Use the second argument as data if available, otherwise use a default
                if len(self.context['func_args']) > 1:
                    self.context['data_arg'] = self.context['func_args'][1]
                else:
                    self.context['data_arg'] = 'data'

                # Add a flag to indicate this is a class method
                self.context['is_class_method'] = True

        # Choose template based on node type and partitioning strategy
        if is_function:
            # Function templates
            if "SDP" in self.partitioning_strategy:
                template_name = "map/function_sdp_multiprocessing.j2"
            elif "SIP" in self.partitioning_strategy:
                template_name = "map/function_sip_multiprocessing.j2"
            else:
                template_name = "map/function_default_multiprocessing.j2"
        else:
            # Loop templates
            if "SDP" in self.partitioning_strategy:
                template_name = "map/sdp_multiprocessing.j2"
            elif "SIP" in self.partitioning_strategy:
                template_name = "map/sip_multiprocessing.j2"
            else:
                template_name = "map/default_multiprocessing.j2"

        # Load and render the template
        try:
            template = self.env.get_template(template_name)
            return template.render(**self.context)
        except jinja2.exceptions.TemplateNotFound:
            # Fallback to default template if specific template not found
            fallback_template_name = "map/default_multiprocessing.j2" if not is_function else "map/function_default_multiprocessing.j2"
            template = self.env.get_template(fallback_template_name)
            return template.render(**self.context)


def generate_parallel_code(bottleneck: Dict[str, Any], pattern: str, partitioning_strategy: List[str]) -> \
        Tuple[str, str, Dict[str, Any]]:
    """
    Generate parallelized code for a bottleneck based on identified pattern.

    Args:
        bottleneck: Dictionary containing bottleneck information
        pattern: Identified parallel pattern (e.g., 'map', 'stencil')
        partitioning_strategy: Recommended partitioning strategy

    Returns:
        Tuple of (original_code, transformed_code)
    """
    # Parse the bottleneck source code
    source_code = bottleneck['source']
    tree = ast.parse(source_code)

    # Create appropriate transformer based on pattern
    if pattern == 'map':
        transformer = MapPatternTransformer(bottleneck, partitioning_strategy)
    # Add more patterns as needed
    else:
        # Default to Map pattern for now
        transformer = MapPatternTransformer(bottleneck, partitioning_strategy)

    # Apply the transformation
    transformer.visit(tree)
    transformer.finalize(tree)

    # Generate code using the template
    transformed_code = transformer.generate_code()

    return source_code, transformed_code, transformer.context
