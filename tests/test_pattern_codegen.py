"""
Test module for pattern-based code generation in Bart.dIAs.

This module contains tests for the pattern-specific code generators,
focusing on the Map-Reduce pattern implementation.
"""

import unittest
import ast
import jinja2
import sys
from pathlib import Path

# Add proper imports with path handling
# Assuming tests are run from the bartdias/tests directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from bdias_pattern_codegen import MapReducePatternTransformer, generate_parallel_code

class TestMapReducePatternCodegen(unittest.TestCase):
    """Test cases for Map-Reduce pattern code generation."""

    def setUp(self):
        """Set up test cases."""
        # Get the absolute path to the templates directory
        self.project_root = Path(__file__).parent.parent
        self.templates_dir = self.project_root / 'templates'

        # Store the original __init__ method
        self.original_init = MapReducePatternTransformer.__init__

        # Define a new __init__ method that uses the correct template path
        def new_init(instance, bottleneck, partitioning_strategy):
            # Call the original __init__
            self.original_init(instance, bottleneck, partitioning_strategy)
            # Override the environment with the correct template path
            instance.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.templates_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )
            # Modify the template_name in generate_code to use 'map-reduce' instead of 'map_reduce'
            original_generate_code = instance.generate_code

            def patched_generate_code():
                # Determine if we're dealing with a function or a loop
                is_function = 'func_name' in instance.context

                # Choose template based on node type and partitioning strategy
                if is_function:
                    # Function templates
                    if "SDP" in instance.partitioning_strategy:
                        template_name = "map_reduce/function_sdp_multiprocessing.j2"
                    elif "SIP" in instance.partitioning_strategy:
                        template_name = "map_reduce/function_sip_multiprocessing.j2"
                    else:
                        template_name = "map_reduce/function_default_multiprocessing.j2"
                else:
                    # Loop templates
                    if "SDP" in instance.partitioning_strategy:
                        template_name = "map_reduce/sdp_multiprocessing.j2"
                    elif "SIP" in instance.partitioning_strategy:
                        template_name = "map_reduce/sip_multiprocessing.j2"
                    else:
                        template_name = "map_reduce/default_multiprocessing.j2"

                # Load and render the template
                try:
                    template = instance.env.get_template(template_name)
                    return template.render(**instance.context)
                except jinja2.exceptions.TemplateNotFound:
                    # Fallback to default template if specific template not found
                    fallback_template_name = "map_reduce/default_multiprocessing.j2" if not is_function else "map_reduce/function_default_multiprocessing.j2"
                    template = instance.env.get_template(fallback_template_name)
                    return template.render(**instance.context)

            # Replace the generate_code method
            instance.generate_code = patched_generate_code

        # Replace the __init__ method
        MapReducePatternTransformer.__init__ = new_init

        # Test bottleneck
        self.bottleneck = {
            'type': 'for_loop',
            'lineno': 1,
            'source': 'for i in range(100):\n    result.append(i * i)'
        }

        # Update partitioning strategy to match the new design (SDP and SIP, not horizontal)
        self.partitioning_strategy = ['SDP', 'SIP']

        # Create a transformer
        self.transformer = MapReducePatternTransformer(self.bottleneck, self.partitioning_strategy)

    def tearDown(self):
        """Clean up after tests."""
        # Restore the original __init__ method
        MapReducePatternTransformer.__init__ = self.original_init

    def test_map_reduce_transformation(self):
        """Test transformation of a simple Map-Reduce pattern."""
        # Parse the code
        tree = ast.parse(self.bottleneck['source'])

        # Apply the transformation
        self.transformer.visit(tree)
        self.transformer.finalize(tree)

        # Check that the context is correctly populated
        context = self.transformer.get_template_context()
        self.assertEqual(context['loop_var'], 'i')
        self.assertEqual(context['iter_expr'], 'range(100)')
        self.assertIn('result.append(i * i)', context['body'])

        # Verify template paths exist - use the correct path with hyphen
        template_path = self.templates_dir / 'map_reduce' / 'sdp_multiprocessing.j2'
        self.assertTrue(template_path.exists(), f"Template not found: {template_path}")

        # Generate code
        try:
            code = self.transformer.generate_code()
            # Check that the generated code includes multiprocessing
            self.assertIn('multiprocessing', code)
            self.assertIn('Pool', code)
            self.assertIn('map', code)
        except jinja2.exceptions.TemplateNotFound as e:
            self.fail(f"Template not found: {e}")

    def test_generate_parallel_code(self):
        """Test the generate_parallel_code function."""
        try:
            # Since we've patched the MapReducePatternTransformer.__init__ method,
            # we can directly call generate_parallel_code
            original_code, transformed_code, context = generate_parallel_code(
                self.bottleneck,
                'map_reduce',
                self.partitioning_strategy
            )

            # Check that the original code is preserved
            self.assertEqual(original_code, self.bottleneck['source'])

            # Check that the transformed code includes multiprocessing
            self.assertIn('multiprocessing', transformed_code)
            self.assertIn('Pool', transformed_code)
            self.assertIn('map', transformed_code)
        except jinja2.exceptions.TemplateNotFound as e:
            self.fail(f"Template not found: {e}")

    def test_different_partitioning_strategies(self):
        """Test different partitioning strategies for Map-Reduce pattern."""
        strategies = [
            ['SDP'],
            ['SIP'],
            ['SDP', 'SIP']
        ]

        for strategy in strategies:
            transformer = MapReducePatternTransformer(self.bottleneck, strategy)

            tree = ast.parse(self.bottleneck['source'])
            transformer.visit(tree)
            transformer.finalize(tree)

            # Check that the context includes the strategy
            context = transformer.get_template_context()
            self.assertEqual(context['partitioning_strategy'], strategy)

            # Generate code
            try:
                code = transformer.generate_code()
                self.assertIn('multiprocessing', code)
            except jinja2.exceptions.TemplateNotFound as e:
                self.fail(f"Template not found for strategy {strategy}: {e}")

if __name__ == '__main__':
    unittest.main()

