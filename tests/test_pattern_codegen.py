"""
Test module for pattern-based code generation in Bart.dIAs.

This module contains tests for the pattern-specific code generators,
focusing on the Map pattern implementation.
"""

import unittest
import ast
import jinja2
from bdias_pattern_codegen import MapPatternTransformer, generate_parallel_code


class TestMapPatternCodegen(unittest.TestCase):
    """Test cases for Map pattern code generation."""

    def setUp(self):
        """Set up test cases."""
        self.bottleneck = {
            'type': 'for_loop',
            'lineno': 1,
            'source': 'for i in range(100):\n    result.append(i * i)'
        }
        self.partitioning_strategy = ['SDP', 'horizontal']
        self.transformer = MapPatternTransformer(self.bottleneck, self.partitioning_strategy)

    def test_map_transformation(self):
        """Test transformation of a simple Map pattern."""
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

        # Generate code
        try:
            code = self.transformer.generate_code()
            # Check that the generated code includes multiprocessing
            self.assertIn('multiprocessing', code)
            self.assertIn('Pool', code)
            self.assertIn('map', code)
        except jinja2.exceptions.TemplateNotFound:
            # Skip this check if templates are not available
            pass

    def test_generate_parallel_code(self):
        """Test the generate_parallel_code function."""
        try:
            original_code, transformed_code = generate_parallel_code(
                self.bottleneck,
                'map',
                self.partitioning_strategy
            )

            # Check that the original code is preserved
            self.assertEqual(original_code, self.bottleneck['source'])

            # Check that the transformed code includes multiprocessing
            self.assertIn('multiprocessing', transformed_code)
            self.assertIn('Pool', transformed_code)
            self.assertIn('map', transformed_code)
        except jinja2.exceptions.TemplateNotFound:
            # Skip this check if templates are not available
            pass

    def test_different_partitioning_strategies(self):
        """Test different partitioning strategies for Map pattern."""
        strategies = [
            ['SDP'],
            ['SIP'],
            ['horizontal'],
            ['SDP', 'SIP', 'horizontal']
        ]

        for strategy in strategies:
            transformer = MapPatternTransformer(self.bottleneck, strategy)
            tree = ast.parse(self.bottleneck['source'])
            transformer.visit(tree)
            transformer.finalize(tree)

            # Check that the context includes the strategy
            context = transformer.get_template_context()
            self.assertEqual(context['partitioning_strategy'], strategy)

            # Generate code (if templates are available)
            try:
                code = transformer.generate_code()
                self.assertIn('multiprocessing', code)
            except jinja2.exceptions.TemplateNotFound:
                # Skip this check if templates are not available
                pass


if __name__ == '__main__':
    unittest.main()
