"""
BDiasPatternPresenter: Pattern-Based Code Transformation Presentation Module for Bart.dIAs

This module handles the presentation of pattern-based code transformations, providing
clear visualizations and explanations of the transformations.

Features:
- Side-by-side comparison of original and transformed code
- Explanations of transformations based on pattern characteristics
- Visualization of pattern-specific concepts (data distribution, etc.)

Functions:
- present_transformation: Present the transformation from sequential to parallel code
- generate_explanation: Generate an explanation of the transformation

Dependencies:
- jinja2: For template-based presentation
"""

import jinja2
from typing import Dict, Any


def present_transformation(original_code: str, transformed_code: str, pattern_info: Dict[str, Any]) -> str:
    # Initialize Jinja2 environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader('templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    # Load the appropriate template
    pattern = pattern_info['pattern']
    template_name = f"{pattern}/visualization.html"

    try:
        template = env.get_template(template_name)
    except jinja2.exceptions.TemplateNotFound:
        # Fall back to default template
        template = env.get_template("base/code_transformation.html")

    # Generate explanation
    explanation = generate_explanation(pattern_info)

    # Combine context from pattern_info with additional visualization context
    context = {
        'original_code': original_code,
        'transformed_code': transformed_code,
        'pattern_name': pattern.upper(),
        'pattern_description': get_pattern_description(pattern),
        'partitioning_strategy': ', '.join(pattern_info['partitioning_strategy']),
        'explanation': explanation,
        # Add defaults for visualization variables
        'data_size': 10,
        'processor_count': 4,
        'processor_ranges': [(p, p * 3, min((p + 1) * 3, 10)) for p in range(4)],
        'elements_per_processor': 3
    }

    # Add the transformer context if available
    if 'context' in pattern_info:
        context.update(pattern_info['context'])

    # Render the template
    return template.render(**context)


def generate_explanation(pattern_info: Dict[str, Any]) -> str:
    """
    Generate an explanation of the transformation based on pattern and strategy.

    Args:
        pattern_info: Dictionary containing pattern information

    Returns:
        Explanation as a string
    """
    pattern = pattern_info['pattern']
    partitioning_strategy = pattern_info['partitioning_strategy']

    explanations = {
        'map': {
            'SDP': "This transformation applies Spatial Domain Partitioning to the Map pattern. "
                   "The data is divided into chunks, each processed by a separate worker. "
                   "This approach is efficient for large datasets that can be easily divided.",
            'SIP': "This transformation applies Spatial Instruction Partitioning to the Map pattern. "
                   "The same operation is applied to different data elements in parallel. "
                   "This approach is efficient for independent operations on data elements.",
            'default': "This transformation parallelizes the Map pattern using Python's multiprocessing module. "
                       "The same operation is applied to different data elements in parallel."
        }
    }

    # Get explanation based on pattern and partitioning strategy
    if pattern in explanations:
        for strategy in partitioning_strategy:
            if strategy in explanations[pattern]:
                return explanations[pattern][strategy]
        return explanations[pattern]['default']

    return f"This transformation parallelizes the {pattern.upper()} pattern using appropriate techniques."


def get_pattern_description(pattern: str) -> str:
    """
    Get a description of the pattern.

    Args:
        pattern: Pattern name

    Returns:
        Pattern description as a string
    """
    descriptions = {
        'map': "The Map pattern involves applying the same operation independently to each element in a dataset. "
               "It is highly parallelizable because each operation is independent of the others.",
        'stencil': "The Stencil pattern involves updating array elements based on neighboring elements. "
                   "It is common in scientific computing and image processing.",
        'pipeline': "The Pipeline pattern involves dividing a task into a series of stages, "
                    "with data flowing through the stages. Each stage can be executed in parallel.",
        'master_worker': "The Master-Worker pattern involves a master process distributing tasks to worker processes. "
                         "It is useful for load balancing and dynamic task allocation.",
        'reduction': "The Reduction pattern involves combining multiple elements into a single result "
                     "using an associative operation. It is common in aggregation operations.",
        'fork_join': "The Fork-Join pattern involves splitting a task into subtasks, executing them in parallel, "
                     "and then joining the results. It is useful for recursive divide-and-conquer algorithms.",
        'divide_conquer': "The Divide and Conquer pattern involves recursively breaking down a problem into smaller subproblems, "
                          "solving them independently, and combining the results.",
        'scatter_gather': "The Scatter-Gather pattern involves distributing data across processes, "
                          "processing it independently, and then collecting the results."
    }

    return descriptions.get(pattern, f"The {pattern.upper()} pattern is a parallel programming pattern.")
