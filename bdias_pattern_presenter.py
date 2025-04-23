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
    """
    Present the transformation from sequential to parallel code in plain text format.

    Args:
        original_code: The original sequential code
        transformed_code: The parallelized code
        pattern_info: Dictionary containing pattern information

    Returns:
        Formatted presentation as a string
    """
    # Generate explanation
    explanation = generate_explanation(pattern_info)
    pattern = pattern_info['pattern']

    # Format the output as plain text
    output = [
        f"=== {pattern.upper()} Pattern Transformation ===\n",
        f"Pattern: {pattern.upper()}",
        f"Description: {get_pattern_description(pattern)}",
        f"Partitioning Strategy: {', '.join(pattern_info['partitioning_strategy'])}",
        "\n=== Original Code ===\n",
        original_code,
        "\n=== Parallelized Code ===\n",
        transformed_code,
        "\n=== Transformation Explanation ===\n",
        explanation
    ]

    return "\n".join(output)


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
        'map_reduce': {
            'SDP': "This transformation applies Spatial Domain Partitioning to the Map-Reduce pattern. "
                   "The data is divided into chunks, each processed by a separate worker in the map phase. "
                   "The results are then combined in the reduce phase. This approach is efficient for large datasets.",
            'SIP': "This transformation applies Spatial Instruction Partitioning to the Map-Reduce pattern. "
                   "The same map operation is applied to different data elements in parallel, followed by a "
                   "tree-based reduction phase. This approach is efficient for computations with independent map "
                   "operations followed by associative reductions.",
            'default': "This transformation parallelizes the Map-Reduce pattern using Python's multiprocessing module. "
                       "Data elements are processed independently in the map phase, and the results are combined "
                       "in the reduce phase."
        }
        # ... other patterns ...
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
        'map_reduce': "The Map-Reduce pattern involves applying the same operation independently to each element "
                     "in a dataset (map phase) and then combining the results using an associative operation "
                     "(reduce phase). It is highly scalable and widely used for processing large datasets.",
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
