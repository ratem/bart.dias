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
from typing import Dict, Any


def present_transformation(original_code: str, transformed_code: str,
                           pattern_info: Dict[str, Any]) -> str:
    """Conditionally show original code based on 'show_original' flag"""

    output = [
        f"=== {pattern_info['pattern'].upper()} Pattern Transformation ===",
        f"Description: {get_pattern_description(pattern_info['pattern'])}",
        f"Partitioning Strategy: {', '.join(pattern_info['partitioning_strategy'])}"
        "\n=== Parallelized Code ===",
        transformed_code,
        "\n=== Transformation Explanation ===",
        generate_explanation(pattern_info)
    ]

    return '\n'.join(output)


def generate_explanation(pattern_info: Dict[str, Any]) -> str:
    """Generate focused explanation without errors."""
    pattern = pattern_info['pattern']
    strategies = pattern_info['partitioning_strategy']

    explanations = {
        'pipeline': {
            'default': (
                "This pipeline transformation parallelizes sequential stages using multiprocessing.Queues for "
                "inter-process communication. The key changes include:\n"
                "1. Separated each processing stage into independent processes\n"
                "2. Added queue-based communication between stages\n"
                "3. Implemented batched processing for better throughput"
            ),
            'TDP': "Temporal Domain Partitioning with batch processing between pipeline stages",
            'TIP': "Temporal Instruction Partitioning with parallel workers in each stage"
        },
        'map_reduce': {
            'default': (
                "Parallelized using a pool of workers for the map phase and a tree-based reduction. "
                f"Using {len(strategies)} workers with {strategies[0]} partitioning."
            ),
            'SDP': "Spatial Domain Partitioning of input data across workers",
            'SIP': "Spatial Instruction Partitioning with parallel task execution"
        }
    }

    pattern_explanations = explanations.get(pattern, {})
    if not pattern_explanations:
        return f"This transformation parallelizes the {pattern.upper()} pattern using {', '.join(strategies)} strategies."

    # Find the first matching strategy explanation
    for strategy in strategies:
        if strategy in pattern_explanations:
            return pattern_explanations[strategy]

    return pattern_explanations.get('default', f"Applied {pattern.upper()} pattern with recommended partitioning.")


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
        'pool_worker': "The pool-workers pattern involves a master process distributing tasks to worker processes. "
                         "It is useful for load balancing and dynamic task allocation.",
        'fork_join': "The Fork-Join pattern involves splitting a task into subtasks, executing them in parallel, "
                     "and then joining the results. It is useful for recursive divide-and-conquer algorithms.",
        'divide_conquer': "The Divide and Conquer pattern involves recursively breaking down a problem into smaller subproblems, "
                          "solving them independently, and combining the results.",
        'scatter_gather': "The Scatter-Gather pattern involves distributing data across processes, "
                          "processing it independently, and then collecting the results."
    }

    return descriptions.get(pattern, f"The {pattern.upper()} pattern is a parallel programming pattern.")
