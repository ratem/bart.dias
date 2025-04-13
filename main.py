"""
Bart.dIAs: Main Entry Point

This module serves as the main entry point for the Bart.dIAs system, a Python
assistant that analyzes code to identify and suggest parallelization opportunities
using the multiprocessing module.

Features:
- Initializes and coordinates the main components of Bart.dIAs
- Sets up explanations and partitioning suggestions for various code patterns
- Launches the interactive session for code analysis

Components Initialized:
- BDiasParser: For parsing and analyzing Python code
- BDiasCodeGen: For generating parallelization suggestions
- BDiasAssist: For handling user interactions and presenting results

Usage:
Run this script to start an interactive session with Bart.dIAs.
"""


from bdias_parser import BDiasParser
from bdias_assist import BDiasAssist
from bdias_code_gen import BDiasCodeGen


if __name__ == "__main__":
    # Explanations corresponding to the indices used in find_parallelization_opportunities
    EXPLANATIONS = {
        'loop': "This 'for' loop iterates over {iterable_name} using the loop variable '{loop_var}'. "
                "If the operations performed on each item in {iterable_name} are independent, "
                "this loop could be parallelized.",
        'nested loop': "This nested 'for' loop could be parallelized. It iterates over {iterable_name} using "
                       "the loop variable '{loop_var}'. If the operations performed on each item in the nested "
                       "loop are independent, this loop could be parallelized.",
        'function': "The function '{func_name}' has minimal side effects and its calls with different "
                    "arguments appear to be independent, making it potentially parallelizable.",
        'recursive function': "The recursive call to '{func_name}' could potentially be executed in parallel.",
        'function call': "The function '{func_name}' is called inside a function called '{parent_func_name}'.",
        'list comprehension': "This list comprehension processes elements independently, making it potentially parallelizable.",
        'loop and function': "This loop is of the type 'loop and function'. It iterates over {iterable_name} using the loop variable '{loop_var}'. "
                             "Calls to '{func_name}' could be parallelized.",
        # Add new explanations for combo patterns
        'for_with_recursive_call': "This 'for' loop contains calls to the recursive function '{func_name}'. These recursive calls could potentially be executed in parallel for different loop iterations.",
        'while_with_for': "This 'while' loop contains 'for' loops that could be parallelized. Consider restructuring to allow parallel execution of the inner loops.",
        'for_in_while': "This 'for' loop is inside a 'while' loop. Consider parallelizing the 'for' loop iterations within each 'while' loop iteration.",
        'for_with_loop_functions': "This 'for' loop calls the function '{func_name}' which itself contains loops. Consider parallelizing both the outer loop iterations and the inner function loops.",
        'while_with_loop_functions': "This 'while' loop calls functions that contain loops. Consider parallelizing the function calls within each while loop iteration."
    }

    # Partitioning suggestions corresponding to the indices used in find_parallelization_opportunities
    PARTITIONING_SUGGESTIONS = {
        'loop': "For partitioning the data, you could consider spatial, temporal, or hash-based partitioning, depending on the nature of the data and the operations being performed.",
        'nested loop': "For nested loop parallelization, you might apply a two-level approach. Consider first parallelizing the outer loop and then, within each process or thread of the outer loop are independent, this loop could be parallelized.",
        'function': "If you're using the Fork-Join pattern, consider instruction partitioning to divide the function calls into independent subtasks. For Master-Worker, you might use sharding or vertical partitioning to distribute data among the workers.",
        'recursive function': "For recursive function parallelization, consider techniques like divide and conquer to break down the problem into smaller subproblems that can be solved independently.",
        'function call': "Calls of this kind could potentially be moved into different threads/processes by instruction partitioning.",
        'list comprehension': "For partitioning the data within the list comprehension, you could consider spatial, temporal, or hash-based partitioning, depending on the nature of the data and the operations being performed.",
        'loop and function': "For this 'loop and function' scenario, you could combine partitioning strategies. Partition the data for the loop and also consider instruction partitioning for the function calls.",
        'for_with_recursive_call': "For loops with recursive calls, consider a two-level parallelization approach: parallelize the loop iterations and use techniques like divide and conquer for the recursive calls.",
        'while_with_for': "For while loops containing for loops, consider parallelizing the inner for loops while maintaining the sequential nature of the while loop iterations.",
        'for_in_while': "For for loops inside while loops, consider parallelizing the for loop iterations within each while loop iteration, potentially using a thread pool that persists across iterations.",
        'for_with_loop_functions': "For loops calling functions with loops, consider a nested parallelization approach: parallelize the outer loop iterations and optimize the inner function loops separately.",
        'while_with_loop_functions': "For while loops calling functions with loops, consider parallelizing the function calls within each while iteration, being careful to maintain any dependencies."
    }

    parser = BDiasParser()
    code_generator = BDiasCodeGen(EXPLANATIONS, PARTITIONING_SUGGESTIONS)
    assistant = BDiasAssist(parser, code_generator)
    assistant.run_interactive_session()
