from bdias_parser import BDiasParser
from bdias_assist import BDiasAssist
from bdias_code_gen import BDiasCodeGen
import subprocess

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
                            "Calls to '{func_name}' could be parallelized."
    }

    # Partitioning suggestions corresponding to the indices used in find_parallelization_opportunities
    PARTITIONING_SUGGESTIONS = {
        'loop': "For partitioning the data, you could consider spatial, temporal, or hash-based partitioning, "
                "depending on the nature of the data and the operations being performed.",
        'nested loop': "For nested loop parallelization, you might apply a two-level approach. "
                       "Consider first parallelizing the outer loop and then, within each process "
                       "or thread of the outer loop are independent, this loop could be parallelized.",
        'function': "If you're using the Fork-Join pattern, consider instruction partitioning to divide the "
                    "function calls into independent subtasks. For Master-Worker, you might use sharding or vertical "
                    "partitioning to distribute data among the workers.",
        'recursive function': "For recursive function parallelization, consider techniques like divide and conquer to "
                              "break down the problem into smaller subproblems that can be solved independently.",
         'function call': "Calls of this kind could potentially be moved into different threads/processes by instruction partitioning.",
        'list comprehension': "For partitioning the data within the list comprehension, you could "
                              "consider spatial, temporal, or hash-based partitioning, depending on "
                              "the nature of the data and the operations being performed.",
         'loop and function': "For this 'loop and function' scenario, you could combine partitioning strategies. "
                             "Partition the data for the loop and also consider instruction partitioning for the function calls."
    }
    parser = BDiasParser()
    code_generator = BDiasCodeGen(EXPLANATIONS, PARTITIONING_SUGGESTIONS)
    assistant = BDiasAssist(parser, code_generator)
    assistant.run_interactive_session()
