import multiprocessing as mp
from functools import partial
import ast

class BDiasCodeGen:
    """
    Analyzes the code and generates parallelization suggestions.
    """

    def __init__(self, explanations, partitioning_suggestions):
        """Initializes the generator with explanations and partitioning suggestions."""
        self.EXPLANATIONS = explanations
        self.PARTITIONING_SUGGESTIONS = partitioning_suggestions
        self.model = None  # Set to None, skip Gemini for now


    def handle_function(self, function):
        """Generates suggestions for functions."""
        opportunity_type = function.get('type', '')

        if opportunity_type == 'recursive function definition':
            return {
                "lineno": function["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "recursive function",
                "func_name": function.get("name", ""),
                "code_suggestion": self.suggest_parallel_recursive_function(function),
                "llm_suggestion": ""
            }
        elif opportunity_type == 'function':
            return {
                "lineno": function["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "function",
                "func_name": function.get("name", ""),
                "code_suggestion": self.suggest_parallel_function(function),
                "llm_suggestion": ""
            }
        elif opportunity_type == 'function call':
            return {
                "lineno": function["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "function call",
                "func_name": function.get("func_name", ""),
                "parent_func_name": function.get("parent_func_name", ""),
                "code_suggestion": self.suggest_parallel_function_call(function),
                "llm_suggestion": ""
            }
        elif opportunity_type == 'loop and function':
            return {
                "lineno": function["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "loop and function",
                "func_name": function.get("func_name", ""),
                "loop_var": function.get("loop_var", ""),
                "iterable_name": function.get("iterable_name", ""),
                "code_suggestion": self.suggest_parallel_loop_with_function(function),
                "llm_suggestion": ""
            }

        return None

    def suggest_parallel_recursive_function(self, function):
        """Suggests code for parallelizing a recursive function."""
        function_body = function.get("body", "")
        function_name = function.get("name", "function_name")

        try:
            indentation = function_body.index(function_body.strip()[0])
        except ValueError:
            indentation = 0

        code_suggestion = f"""
    {' ' * indentation}import multiprocessing as mp
    {' ' * indentation}from functools import partial

    {' ' * indentation}def {function_name}_parallel(n, num_processes=4):
    {' ' * (indentation + 4)}# Base case handling
    {' ' * (indentation + 4)}if n <= 1:  # Adjust base case as needed
    {' ' * (indentation + 4)}    return {function_name}(n)
    {' ' * (indentation + 4)}
    {' ' * (indentation + 4)}# Split the problem
    {' ' * (indentation + 4)}with mp.Pool(processes=num_processes) as pool:
    {' ' * (indentation + 4)}    # Divide the problem into smaller subproblems
    {' ' * (indentation + 4)}    subproblems = [n-i for i in range(1, num_processes+1) if n-i > 0]
    {' ' * (indentation + 4)}    results = pool.map({function_name}, subproblems)
    {' ' * (indentation + 4)}    
    {' ' * (indentation + 4)}    # Combine results according to your recursive formula
    {' ' * (indentation + 4)}    # Example for Fibonacci: return sum(results)
    {' ' * (indentation + 4)}    return sum(results)  # Adjust as needed for your specific recursive function
    """

        return code_suggestion

    def suggest_parallel_function(self, function):
        """Suggests code for parallelizing a function."""
        function_body = function.get("body", "")
        function_name = function.get("name", "function_name")

        try:
            indentation = function_body.index(function_body.strip()[0])
        except ValueError:
            indentation = 0

        code_suggestion = f"""
    {' ' * indentation}import multiprocessing as mp

    {' ' * indentation}def {function_name}_parallel(args_list):
    {' ' * (indentation + 4)}with mp.Pool() as pool:
    {' ' * (indentation + 4)}    results = pool.starmap({function_name}, args_list)
    {' ' * (indentation + 4)}return results

    {' ' * indentation}# Example usage:
    {' ' * indentation}# args_list = [(arg1_1, arg1_2, ...), (arg2_1, arg2_2, ...), ...]
    {' ' * indentation}# results = {function_name}_parallel(args_list)
    """

        return code_suggestion

    def suggest_parallel_function_call(self, function):
        """Suggests code for parallelizing a function call."""
        function_name = function.get("func_name", "function_name")
        parent_function_name = function.get("parent_func_name", "parent_function_name")

        code_suggestion = f"""
    import multiprocessing as mp

    def {parent_function_name}_parallel(args_list):
        # Prepare arguments for {function_name}
        function_args = []
        for args in args_list:
            # Process args as needed
            function_args.append(args)

        # Parallelize the function call
        with mp.Pool() as pool:
            results = pool.starmap({function_name}, function_args)

        return results

    # Example usage:
    # args_list = [(arg1_1, arg1_2, ...), (arg2_1, arg2_2, ...), ...]
    # results = {parent_function_name}_parallel(args_list)
    """

        return code_suggestion

    def suggest_parallel_loop_with_function(self, function):
        """Suggests code for parallelizing a loop that calls a function."""
        function_name = function.get("func_name", "function_name")
        loop_var = function.get("loop_var", "item")
        iterable_name = function.get("iterable_name", "iterable")

        code_suggestion = f"""
    import multiprocessing as mp

    def process_item_with_function({loop_var}):
        # Process each item with the function
        result = {function_name}({loop_var})
        return result

    if __name__ == '__main__':
        with mp.Pool() as pool:
            results = pool.map(process_item_with_function, {iterable_name})

        # Process results as needed
        # ...
    """

        return code_suggestion

    def suggest_parallel_while(self, loop):
        """Suggests code for parallelizing a while loop."""
        loop_body = loop.get("body", "")

        # Detect indentation of the original loop
        try:
            indentation = loop_body.index(loop_body.strip()[0])
        except ValueError:
            indentation = 0

        code_suggestion = f"""
    {' ' * indentation}import multiprocessing as mp

    {' ' * indentation}# While loops are challenging to parallelize directly because:
    {' ' * indentation}# 1. The termination condition is checked in each iteration
    {' ' * indentation}# 2. Each iteration may depend on the previous one
    {' ' * indentation}# 3. The number of iterations is not known in advance

    {' ' * indentation}# Approach 1: Convert to a bounded iteration if possible
    {' ' * indentation}def process_chunk(chunk_id, max_iterations=100):
    {' ' * indentation}    result = []
    {' ' * indentation}    # Initialize any variables needed for your while condition
    {' ' * indentation}    condition = True  # Replace with your actual condition
    {' ' * indentation}    iteration = 0
    {' ' * indentation}    
    {' ' * indentation}    while condition and iteration < max_iterations:
    {' ' * indentation}        # Your original while loop body here
    {' ' * indentation}        # ...
    {' ' * indentation}        
    {' ' * indentation}        # Update condition based on your logic
    {' ' * indentation}        iteration += 1
    {' ' * indentation}    
    {' ' * indentation}    return result

    {' ' * indentation}# Approach 2: If the while loop processes items from a queue/collection
    {' ' * indentation}def parallel_process_queue(work_queue):
    {' ' * indentation}    with mp.Pool() as pool:
    {' ' * indentation}        # Distribute chunks of work
    {' ' * indentation}        chunk_size = max(1, work_queue.qsize() // mp.cpu_count())
    {' ' * indentation}        chunks = [[] for _ in range(mp.cpu_count())]
    {' ' * indentation}        
    {' ' * indentation}        # Distribute items across chunks
    {' ' * indentation}        i = 0
    {' ' * indentation}        while not work_queue.empty():
    {' ' * indentation}            item = work_queue.get()
    {' ' * indentation}            chunks[i % len(chunks)].append(item)
    {' ' * indentation}            i += 1
    {' ' * indentation}        
    {' ' * indentation}        # Process chunks in parallel
    {' ' * indentation}        results = pool.map(process_chunk, chunks)
    {' ' * indentation}        
    {' ' * indentation}    return results
    """

        return code_suggestion


    def generate_suggestions(self, structured_code):
        """Generates parallelization suggestions based on the structured code."""
        suggestions = []

        for item_type in structured_code:
            if item_type == 'loops':
                for loop in structured_code['loops']:
                    suggestion = self.handle_loop(loop)
                    if suggestion:
                        suggestions.append(suggestion)
            elif item_type == 'functions':
                for function in structured_code['functions']:
                    suggestion = self.handle_function(function)
                    if suggestion:
                        suggestions.append(suggestion)
            elif item_type == 'list_comprehensions':
                for list_comp in structured_code['list_comprehensions']:
                    suggestion = self.handle_list_comp(list_comp)
                    if suggestion:
                        suggestions.append(suggestion)
            elif item_type == 'combos':
                for combo in structured_code['combos']:
                    suggestion = self.handle_combo(combo)
                    if suggestion:
                        suggestions.append(suggestion)

        return suggestions

    def handle_loop(self, loop):
        """Generates suggestions for for loops."""
        opportunity_type = loop.get('type', '')

        if opportunity_type == 'nested loop':
            return {
                "lineno": loop["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "nested loop",
                "loop_var": loop.get('loop_var', ''),
                "iterable_name": loop.get('iterable_name', ''),
                "code_suggestion": self.suggest_parallel_nested_loop(loop),
                "llm_suggestion": ""
            }
        elif opportunity_type == 'while':
            return {
                "lineno": loop["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": 'loop',
                "loop_var": "",  # Empty string for while loops
                "iterable_name": "",  # Empty string for while loops
                "code_suggestion": self.suggest_parallel_while(loop),
                "llm_suggestion": ""
            }
        elif opportunity_type == 'loop':
            return {
                "lineno": loop["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "loop",
                "loop_var": loop.get('loop_var', ''),
                "iterable_name": loop.get('iterable_name', ''),
                "code_suggestion": self.suggest_parallel_loop(loop),
                "llm_suggestion": ""
            }

        return None

    def handle_list_comp(self, list_comp):
         return {
            "lineno": list_comp["lineno"],
            "opportunity_type": list_comp["type"],
            "explanation_index": 'list comprehension',
             "code_suggestion": self.suggest_parallel_listcomp(list_comp)
           }

    def suggest_parallel_loop(self, loop):
        """Suggests code for parallelizing a basic loop."""
        loop_body = loop["body"]
        loop_var = loop.get("loop_var", "item")
        iterable_name = loop.get("iterable_name", "iterable")

        # Detect indentation of the original loop:
        try:
            indentation = loop_body.index(loop_body.strip()[0])  # Get indentation of first line
        except ValueError:
            indentation = 0

        code_suggestion = f"""
    {' ' * indentation}import multiprocessing
    {' ' * indentation}def process_item({loop_var}):
    {' ' * (indentation + 4)}result = [] #Initialize an empty list to hold the results of each iteration
    """
        for line in loop_body.splitlines():  # Needed .splitlines here!
            code_suggestion += f"{' ' * (indentation + 4)}{line}\n"  # Indent the loop body
        code_suggestion += f"""{' ' * (indentation + 4)}result.append(res) # Assuming the result of an iteration is 'res'
    {' ' * indentation}    return result
    {' ' * indentation}if __name__ == '__main__': #Needed the if here!
    {' ' * indentation}    with multiprocessing.Pool() as pool:
    {' ' * indentation}        results = pool.map(process_item, {iterable_name})
    """
        return code_suggestion


    def suggest_parallel_nested_loop(self, loop):
        """Suggests code for parallelizing a nested loop."""

        loop_body = loop["body"]
        loop_var = loop.get("loop_var", "item")
        iterable_name = loop.get("iterable_name", "iterable")

        try:
            indentation = loop_body.index(loop_body.strip()[0])  # Get indentation of first line
        except ValueError:
            indentation = 0
        code_suggestion = f"""
    {' ' * indentation}import multiprocessing
    {' ' * indentation}def process_nested_loop({loop_var}):
    {' ' * (indentation + 4)}result = []  # Initialize result list
    """

        for line in loop_body.splitlines():  # Needed splitlines
            code_suggestion += f"{' ' * (indentation + 4)}{line}\n"  # Indent the loop body

        code_suggestion += f"""{' ' * (indentation + 4)}result.append(res)  # Assuming each iteration returns 'res'
    {' ' * indentation}    return result
    {' ' * indentation}if __name__ == '__main__': #Needed the if here!
    {' ' * indentation}   with multiprocessing.Pool() as pool:
    {' ' * indentation}       results = pool.map(process_nested_loop, {iterable_name})
    """
        return code_suggestion

    def suggest_parallel_listcomp(self, list_comp):
      """Suggests code for parallelizing a list comprehension."""
      listcomp_body = list_comp["body"]
      indentation = listcomp_body.index(listcomp_body.strip()[0])  # Get indentation
      code_suggestion = f"""
{' '*indentation}import multiprocessing
{' '*indentation}def process_item(item):
"""
      code_suggestion+=f"""{' '*(indentation+4)}    return {listcomp_body}
{' '*indentation}with multiprocessing.Pool() as pool:
{' '*indentation}    results = pool.map(process_item, iterable)
"""
      return code_suggestion

    def handle_combo(self, combo):
        """Generates suggestions for combo patterns."""
        combo_type = combo.get("type", "")

        if "for_in_while" in combo_type:
            return {
                "lineno": combo["lineno"],
                "opportunity_type": combo_type,
                "explanation_index": "for_in_while",
                "loop_var": combo.get("loop_var", ""),
                "iterable_name": combo.get("iterable_name", ""),
                "code_suggestion": self.suggest_parallel_combo_loop(combo),
                "llm_suggestion": ""
            }
        elif "while_with_for" in combo_type:
            return {
                "lineno": combo["lineno"],
                "opportunity_type": combo_type,
                "explanation_index": "while_with_for",
                "loop_var": combo.get("loop_var", ""),
                "iterable_name": combo.get("iterable_name", ""),
                "code_suggestion": self.suggest_parallel_combo_loop(combo),
                "llm_suggestion": ""
            }
        elif "for_with_recursive_call" in combo_type:
            return {
                "lineno": combo["lineno"],
                "opportunity_type": combo_type,
                "explanation_index": "for_with_recursive_call",
                "func_name": combo.get("recursive_calls", [""])[0],
                "loop_var": combo.get("loop_var", ""),
                "iterable_name": combo.get("iterable_name", ""),
                "code_suggestion": self.suggest_parallel_recursive_loop(combo),
                "llm_suggestion": ""
            }
        elif "for_with_loop_functions" in combo_type:
            return {
                "lineno": combo["lineno"],
                "opportunity_type": combo_type,
                "explanation_index": "for_with_loop_functions",
                "func_name": combo.get("loop_function_calls", [""])[0],
                "loop_var": combo.get("loop_var", ""),
                "iterable_name": combo.get("iterable_name", ""),
                "code_suggestion": self.suggest_parallel_loop_with_functions(combo),
                "llm_suggestion": ""
            }
        elif "while_with_loop_functions" in combo_type:
            return {
                "lineno": combo["lineno"],
                "opportunity_type": combo_type,
                "explanation_index": "while_with_loop_functions",
                "func_name": combo.get("loop_function_calls", [""])[0],
                "loop_var": combo.get("loop_var", ""),
                "iterable_name": combo.get("iterable_name", ""),
                "code_suggestion": self.suggest_parallel_loop_with_functions(combo),
                "llm_suggestion": ""
            }

        return None

    def suggest_parallel_recursive_loop(self, combo):
        """Suggests code for parallelizing a loop with recursive function calls."""
        loop_body = combo["body"]
        loop_var = combo.get("loop_var", "item")
        iterable_name = combo.get("iterable_name", "iterable")
        recursive_calls = combo.get("recursive_calls", [])

        try:
            indentation = loop_body.index(loop_body.strip()[0])
        except ValueError:
            indentation = 0

        code_suggestion = f"""
    {' ' * indentation}import multiprocessing as mp
    {' ' * indentation}from functools import partial

    {' ' * indentation}def process_item_with_recursion({loop_var}):
    {' ' * (indentation + 4)}result = []  # Initialize result list
    """

        for line in loop_body.splitlines():
            code_suggestion += f"{' ' * (indentation + 4)}{line}\n"

        code_suggestion += f"""
    {' ' * indentation}    return result

    {' ' * indentation}if __name__ == '__main__':
    {' ' * indentation}    with mp.Pool() as pool:
    {' ' * indentation}        results = pool.map(process_item_with_recursion, {iterable_name})
    {' ' * indentation}    
    {' ' * indentation}    # Note: When parallelizing recursive functions, consider:
    {' ' * indentation}    # 1. Setting an appropriate recursion depth limit
    {' ' * indentation}    # 2. Using a work-stealing approach for better load balancing
    {' ' * indentation}    # 3. Implementing memoization to avoid redundant calculations
    """

        return code_suggestion

    def suggest_parallel_combo_loop(self, combo):
        """Suggests code for parallelizing a combo of while and for loops."""
        loop_body = combo["body"]

        try:
            indentation = loop_body.index(loop_body.strip()[0])
        except ValueError:
            indentation = 0

        code_suggestion = f"""
    {' ' * indentation}import multiprocessing as mp

    {' ' * indentation}# For a combination of while and for loops, consider:
    {' ' * indentation}# 1. Restructuring the code to separate the loops
    {' ' * indentation}# 2. Converting the while loop to a for loop if possible
    {' ' * indentation}# 3. Parallelizing only the inner loops

    {' ' * indentation}def process_chunk(chunk):
    {' ' * (indentation + 4)}results = []
    {' ' * (indentation + 4)}# Process the chunk with appropriate logic
    {' ' * (indentation + 4)}return results

    {' ' * indentation}if __name__ == '__main__':
    {' ' * indentation}    # Divide work into chunks
    {' ' * indentation}    chunks = [...]  # Define your chunks based on the problem
    {' ' * indentation}    
    {' ' * indentation}    with mp.Pool() as pool:
    {' ' * indentation}        all_results = pool.map(process_chunk, chunks)
    """

        return code_suggestion

    def suggest_parallel_loop_with_functions(self, combo):
        """Suggests code for parallelizing a loop with function calls that contain loops."""
        loop_body = combo["body"]
        loop_var = combo.get("loop_var", "item")
        iterable_name = combo.get("iterable_name", "iterable")
        loop_function_calls = combo.get("loop_function_calls", [])

        try:
            indentation = loop_body.index(loop_body.strip()[0])
        except ValueError:
            indentation = 0

        code_suggestion = f"""
    {' ' * indentation}import multiprocessing as mp

    {' ' * indentation}def process_item_with_nested_functions({loop_var}):
    {' ' * (indentation + 4)}result = []  # Initialize result list
    """

        for line in loop_body.splitlines():
            code_suggestion += f"{' ' * (indentation + 4)}{line}\n"

        code_suggestion += f"""
    {' ' * indentation}    return result

    {' ' * indentation}if __name__ == '__main__':
    {' ' * indentation}    with mp.Pool() as pool:
    {' ' * indentation}        results = pool.map(process_item_with_nested_functions, {iterable_name})
    {' ' * indentation}    
    {' ' * indentation}    # Note: When parallelizing loops with nested function calls:
    {' ' * indentation}    # 1. Consider the granularity of parallelization
    {' ' * indentation}    # 2. Be aware of potential overhead from creating too many processes
    {' ' * indentation}    # 3. Ensure that the functions called within the loop are thread-safe
    """

        return code_suggestion

    def get_llm_suggestion(self, prompt):
        """Makes a call to Gemini for a code suggestion."""
        return " " # To ensure that it exists!