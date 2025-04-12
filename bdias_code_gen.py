
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
                "code_suggestion": self.suggest_parallel_loop_with_functions(function),
                "llm_suggestion": ""
            }

        return None

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

    def suggest_parallel_listcomp(self, list_comp):
        """Suggests code for parallelizing a list comprehension."""
        loop_var = list_comp.get("loop_var", "item")
        iterable = list_comp.get("iterable", "iterable")
        body = list_comp.get("body", "item")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing

    def process_item({loop_var}):
        return {body}

    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            results = pool.map(process_item, {iterable})
        # results now contains what the list comprehension would produce
    """

        return code_suggestion


    def suggest_parallel_recursive_function(self, function):
        """Suggests code for parallelizing a recursive function."""
        function_name = function.get("name", "function_name")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing as mp
    from functools import partial

    def {function_name}_parallel(n, num_processes=4):
        # Base case handling
        if n <= 1:  # Adjust base case as needed
            return {function_name}(n)

        # Split the problem
        with mp.Pool(processes=num_processes) as pool:
            # Divide the problem into smaller subproblems
            subproblems = [n-i for i in range(1, num_processes+1) if n-i > 0]
            results = pool.map({function_name}, subproblems)

            # Combine results according to your recursive formula
            # Example for Fibonacci: return sum(results)
            return sum(results)  # Adjust as needed for your specific recursive function
    """

        return code_suggestion


    def suggest_parallel_function(self, function):
        """Suggests code for parallelizing a function."""
        function_name = function.get("name", "function_name")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing as mp

    def {function_name}_parallel(args_list):
        with mp.Pool() as pool:
            results = pool.starmap({function_name}, args_list)
        return results

    # Example usage:
    # args_list = [(arg1_1, arg1_2, ...), (arg2_1, arg2_2, ...), ...]
    # results = {function_name}_parallel(args_list)
    """

        return code_suggestion

    def suggest_parallel_function_call(self, function):
        """Suggests code for parallelizing a function call."""
        function_name = function.get("func_name", "function_name")
        parent_function_name = function.get("parent_func_name", "parent_function_name")

        # Create a template with proper indentation
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


    def suggest_parallel_loop_with_functions(self, combo):
        """Suggests code for parallelizing a loop with function calls that contain loops."""
        loop_var = combo.get("loop_var", "item")
        iterable_name = combo.get("iterable_name", "iterable")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing as mp

    def process_item_with_nested_functions({loop_var}):
        result = []
        # LOOP_BODY_PLACEHOLDER
        return result

    if __name__ == '__main__':
        with mp.Pool() as pool:
            results = pool.map(process_item_with_nested_functions, {iterable_name})
        # Process results as needed
    """

        # Replace the placeholder with properly indented loop body
        indented_body = ""
        for line in combo.get("body", "").splitlines():
            if line.strip():
                indented_body += "    " + line + "\n"
            else:
                indented_body += "\n"

        # Replace the placeholder
        code_suggestion = code_suggestion.replace("    # LOOP_BODY_PLACEHOLDER", indented_body)

        return code_suggestion

    def suggest_parallel_while(self, loop):
        """Suggests code for parallelizing a while loop."""
        condition = loop.get("condition", "condition")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing as mp

    # While loops are challenging to parallelize directly because:
    # 1. The termination condition is checked in each iteration
    # 2. Each iteration may depend on the previous one
    # 3. The number of iterations is not known in advance

    # Approach 1: Convert to a bounded iteration if possible
    def process_chunk(chunk_id, max_iterations=100):
        result = []
        # Initialize any variables needed for your while condition
        condition = {condition}
        iteration = 0

        while condition and iteration < max_iterations:
            # LOOP_BODY_PLACEHOLDER

            # Update condition based on your logic
            iteration += 1

        return result

    # Approach 2: If the while loop processes items from a queue/collection
    def parallel_process_queue(work_queue):
        with mp.Pool() as pool:
            # Distribute chunks of work
            chunk_size = max(1, work_queue.qsize() // mp.cpu_count())
            chunks = [[] for _ in range(mp.cpu_count())]

            # Distribute items across chunks
            i = 0
            while not work_queue.empty():
                item = work_queue.get()
                chunks[i % len(chunks)].append(item)
                i += 1

            # Process chunks in parallel
            results = pool.map(process_chunk, chunks)

        return results
    """

        # Replace the placeholder with properly indented loop body
        indented_body = ""
        for line in loop.get("body", "").splitlines():
            if line.strip():
                indented_body += "        " + line + "\n"
            else:
                indented_body += "\n"

        # Replace the placeholder
        code_suggestion = code_suggestion.replace("        # LOOP_BODY_PLACEHOLDER", indented_body)

        return code_suggestion

    def suggest_parallel_loop(self, loop):
        """Suggests code for parallelizing a basic loop."""
        loop_var = loop.get("loop_var", "item")
        iterable_name = loop.get("iterable_name", "iterable")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing

    def process_item({loop_var}):
        result = []
        # LOOP_BODY_PLACEHOLDER
        return result

    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            results = pool.map(process_item, {iterable_name})
    """

        # Replace the placeholder with properly indented loop body
        indented_body = ""
        for line in loop.get("body", "").splitlines():
            if line.strip():
                indented_body += "    " + line + "\n"
            else:
                indented_body += "\n"

        # Replace the placeholder
        code_suggestion = code_suggestion.replace("    # LOOP_BODY_PLACEHOLDER", indented_body)

        return code_suggestion


    def suggest_parallel_nested_loop(self, loop):
        """Suggests code for parallelizing a nested loop."""
        outer_var = loop.get("outer_var", "i")
        outer_iterable = loop.get("outer_iterable", "range(n)")
        inner_var = loop.get("inner_var", "j")
        inner_iterable = loop.get("inner_iterable", "range(m)")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing as mp
    import itertools

    def process_pair(pair):
        {outer_var}, {inner_var} = pair
        result = []
        # LOOP_BODY_PLACEHOLDER
        return result

    if __name__ == '__main__':
        pairs = list(itertools.product({outer_iterable}, {inner_iterable}))
        with mp.Pool() as pool:
            results = pool.map(process_pair, pairs)
        # Process results as needed
    """

        # Replace the placeholder with properly indented loop body
        indented_body = ""
        for line in loop.get("body", "").splitlines():
            if line.strip():
                indented_body += "    " + line + "\n"
            else:
                indented_body += "\n"

        # Replace the placeholder
        code_suggestion = code_suggestion.replace("    # LOOP_BODY_PLACEHOLDER", indented_body)

        return code_suggestion

    def suggest_parallel_recursive_loop(self, combo):
        """Suggests code for parallelizing a loop with recursive function calls."""
        loop_var = combo.get("loop_var", "item")
        iterable_name = combo.get("iterable_name", "iterable")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing as mp
    from functools import partial

    def process_item_with_recursion({loop_var}):
        result = []
        # LOOP_BODY_PLACEHOLDER
        return result

    if __name__ == '__main__':
        with mp.Pool() as pool:
            results = pool.map(process_item_with_recursion, {iterable_name})
        # Process results as needed
    """

        # Replace the placeholder with properly indented loop body
        indented_body = ""
        for line in combo.get("body", "").splitlines():
            if line.strip():
                indented_body += "    " + line + "\n"
            else:
                indented_body += "\n"

        # Replace the placeholder
        code_suggestion = code_suggestion.replace("    # LOOP_BODY_PLACEHOLDER", indented_body)

        return code_suggestion

    def suggest_parallel_combo_loop(self, combo):
        """Suggests code for parallelizing a combo of while and for loops."""
        outer_condition = combo.get("outer_condition", "condition")
        inner_var = combo.get("inner_var", "item")
        inner_iterable = combo.get("inner_iterable", "items")

        # Create a template with proper indentation
        code_suggestion = f"""
    import multiprocessing as mp
    from queue import Queue

    def process_chunk(chunk):
        results = []
        for {inner_var} in chunk:
            # LOOP_BODY_PLACEHOLDER
        return results

    def parallel_combo_loop(condition_func, data_generator):
        results = []
        with mp.Pool() as pool:
            while condition_func():
                chunk = data_generator()
                if not chunk:
                    break
                chunk_results = pool.apply_async(process_chunk, (chunk,))
                results.extend(chunk_results.get())
        return results

    if __name__ == '__main__':
        def check_condition():
            return {outer_condition}

        def generate_data():
            # Generate or fetch the next batch of data
            return {inner_iterable}

        final_results = parallel_combo_loop(check_condition, generate_data)
        # Process final_results as needed
    """

        # Replace the placeholder with properly indented loop body
        indented_body = ""
        for line in combo.get("body", "").splitlines():
            if line.strip():
                indented_body += "        " + line + "\n"  # Note: 8 spaces for double indentation
            else:
                indented_body += "\n"

        # Replace the placeholder
        code_suggestion = code_suggestion.replace("        # LOOP_BODY_PLACEHOLDER", indented_body)

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
