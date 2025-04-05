import ast

class BDiasCodeGen:
    """
    Analyzes the code and generates parallelization suggestions.
    """
    def __init__(self, explanations, partitioning_suggestions):
       """Initializes the generator with explanations and partitioning suggestions."""
       self.EXPLANATIONS = explanations
       self.PARTITIONING_SUGGESTIONS = partitioning_suggestions
       self.model = None #Set to None, skip Gemini for now

    def is_loop_parallelizable(self, loop_node):  # Now receives a node (again)
        """Checks if a loop is potentially parallelizable, now receiving an AST node."""

        if loop_node is None: #Handle empty loop body
             return False
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in [loop_node.target.id]:  # Check against loop variable
                    # Ignore variables assigned in outer loops (as in bdias140.py)
                    is_in_outer_loop = False
                    current_node = node
                    while current_node and current_node != loop_node:
                        if isinstance(current_node, ast.For) and node.id != current_node.target.id:
                            is_in_outer_loop = True
                            break
                        current_node = getattr(current_node, 'parent', None)  # Use getattr to prevent AttributeError

                    if not is_in_outer_loop:
                        return False
        for node in ast.walk(loop_node):  # Check for break/continue
            if isinstance(node, (ast.Break, ast.Continue)):
                return False
        return True  # Simplified checks, as before

    def is_function_parallelizable(self, function_node):  # Now receives a node
        """Checks if a function is potentially parallelizable."""
        if function_node is None:  # Handle empty function body
            return False

        for node in ast.walk(function_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in (arg.arg for arg in function_node.args.args) and node.id != "self":  # Now using function_node to extract args.
                    return False  # Accessing non-argument variables

        for node in ast.walk(function_node):
            if isinstance(node, ast.Return) and any(isinstance(parent, ast.For) for parent in ast.walk(function_node)):
                return False  # No returns inside for loops

        return True  # Simplified check for now (no bottlenecks, as in bdias140.py)

    def is_listcomp_parallelizable(self, listcomp_node):  # Now receives a node
        """Checks if a list comprehension is potentially parallelizable."""

        if listcomp_node is None: #Handle empty listcomp body
              return False

        return True  # All are potentially parallelizable for now (no checks)

    def uses_global_variables(self, function_node):  # Now receives a node
        """Checks if a function modifies global variables."""
        if function_node is None:  # Handle empty function body
              return False

        for node in ast.walk(function_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in [arg.arg for arg in function_node.args.args]:
                    return True  # Modifying a global variable
        return False

    def generate_suggestions(self, structured_code):
        """Generates parallelization suggestions based on the structured code."""

        suggestions = []
        processed_functions = set()  # Track processed functions
        for item_type in structured_code:
            if item_type == 'loops':
                for loop in structured_code['loops']:
                    # Corrected the code to create the AST node and call it in is_loop_parallelizable
                    try: #Handles possible errors when creating a node
                        loop_node = ast.parse("".join(loop['body'])).body[0] if loop.get('body') else None
                        if self.is_loop_parallelizable(loop_node):
                            suggestion = self.handle_loop(loop)
                            if suggestion:
                                suggestions.append(suggestion)
                    except (SyntaxError,AttributeError,IndexError,TypeError):
                        pass #Skip this loop

            elif item_type == 'functions':
                for function in structured_code['functions']:
                    function_body = function.get('body')
                    if function.get("type") == 'function':  # Check for common functions first
                         if function.get("name") not in processed_functions: #Prevent repeated output
                            try: #Handle errors when parsing body
                                function_node = ast.parse("".join(function_body)).body[0] if function_body else None
                                if self.is_function_parallelizable(function_node):
                                     suggestion = self.handle_function(function)
                                     if suggestion:
                                            suggestions.append(suggestion)
                                            processed_functions.add(function.get("name"))
                            except (SyntaxError,AttributeError,IndexError,TypeError):
                                  pass #Skip this function
                    elif function["type"] == "recursive function definition": #Handle recursive function calls
                         if function["name"] not in processed_functions:
                           try:#Handles possible errors when creating the node
                             function_node = ast.parse("".join(function['body'])).body[0] if function.get('body') else None
                             if self.is_function_parallelizable(function_node): #Check for parallelization
                                suggestion = self.handle_function(function)
                                if suggestion:
                                    suggestions.append(suggestion)
                                    processed_functions.add(function["name"])
                           except (SyntaxError,AttributeError,IndexError,TypeError):
                              pass #skip this recursive function

                    elif function["type"] == 'function call': #Handle function calls
                        if function.get("parent_func_name") not in processed_functions:
                            suggestion = self.handle_function(function)
                            if suggestion:
                                suggestions.append(suggestion)
                                processed_functions.add(function.get("parent_func_name"))
                    elif function["type"] == 'loop and function':
                         try: #Handles possible errors when creating the node
                           function_node = ast.parse("".join(function['body'])).body[0] if function.get('body') else None
                           if self.is_function_parallelizable(function_node): #Correctly passes a node
                               suggestion = self.handle_function(function) #Fixed call
                               if suggestion:
                                   suggestions.append(suggestion)

                         except (SyntaxError,AttributeError,IndexError,TypeError):
                                  pass #Skip this loop and function

            elif item_type == 'list_comprehensions':
                  for list_comp in structured_code['list_comprehensions']:
                     if self.is_listcomp_parallelizable(ast.parse(list_comp['body']).body[0] if list_comp.get('body') else None): #Check here
                          suggestion = self.handle_list_comp(list_comp)
                          if suggestion:
                              suggestions.append(suggestion)

        return suggestions

    def handle_loop(self, loop):
        """Generates suggestions for for loops."""
        opportunity_type = loop.get('type','')
        loop_var = loop.get('loop_var','')
        iterable_name = loop.get('iterable_name','')
        if opportunity_type=='nested loop':
            return{
                  "lineno": loop["lineno"],
                  "opportunity_type": opportunity_type,
                  "explanation_index": "nested loop",
                  "loop_var": loop_var,
                  "iterable_name": iterable_name,
                   "code_suggestion": self.suggest_parallel_nested_loop(loop),
                   "llm_suggestion": ""
               }
        elif opportunity_type == 'while':
            return {
                "lineno": loop["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": 'loop', #for while loops the suggestion is similar to basic loops
                "code_suggestion": self.suggest_parallel_while(loop),
                "llm_suggestion": ""
            }
        else:
            return {
                "lineno": loop["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "loop",
                "loop_var": loop_var,
                "iterable_name": iterable_name,
                "code_suggestion": self.suggest_parallel_loop(loop),
                "llm_suggestion": ""
            }


    def handle_function(self, function):
        """Generates suggestions for functions and function calls."""
        opportunity_type = function.get('type',"")
        if opportunity_type == 'recursive function definition':
           return {
               "lineno": function["lineno"],
                "opportunity_type": opportunity_type,
               "explanation_index": 'recursive function',
               "func_name": function.get('name',''),
               "code_suggestion": "Recursive function parallelization not yet implemented.",
                "llm_suggestion": ""
           }
        elif opportunity_type == 'loop and function':
            return{
                "lineno": function["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "loop and function",
                "func_name": function.get('func_name',''),
                "loop_var": function.get('loop_var',''),
                "iterable_name": function.get('iterable_name',''),
                 "code_suggestion": self.suggest_parallel_function(function), #I MADE THE CALL TO SUGGEST_PARALLEL_FUNCTION
                "llm_suggestion": ""
            }
        elif opportunity_type == 'function call':
             return {
              "lineno": function["lineno"],
              "opportunity_type": opportunity_type,
             "explanation_index": "function call",
             "func_name": function.get('func_name',""),
             "parent_func_name": function.get("parent_func_name", ""),
              "code_suggestion": "function calls may potentially be executed concurrently",
             "llm_suggestion": ""
               }
        else:
            return {
              "lineno": function["lineno"],
              "opportunity_type": opportunity_type,
              "explanation_index": "function",
              "func_name": function.get('name',''),
            "code_suggestion": self.suggest_parallel_function(function),
             "llm_suggestion": ""
           }

    def handle_list_comp(self, list_comp):
         return {
            "lineno": list_comp["lineno"],
            "opportunity_type": list_comp["type"],
            "explanation_index": 'list comprehension',
            "code_suggestion": self.suggest_parallel_listcomp(list_comp),
            "llm_suggestion": ""
           }

    def suggest_parallel_loop(self, loop):
        """Suggests code for parallelizing a basic loop."""
        loop_body = loop.get("body", "")  # Get the default value to protect against KeyError
        loop_var = loop.get("loop_var","item")
        iterable_name = "some list" #loop.get("iterable_name", "iterable")  #It was throwing Exceptions, so we're simplifying

        if not loop_body:
            return "" #If loop body is missing

        try:
             indentation = loop_body[0].index(loop_body[0].strip()[0]) # Get indentation

        except: #If can't find the index, set the identation to zero, otherwise, use the correct value
           indentation = 0

        code_suggestion = f"""
{' '*indentation}import multiprocessing
{' '*indentation}def process_item({loop_var}):
"""
        for line in loop_body:
              code_suggestion += f"{' '*(indentation+4)}{line}\n" # Indent the loop body
        code_suggestion += f"""{' '*indentation}   return result
{' '*indentation}with multiprocessing.Pool() as pool:
{' '*indentation}   results = pool.map(process_item, {iterable_name})
"""
        return code_suggestion

    def suggest_parallel_while(self, loop):
      """Suggests code for parallelizing a while loop."""
      loop_body =  loop.get("body", "")
      if not loop_body:
            return "" #If loop body is missing

      try:
        indentation = loop_body[0].index(loop_body[0].strip()[0]) if loop_body else 0 # Get indentation
      except:
          indentation = 0 #If no identation, define it as zero

      code_suggestion = f"""
{' '*indentation}import multiprocessing
{' '*indentation}def process_item():
"""
      for line in loop_body:
           code_suggestion += f"{' '*(indentation+4)}{line}\n" # Indent the loop body
      code_suggestion += f"""{' '*indentation}    return result
{' '*indentation}with multiprocessing.Pool(processes=2) as pool:
{' '*indentation}   results = pool.map(process_item, [None, None])  # Using [None, None] as two parameters
"""
      return code_suggestion

    def suggest_parallel_nested_loop(self, loop):
      """Suggests code for parallelizing a nested loop."""
      loop_body = loop["body"]
      loop_var = loop.get("loop_var","item")  # Default value if missing
      iterable_name =  "some list" #loop.get("iterable_name", "iterable")  # Default value if missing

      if not loop_body:
           return "" #If loop body is missing

      try:
          indentation = loop_body[0].index(loop_body[0].strip()[0])  # Get indentation
      except:
          indentation = 0

      code_suggestion = f"""
{' '*indentation}import multiprocessing
{' '*indentation}def process_outer_item({loop_var}):
"""
      for line in loop_body:
          code_suggestion += f"{' '*(indentation+4)}{line}\n" # Indent the loop body
      code_suggestion += f"""{' '*indentation}   return result
{' '*indentation}with multiprocessing.Pool() as pool:
{' '*indentation}   results = pool.map(process_outer_item, {iterable_name})
"""
      return code_suggestion

    def suggest_parallel_function(self, function):
        """Suggests code for parallelizing a function."""
        function_name = function["name"]
        function_body = function["body"]

        if not function_body:
           return ""  #If there is no code

        try:
          indentation = function_body[0].index(function_body[0].strip()[0])  # Get indentation
        except:
            indentation = 0 #Or define as zero.

        code_suggestion = f"""{' '*indentation}import multiprocessing
{' '*indentation}def {function_name}_worker(*args):
"""
        for line in function_body:
           code_suggestion += f"{' '*(indentation+4)}{line}\n"
        code_suggestion+= f"""{' '*indentation}   return result
{' '*indentation}if __name__ == '__main__':
{' '*indentation}    with multiprocessing.Pool() as pool:
{' '*indentation}       results = pool.map({function_name}_worker, list_of_arguments)
"""
        return code_suggestion

    def suggest_parallel_loop_function(self, function):
        """Suggests code for parallelizing a loop with function calls."""
        function_name = function.get("func_name", "")
        loop_var = function.get("loop_var","item")
        iterable_name = function.get("iterable_name","iterable")
        indentation = 0

        code_suggestion = f"""{' '*indentation}import multiprocessing
{' '*indentation}def {function_name}_worker({loop_var}):
{' '*(indentation+4)} # The code of the original function has to be included here.
{' '*indentation}   return result
{' '*indentation}if __name__ == '__main__':
{' '*indentation}    with multiprocessing.Pool() as pool:
{' '*indentation}       results = pool.map({function_name}_worker, {iterable_name})
"""
        return code_suggestion


    def suggest_parallel_listcomp(self, list_comp):
      """Suggests code for parallelizing a list comprehension."""
      listcomp_body = list_comp["body"]
      if not listcomp_body:
           return "" #If the list comprehension is empty

      try:
           indentation = listcomp_body.index(listcomp_body.strip()[0])  # Get indentation
      except:
           indentation = 0 #Or define as zero.
      code_suggestion = f"""
{' '*indentation}import multiprocessing
{' '*indentation}def process_item(item):
"""
      code_suggestion+=f"""{' '*(indentation+4)}    return {listcomp_body}
{' '*indentation}with multiprocessing.Pool() as pool:
{' '*indentation}    results = pool.map(process_item, iterable)
"""
      return code_suggestion
    def get_llm_suggestion(self, prompt):
        """Makes a call to Gemini for a code suggestion."""
        # It was removed to make it not the code
        return ""