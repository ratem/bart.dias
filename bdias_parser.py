import ast

class BDiasParser:
    """
    Parses Python code and identifies parallelization opportunities,
    filtering based on dependency checks.
    """

    def __init__(self):
        """Initializes the parser."""
        pass

    def parse(self, code):
        try:
            tree = ast.parse(code)
            return self._analyze_tree(tree)
        except SyntaxError as e:
            print(f"Syntax error in code at line {e.lineno}: {e.msg}")  # More informative error message
            return None

    def _analyze_tree(self, tree):
        """Analyzes the AST and extracts parallelization opportunities."""

        structured_code = {
            "loops": [],
            "functions": [],
            "list_comprehensions": []
        }
        processed_functions = set()  # Keep track of functions already analyzed

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_data = self._extract_loop_data(node, tree)  # Passing the tree to check for nested loops
                if loop_data and self.is_loop_parallelizable(node):  # Check if parallelizable before storing
                    structured_code["loops"].append(loop_data)

            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
                if function_name in processed_functions:
                    continue

                processed_functions.add(function_name)

                function_data = self._extract_function_data(node, tree, function_name)  # Extract function before dependencies check
                if function_data:
                    is_recursive = any(f.get('type') == 'recursive function' for f in function_data)  # Check if it has been identified as recursive by the call check

                    if not self.uses_global_variables(node) and self.is_function_parallelizable(node):
                        for data in function_data:
                            if data["type"] == "function":
                                data["type"] = "recursive function definition" if is_recursive else "function"  # Mark function as recursive or not
                            structured_code["functions"].append(data)

            elif isinstance(node, ast.ListComp):
                listcomp_data = self._extract_list_comprehension_data(node)
                if listcomp_data and self.is_listcomp_parallelizable(listcomp_data.get("body", "")):  # Check dependencies
                    structured_code["list_comprehensions"].append(listcomp_data)

        return structured_code

    def _extract_loop_data(self, node, tree):  # Now takes the full tree
        """Extract data for loops (for or while)."""

        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name): # Simple loop variable
                loop_var = node.target.id
                iterable_name = self._get_iterable_name(node.iter)

                # Check for nested loops (corrected logic)
                is_nested = False
                for parent in ast.walk(tree):  # Corrected nested loop check
                    if parent != node and isinstance(parent, ast.For) and node in ast.walk(parent):
                        is_nested = True
                        break
                loop_type = "nested loop" if is_nested else "loop"

                return {
                    "type": loop_type,
                    "lineno": node.lineno,
                    "loop_var": loop_var,
                    "iterable_name": iterable_name,
                    "body": ast.unparse(node)  # Now returns the full loop, to be later checked
                }
            return None #Returns None for complex loops, for instance, tuple unpacking
        elif isinstance(node, ast.While):
            return {
                "type": "while",
                "lineno": node.lineno,
                "body": ast.unparse(node)  # Now returns the full loop, to be later checked
            }

    def _extract_function_data(self, node, tree, function_name):
      """Extract data for functions, loops calling functions, and function calls inside other functions."""
      function_data = [] # Store in a list since now can be more than one type for a function

      # Check for recursive function calls within this function definition:
      is_recursive = False
      for inner_node in ast.walk(node): #Iterate through the function's body
          if isinstance(inner_node, ast.Call):
              if isinstance(inner_node.func, ast.Name) and inner_node.func.id == function_name:
                   function_data.append({
                    "type": 'recursive function',
                     "lineno": inner_node.lineno,
                     "func_name": function_name
                  })
                   is_recursive = True
                   break #Stop recursive function checks here

      # Check for loops outside the function that call the function
      for other_node in ast.walk(tree):
        if (isinstance(other_node, (ast.For, ast.While)) and #Now considering While loops also
           node not in ast.walk(other_node)): #Make sure the loop is not inside the function

            if isinstance(other_node, ast.For): #Extract data for "For" Loops
                loop_var = other_node.target.id if isinstance(other_node.target, ast.Name) else None #Extract loop variable from the outer loop
                iterable_name = self._get_iterable_name(other_node.iter) if loop_var else None #Extract iterable name from the outer loop
            else: #Extract data for "While" Loops
                loop_var = None
                iterable_name = None

            for inner_node in ast.walk(other_node):
                if (isinstance(inner_node, ast.Call) and
                    isinstance(inner_node.func, ast.Name) and
                    inner_node.func.id == function_name):
                        function_data.append({
                            "type": 'loop and function',
                            "lineno": other_node.lineno,
                            "func_name": function_name,
                            "loop_var": loop_var,
                            "iterable_name": iterable_name,
                        })
                        break #No need to continue searching in this loop

        # Check for function calls inside the function
      for inner_node in ast.walk(node): #Iterate through the function's body
           if isinstance(inner_node, ast.Call):
                if isinstance(inner_node.func, ast.Name) and inner_node.func.id != function_name: #Avoid recursive calls from here
                    function_data.append({
                        "type": 'function call',
                        "lineno": inner_node.lineno,
                        "func_name": inner_node.func.id,
                        "parent_func_name": function_name
                    })
                    
      function_data.append({
           "type": "recursive function definition" if is_recursive else "function", #Mark as recursive
           "lineno": node.lineno,
           "name": node.name,
           "body": ast.unparse(node),  # Return the full function code, to check dependencies later
           "args": [arg.arg for arg in node.args.args]
        })
      return function_data # Now it always returns a list

    def _extract_list_comprehension_data(self, node):
        """Extract data for list comprehensions."""
        return {
            "type": "list_comprehension",
            "lineno": node.lineno,
            "body": ast.unparse(node)  # Store body to check later
        }

    def _get_iterable_name(self, iterable_node):
        """Attempts to extract the name of the iterable being used in a loop."""
        if isinstance(iterable_node, ast.Name):
            return iterable_node.id #Using the correct variable now
        elif isinstance(iterable_node, ast.Call):
            if isinstance(iterable_node.func, ast.Name) and iterable_node.func.id == 'range':
                return 'a range of numbers'
        return "the sequence"

    def uses_global_variables(self, function_node):
        """Checks if a function uses or modifies global variables."""
        for node in ast.walk(function_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Load)): # Now checks uses and modification of globals
                if node.id not in function_node.args.args and node.id != 'self':
                    current_node = node
                    while hasattr(current_node, 'parent'): # Search for a call node among parents
                        current_node = current_node.parent
                        if isinstance(current_node, ast.Call):
                           break #Stop if found a call
                    else: # If no call is among the parents, there might be a global
                       return True
        return False

    def is_loop_parallelizable(self, loop_node):
        """Checks if a loop is potentially parallelizable."""

        loop_var = None
        if isinstance(loop_node, ast.For):
            if isinstance(loop_node.target, ast.Name):
               loop_var = loop_node.target.id
            else:
              return False  # Skip loops with complex targets

        for node in ast.walk(loop_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if loop_var and node.id != loop_var:
                    return False
            elif isinstance(node, (ast.Break, ast.Continue)):
                return False

        return True


    def is_function_parallelizable(self, function_node):
        """Checks if a function is potentially parallelizable."""

        for node in ast.walk(function_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in function_node.args.args and node.id != 'self':
                    current = node
                    while hasattr(current, 'parent'): # Search for a call node among parents
                        current = current.parent
                        if isinstance(current, ast.Call):
                           break #Stop if found a call
                    else: # If no call is among the parents, there might be a global
                       return False  # Not parallelizable if stores to non-arguments outside a call
            elif isinstance(node, ast.Return) and any(isinstance(parent, (ast.For, ast.While)) for parent in ast.walk(function_node)):
                  return False
        return True

    def is_listcomp_parallelizable(self, listcomp_code):
        """Checks if a list comprehension is potentially parallelizable."""

        try:
            tree = ast.parse(listcomp_code)  # Parse the list comprehension body

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    # Ignore assignments to the list comprehension's target variable
                    if not any(isinstance(parent, ast.comprehension) and node.id == parent.target.id for parent in ast.walk(tree)):
                        return False

            return True
        except SyntaxError:
            return False