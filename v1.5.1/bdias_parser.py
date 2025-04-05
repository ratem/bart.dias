import ast

class BDiasParser:
    """
    Parses Python code to extract relevant information for parallelization.
    """

    def __init__(self):
        """Initializes the parser."""
        pass

    def parse(self, code):
      """Parses the given Python code using ast.
      Args:
        code: The Python code as a string.
      Returns:
          A dictionary containing the structured representation of the code,
          or None if there's a parsing error.
      """
      try:
        tree = ast.parse(code)
        return self.analyze_tree(tree)
      except SyntaxError as e:
        print(f"Error: Syntax error in provided code, check line: {e.lineno}")
        return None

    def analyze_tree(self, tree):
        """Analyzes the AST and extracts information.
        Args:
          tree: The parsed AST.
        Returns:
          A dictionary containing the structured representation of the code
        """
        structured_code = {
          "loops": [],
          "functions": [],
           "list_comprehensions": []
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for nested loops by examining parent nodes
                is_nested = False
                for parent in ast.walk(tree):
                    if parent != node and isinstance(parent, ast.For) and node in ast.walk(parent):
                        is_nested = True
                        break

                if isinstance(node.target, ast.Name):
                    loop_var = node.target.id
                    iterable_name = self.get_iterable_name(node.iter)
                    opportunity_type = 'nested loop' if is_nested else 'loop'  # Distinguish loop types

                    # Add the loop as a potential opportunity even if not parallelizable
                    structured_code["loops"].append({
                        "type": opportunity_type,
                        "lineno": node.lineno,
                        "loop_var": loop_var,
                        "iterable_name": iterable_name,
                        "body": [ast.unparse(body_line) for body_line in node.body]  #Extract source code for loop body
                   })
            elif isinstance(node, ast.FunctionDef):
                func_name = node.name

                # Check for loops outside the function that call the function
                for other_node in ast.walk(tree):
                  if (isinstance(other_node, ast.For) and
                    node not in ast.walk(other_node)): #Make sure the loop is not inside the function
                     for inner_node in ast.walk(other_node):
                       if (isinstance(inner_node, ast.Call) and
                          isinstance(inner_node.func, ast.Name) and
                         inner_node.func.id == func_name):
                         loop_var = other_node.target.id #Extract loop variable from the outer loop
                         iterable_name = self.get_iterable_name(other_node.iter) #Extract iterable name from the outer loop
                         structured_code["functions"].append({
                             "type": 'loop and function',
                            "lineno": other_node.lineno,
                             "func_name": func_name,
                             "loop_var": loop_var,
                             "iterable_name": iterable_name
                          })
                         break #No need to continue searching in this loop

                # Check for recursive function calls within this function definition:
                for inner_node in ast.walk(node): #Iterate through the function's body
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Name) and inner_node.func.id == func_name:
                             structured_code["functions"].append({
                              "type": 'recursive function',
                               "lineno": inner_node.lineno,
                               "func_name": func_name
                            })

                # Add the function as a potential opportunity
                structured_code["functions"].append({
                    "type": "function",
                    "lineno": node.lineno,
                     "name": node.name,
                      "body": [ast.unparse(body_line) for body_line in node.body], #Extract source code for function body
                    "args": [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, ast.ListComp):
                structured_code["list_comprehensions"].append({
                  "type": "list_comprehension",
                  "lineno": node.lineno,
                  "body": ast.unparse(node) #Extract source code of list comp
                })
        return structured_code

    def get_iterable_name(self, iterable_node):
      """Attempts to extract the name of the iterable being used in a loop.
      """
      if isinstance(iterable_node, ast.Name):
          return iterable_node.id
      elif isinstance(iterable_node, ast.Call):
        if isinstance(iterable_node.func, ast.Name) and iterable_node.func.id == 'range':
          return 'a range of numbers'
      return "the sequence"