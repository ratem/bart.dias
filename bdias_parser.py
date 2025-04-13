"""
BDiasParser: Code Parsing and Analysis Module for Bart.dIAs

This module is responsible for parsing Python code and identifying parallelization
opportunities. It performs static analysis on the Abstract Syntax Tree (AST) to
detect various code patterns and analyze dependencies.

Features:
- Parses Python code using the ast module
- Identifies parallelizable patterns such as loops, functions, and list comprehensions
- Performs dependency analysis to determine parallelization safety
- Detects complex combo patterns like nested loops and recursive function calls
- Analyzes data flow and cross-function dependencies
- Provides methods to check if loops, functions, and list comprehensions are parallelizable

Classes:
- BDiasParser: Main class for parsing and analyzing Python code

Dependencies:
- ast (Python standard library)
"""


import ast


class BDiasParser:
    """
    Parses Python code and identifies parallelization opportunities,
    filtering based on dependency checks.
    """

    def __init__(self):
        """Initializes the parser."""
        pass

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


    def _calculate_nesting_depth(self, node, tree):
        """Calculate the nesting depth of a loop."""
        depth = 1  # Start with depth 1 for the current loop

        # Find all parent nodes that are loops
        current = node
        for parent in ast.walk(tree):
            if parent != current and (isinstance(parent, ast.For) or isinstance(parent, ast.While)):
                # Check if current is inside parent
                if current in ast.walk(parent):
                    depth += 1
                    current = parent

        return depth

    def _find_function_def(self, function_name, tree):
        """Find the function definition node for a given function name."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None

    def _is_recursive_function(self, function_node):
        """Check if a function contains recursive calls to itself."""
        function_name = function_node.name
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == function_name:
                    return True
        return False

    def _function_contains_loops(self, function_node):
        """Check if a function contains loops."""
        for node in ast.walk(function_node):
            if isinstance(node, (ast.For, ast.While)):
                return True
        return False


    def _extract_loop_data(self, node, tree):
        """Extract data for loops (for or while) with enhanced combo detection."""
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):  # Simple loop variable
                loop_var = node.target.id
                iterable_name = self._get_iterable_name(node.iter)

                # Calculate nesting depth
                nesting_depth = self._calculate_nesting_depth(node, tree)

                # Check if this for loop is inside a while loop
                is_in_while = False
                for parent in ast.walk(tree):
                    if parent != node and isinstance(parent, ast.While) and node in ast.walk(parent):
                        is_in_while = True
                        break

                # Check for recursive function calls inside this for loop
                recursive_calls = []
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                        # Find the function definition
                        func_def = self._find_function_def(subnode.func.id, tree)
                        if func_def:
                            # Check if this function is recursive
                            is_recursive = self._is_recursive_function(func_def)
                            if is_recursive:
                                recursive_calls.append(subnode.func.id)

                # Check for function calls inside this loop that themselves contain loops
                loop_function_calls = []
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                        # Find the function definition
                        func_def = self._find_function_def(subnode.func.id, tree)
                        if func_def:
                            # Check if this function contains loops
                            has_loops = self._function_contains_loops(func_def)
                            if has_loops:
                                loop_function_calls.append(subnode.func.id)

                # Check for nested loops (corrected logic)
                is_nested = False
                for parent in ast.walk(tree):  # Corrected nested loop check
                    if parent != node and isinstance(parent, ast.For) and node in ast.walk(parent):
                        is_nested = True
                        break

                # Determine loop type based on various conditions
                if is_in_while:
                    loop_type = "for_in_while"
                elif recursive_calls:
                    loop_type = "for_with_recursive_call"
                elif loop_function_calls:
                    loop_type = "for_with_loop_functions"
                elif is_nested:
                    loop_type = "nested loop"
                else:
                    loop_type = "loop"

                loop_data = {
                    "type": loop_type,
                    "lineno": node.lineno,
                    "loop_var": loop_var,
                    "iterable_name": iterable_name,
                    "nesting_depth": nesting_depth,
                    "body": ast.unparse(node)
                }

                # Add additional data based on loop type
                if recursive_calls:
                    loop_data["recursive_calls"] = recursive_calls
                if loop_function_calls:
                    loop_data["loop_function_calls"] = loop_function_calls

                return loop_data

            return None  # Returns None for complex loops, for instance, tuple unpacking

        elif isinstance(node, ast.While):
            # Check for for loops inside this while loop
            has_for_loop = False
            for_loops_inside = []
            for subnode in ast.walk(node):
                if subnode != node and isinstance(subnode, ast.For):
                    has_for_loop = True
                    for_loops_inside.append(subnode.lineno)

            # Check for function calls inside this loop that themselves contain loops
            loop_function_calls = []
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                    # Find the function definition
                    func_def = self._find_function_def(subnode.func.id, tree)
                    if func_def:
                        # Check if this function contains loops
                        has_loops = self._function_contains_loops(func_def)
                        if has_loops:
                            loop_function_calls.append(subnode.func.id)

            # Calculate nesting depth
            nesting_depth = self._calculate_nesting_depth(node, tree)

            # Determine loop type
            if has_for_loop:
                loop_type = "while_with_for"
            elif loop_function_calls:
                loop_type = "while_with_loop_functions"
            else:
                loop_type = "while"

            loop_data = {
                "type": loop_type,
                "lineno": node.lineno,
                "nesting_depth": nesting_depth,
                "body": ast.unparse(node)
            }

            # Add additional data based on loop type
            if has_for_loop:
                loop_data["for_loops_inside"] = for_loops_inside
            if loop_function_calls:
                loop_data["loop_function_calls"] = loop_function_calls

            return loop_data

    def is_loop_parallelizable(self, loop_node):
        """Checks if a loop is potentially parallelizable."""
        loop_var = None
        if isinstance(loop_node, ast.For):
            if isinstance(loop_node.target, ast.Name):
                loop_var = loop_node.target.id
            else:
                return False  # Skip loops with complex targets

        dependency_graph = self.build_dependency_graph(loop_node)
        data_flow_deps = self.analyze_data_flow(loop_node)

        # Check for loop-carried dependencies
        for write_line, read_line, var in data_flow_deps:
            if var != loop_var and var in dependency_graph:
                return False

        # Check for break/continue statements
        for node in ast.walk(loop_node):
            if isinstance(node, (ast.Break, ast.Continue)):
                return False

        return True

    def is_function_parallelizable(self, function_node):
        """Checks if a function is potentially parallelizable."""
        dependency_graph = self.build_dependency_graph(function_node)
        data_flow_deps = self.analyze_data_flow(function_node)
        modified_globals = self.track_cross_function_dependencies(function_node)

        if modified_globals:
            return False  # Function modifies global variables

        for var, deps in dependency_graph.items():
            if deps and var not in [arg.arg for arg in function_node.args.args]:
                for dep in deps:
                    if dep in [arg.arg for arg in function_node.args.args]:
                        return False  # Dependency on a parameter

        # Check for recursive calls
        recursive_analysis = self.analyze_recursive_calls(function_node)
        if recursive_analysis["calls"] and not recursive_analysis["parallelizable"]:
            return False  # Recursive calls are not independent

        return True

    def is_listcomp_parallelizable(self, listcomp_code):
        """Checks if a list comprehension is potentially parallelizable."""
        try:
            tree = ast.parse(listcomp_code)
            dependency_graph = self.build_dependency_graph(tree)
            data_flow_deps = self.analyze_data_flow(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if not any(isinstance(parent, ast.comprehension) and
                               isinstance(node, ast.Name) and
                               node.id == parent.target.id for parent in ast.walk(tree)):
                        return False
                    #if not any(isinstance(parent, ast.comprehension) and node.id == parent.target.id for parent in ast.walk(tree)):
                    #    return False

            return True
        except SyntaxError:
            return False

    def parse(self, code):
        """
        Parse Python code and identify parallelization opportunities.

        This enhanced version includes more sophisticated dependency analysis
        to better identify safe parallelization opportunities.
        """
        try:
            self.tree = ast.parse(code)  # Store the full AST for cross-function analysis
            return self._analyze_tree(self.tree)
        except SyntaxError as e:
            print(f"Syntax error in code at line {e.lineno}: {e.msg}")
            return None

    def _analyze_tree(self, tree):
        """Analyzes the AST and extracts parallelization opportunities with enhanced combo detection."""
        structured_code = {
            "loops": [],
            "functions": [],
            "list_comprehensions": [],
            "combos": []  # New category for combo patterns
        }

        processed_functions = set()  # Keep track of functions already analyzed

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_data = self._extract_loop_data(node, tree)

                if loop_data:
                    # Store dependency analysis results with the loop data
                    loop_data["dependency_graph"] = self.build_dependency_graph(node)
                    loop_data["data_flow_deps"] = self.analyze_data_flow(node)

                    # Check if this is a combo pattern
                    if any(combo_type in loop_data.get("type", "") for combo_type in
                           ["for_in_while", "while_with_for", "for_with_recursive_call",
                            "for_with_loop_functions", "while_with_loop_functions"]):
                        structured_code["combos"].append(loop_data)
                    elif self.is_loop_parallelizable(node):
                        structured_code["loops"].append(loop_data)

            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
                if function_name in processed_functions:
                    continue

                processed_functions.add(function_name)
                function_data = self._extract_function_data(node, tree, function_name)

                if function_data:
                    # Enhanced dependency check for functions
                    is_recursive = any(f.get('type') == 'recursive function' for f in function_data)

                    # Add dependency analysis results
                    for data in function_data:
                        if data["type"] == "function":
                            data["dependency_graph"] = self.build_dependency_graph(node)
                            data["data_flow_deps"] = self.analyze_data_flow(node)
                            data["cross_func_deps"] = self.track_cross_function_dependencies(node)

                            if is_recursive:
                                data["recursive_analysis"] = self.analyze_recursive_calls(node)

                            data["type"] = "recursive function definition" if is_recursive else "function"

                            if self.is_function_parallelizable(node):
                                structured_code["functions"].append(data)

            elif isinstance(node, ast.ListComp):
                listcomp_data = self._extract_list_comprehension_data(node)

                if listcomp_data:
                    # Enhanced dependency check for list comprehensions
                    listcomp_data["dependency_graph"] = self.build_dependency_graph(node)

                    if self.is_listcomp_parallelizable(listcomp_data.get("body", "")):
                        structured_code["list_comprehensions"].append(listcomp_data)

        return structured_code

    # Static Dependency Graph Construction

    def build_dependency_graph(self, node):
        """
        Build a comprehensive graph representing data dependencies between statements.

        This graph tracks which variables depend on other variables, allowing us to
        determine if operations can be safely parallelized.

        Args:
            node: The AST node to analyze (typically a function or loop body)

        Returns:
            A dictionary mapping variable names to sets of variables they depend on
        """
        dependency_graph = {}
        defined_vars = set()
        used_vars = set()

        # First pass: identify all variable definitions and uses
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Assign):
                # Track variable definitions (left side of assignments)
                for target in subnode.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)
                        # Initialize an empty dependency set for this variable
                        dependency_graph[target.id] = set()
                    elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                        # Handle tuple unpacking (e.g., a, b = func())
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                defined_vars.add(elt.id)
                                dependency_graph[elt.id] = set()
            elif isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                # Track variable uses (right side of expressions)
                used_vars.add(subnode.id)

        # Second pass: connect dependencies between variables
        for var in used_vars:
            for def_var in defined_vars:
                if self._may_depend_on(var, def_var, node):
                    dependency_graph[def_var].add(var)

        return dependency_graph

    def _may_depend_on(self, var1, var2, node):
        """
        Determine if var1 may depend on var2 based on their usage in the code.
        """
        # Skip self-dependencies
        if var1 == var2:
            return False

        # Rest of the method remains the same...
        # Find all assignments to var2
        var2_assignments = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Assign):
                for target in subnode.targets:
                    if isinstance(target, ast.Name) and target.id == var2:
                        var2_assignments.append(subnode)

        # Find all uses of var1
        var1_uses = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load) and subnode.id == var1:
                var1_uses.append(subnode)

        # Check if any var1 use comes after a var2 assignment in the same scope
        for assignment in var2_assignments:
            for use in var1_uses:
                if hasattr(assignment, 'lineno') and hasattr(use, 'lineno'):
                    if assignment.lineno < use.lineno:
                        # Check if they're in the same scope
                        if self._in_same_scope(assignment, use, node):
                            return True

        return False

    def _in_same_scope(self, node1, node2, parent_node):
        """
        Check if two nodes are in the same scope within a parent node.

        Args:
            node1: First AST node
            node2: Second AST node
            parent_node: Parent AST node containing both nodes

        Returns:
            Boolean indicating if both nodes are in the same scope
        """
        # Find the smallest common ancestor of node1 and node2
        for subnode in ast.walk(parent_node):
            if isinstance(subnode, (ast.FunctionDef, ast.For, ast.While, ast.If)):
                if node1 in ast.walk(subnode) and node2 in ast.walk(subnode):
                    return True

        return False
    # Data Flow Analysis

    def analyze_data_flow(self, node):
        """
        Analyze data flow to detect read-after-write dependencies.

        This function tracks all variable reads and writes, then identifies
        cases where a variable is read after being written to, which indicates
        a data dependency that might prevent parallelization.

        Args:
            node: The AST node to analyze

        Returns:
            A list of tuples (write_line, read_line, variable_name) representing dependencies
        """
        reads = {}  # Maps variable names to lines where they're read
        writes = {}  # Maps variable names to lines where they're written

        # Track line numbers for variable reads and writes
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name):
                var_name = subnode.id
                lineno = getattr(subnode, 'lineno', 0)

                if isinstance(subnode.ctx, ast.Store):
                    # Variable is being written to
                    if var_name not in writes:
                        writes[var_name] = []
                    writes[var_name].append(lineno)
                elif isinstance(subnode.ctx, ast.Load):
                    # Variable is being read from
                    if var_name not in reads:
                        reads[var_name] = []
                    reads[var_name].append(lineno)

        # Detect read-after-write dependencies
        dependencies = []
        for var in writes:
            if var in reads:
                for write_line in writes[var]:
                    for read_line in reads[var]:
                        if read_line > write_line:
                            dependencies.append((write_line, read_line, var))

        # Sort dependencies by the write line number for clearer output
        dependencies.sort()
        return dependencies

    # Cross-Function Dependency Tracking

    def track_cross_function_dependencies(self, function_node):
        """
        Track dependencies across function boundaries.

        This function identifies which global variables are modified by functions
        called within the given function, allowing us to determine if the function
        can be safely parallelized.

        Args:
            function_node: The AST node of the function to analyze

        Returns:
            A set of global variable names that are modified by called functions
        """
        called_functions = []
        modified_globals = set()

        # Find all function calls within this function
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_functions.append(node.func.id)

        # Check if called functions modify globals
        for func_name in called_functions:
            # Pass self.tree as the second argument to _find_function_def
            func_node = self._find_function_def(func_name, self.tree)
            if func_node and self.uses_global_variables(func_node):
                # Identify which globals are modified
                for subnode in ast.walk(func_node):
                    if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Store):
                        # Check if this is a global variable (not a function parameter or local)
                        if not self._is_local_to_function(subnode.id, func_node):
                            modified_globals.add(subnode.id)

        return modified_globals

    def _find_function_def(self, function_name, tree):
        """
        Find the function definition node for a given function name.

        Args:
            function_name: The name of the function to find

        Returns:
            The AST node for the function definition, or None if not found
        """

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None

    def _is_local_to_function(self, var_name, function_node):
        """
        Check if a variable is local to a function (parameter or local variable).

        Args:
            var_name: The name of the variable to check
            function_node: The AST node of the function

        Returns:
            Boolean indicating if the variable is local to the function
        """
        # Check if it's a function parameter
        for arg in function_node.args.args:
            if arg.arg == var_name:
                return True

        # Check if it's defined within the function
        for node in ast.walk(function_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return True

        return False

    # Recursive Call Analysis

    def analyze_recursive_calls(self, function_node):
        """
        Analyze recursive function calls and their dependencies.

        This function identifies recursive calls within a function and determines
        if they can be safely parallelized based on their dependencies.

        Args:
            function_node: The AST node of the function to analyze

        Returns:
            A dictionary containing information about recursive calls and their parallelizability
        """
        function_name = function_node.name
        recursive_calls = []

        # Find all recursive calls within the function
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == function_name:
                    # Extract arguments to the recursive call
                    args = []
                    for arg in node.args:
                        args.append(ast.unparse(arg))
                    recursive_calls.append({
                        "lineno": node.lineno,
                        "args": args
                    })

        # Analyze if recursive calls can be parallelized
        parallelizable = self._are_recursive_calls_independent(recursive_calls, function_node)

        return {
            "calls": recursive_calls,
            "parallelizable": parallelizable
        }

    def _are_recursive_calls_independent(self, recursive_calls, function_node):
        """
        Determine if recursive calls are independent and can be parallelized.

        This function analyzes the arguments to recursive calls to determine
        if they operate on independent data and can be safely parallelized.

        Args:
            recursive_calls: List of dictionaries containing information about recursive calls
            function_node: The AST node of the function containing the recursive calls

        Returns:
            Boolean indicating if the recursive calls can be parallelized
        """
        # If there are no recursive calls, they're trivially independent
        if not recursive_calls:
            return True

        # Check if recursive calls have overlapping data dependencies
        # This is a simplified check - in reality, we'd need more sophisticated analysis
        args_sets = [set(call["args"]) for call in recursive_calls]
        for i in range(len(args_sets)):
            for j in range(i + 1, len(args_sets)):
                if args_sets[i].intersection(args_sets[j]):
                    return False

        # Check for shared state modifications
        # If the function modifies shared state, recursive calls might not be independent
        if self.uses_global_variables(function_node):
            return False

        return True
