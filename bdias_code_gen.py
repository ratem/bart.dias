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

    def generate_suggestions(self, structured_code):
        """Generates parallelization suggestions based on the structured code."""
        suggestions = []
        for item_type in structured_code:
            if item_type == 'loops':
                for loop in structured_code['loops']:
                    suggestion = self.handle_loop(loop)
                    if suggestion:
                        suggestions.append(suggestion)
            elif item_type == 'list_comprehensions':
                  for list_comp in structured_code['list_comprehensions']:
                    suggestion = self.handle_list_comp(list_comp)
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
                "loop_var": loop.get('loop_var',''),  # Use get() to avoid KeyError
                "iterable_name": loop.get('iterable_name',''),  # Use get() to avoid KeyError
                "code_suggestion": self.suggest_parallel_nested_loop(loop),
                "llm_suggestion": ""
            }
        elif opportunity_type == 'while':
            return {
                "lineno": loop["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": 'loop',  # for while loops the suggestion is similar to basic loops
                "code_suggestion": self.suggest_parallel_while(loop),
                "llm_suggestion": ""
            }
        elif opportunity_type == 'loop':  # Only access loop_var and iterable_name if it's a 'for' loop
            return {
                "lineno": loop["lineno"],
                "opportunity_type": opportunity_type,
                "explanation_index": "loop",
                "loop_var": loop.get('loop_var', ''),  # Use get() to avoid KeyError
                "iterable_name": loop.get('iterable_name', ''),  # Use get() to avoid KeyError
                "code_suggestion": self.suggest_parallel_loop(loop),
                "llm_suggestion": ""
            }

        return None # If it is not a loop returns None

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

    def suggest_parallel_while(self, loop):
        """Suggests code for parallelizing a while loop (not really parallelizable)."""
        loop_body = loop.get("body", "")  # Use get, again

        # Detect indentation of the original loop:
        try:
            indentation = loop_body.index(loop_body.strip()[0])  # Get indentation of first line
        except ValueError:
            indentation = 0

        code_suggestion = f"# Parallelization of while loops is generally not straightforward\n# due to the condition being checked in each iteration.\n"

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
    def get_llm_suggestion(self, prompt):
        """Makes a call to Gemini for a code suggestion."""
        return " " # To ensure that it exists!