"""
BDiasCodeGen: Code Generation Module for Bart.dIAs

This module is responsible for generating parallelization suggestions for Python code.
It uses Jinja2 templates to create properly indented and structured code suggestions
for various parallelization opportunities.

Features:
- Generates parallelization suggestions for loops, functions, and list comprehensions
- Uses Jinja2 templates for consistent and maintainable code generation
- Handles various combo patterns like nested loops, recursive calls, and loop-function combinations
- Provides explanations and partitioning suggestions for each parallelization opportunity
- Supports static profiling to identify computationally intensive code sections

Classes:
- BDiasCodeGen: Main class for generating parallelization suggestions

Dependencies:
- Jinja2
- os
"""


import os
from jinja2 import Environment, FileSystemLoader


class BDiasCodeGen:
    """
    Analyzes the code and generates parallelization suggestions.
    """

    def __init__(self, explanations, partitioning_suggestions):
        """Initializes the generator with explanations and partitioning suggestions."""
        self.EXPLANATIONS = explanations
        self.PARTITIONING_SUGGESTIONS = partitioning_suggestions
        # Set up Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        # Future LLM use
        self.model = None  # Skip Gemini for now

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

    def suggest_parallel_loop(self, loop):
        """Suggests code for parallelizing a basic loop using Jinja2."""
        template = self.jinja_env.get_template('parallel_loop.jinja')

        # Prepare context
        context = {
            'loop_var': loop.get("loop_var", "item"),
            'iterable_name': loop.get("iterable_name", "iterable"),
            'loop_body': loop.get("body", "")
        }

        # Render template
        return template.render(**context)

    def suggest_parallel_nested_loop(self, loop):
        """Suggests code for parallelizing a nested loop using Jinja2."""
        template = self.jinja_env.get_template('nested_for_loop.jinja')

        context = {
            'outer_var': loop.get("outer_var", "i"),
            'inner_var': loop.get("inner_var", "j"),
            'outer_iterable': loop.get("outer_iterable", "range(n)"),
            'inner_iterable': loop.get("inner_iterable", "range(m)"),
            'loop_body': loop.get("body", "")
        }

        return template.render(**context)

    def suggest_parallel_loop_with_functions(self, combo):
        """Suggests code for parallelizing a loop with function calls that contain loops using Jinja2."""
        template = self.jinja_env.get_template('for_with_functions.jinja')

        context = {
            'loop_var': combo.get("loop_var", "item"),
            'iterable_name': combo.get("iterable_name", "iterable"),
            'loop_body': combo.get("body", "")
        }

        return template.render(**context)

    def suggest_parallel_while(self, loop):
        """Suggests code for parallelizing a while loop using Jinja2."""
        template = self.jinja_env.get_template('while_loop.jinja')

        context = {
            'condition': loop.get("condition", "condition"),
            'loop_body': loop.get("body", ""),
            'data_generator': loop.get("data_generator", "items")
        }

        return template.render(**context)

    def suggest_parallel_listcomp(self, list_comp):
        """Suggests code for parallelizing a list comprehension using Jinja2."""
        template = self.jinja_env.get_template('listcomp.jinja')

        context = {
            'loop_var': list_comp.get("loop_var", "item"),
            'iterable': list_comp.get("iterable", "iterable"),
            'body': list_comp.get("body", "item")
        }

        return template.render(**context)

    def suggest_parallel_function(self, function):
        """Suggests code for parallelizing a function using Jinja2."""
        template = self.jinja_env.get_template('parallel_function.jinja')

        context = {
            'function_name': function.get("name", "function_name")
        }

        return template.render(**context)

    def suggest_parallel_recursive_function(self, function):
        """Suggests code for parallelizing a recursive function using Jinja2."""
        template = self.jinja_env.get_template('parallel_recursive_function.jinja')

        context = {
            'function_name': function.get("name", "function_name")
        }

        return template.render(**context)

    def suggest_parallel_function_call(self, function):
        """Suggests code for parallelizing a function call using Jinja2."""
        template = self.jinja_env.get_template('parallel_function_call.jinja')

        context = {
            'function_name': function.get("func_name", "function_name"),
            'parent_function_name': function.get("parent_func_name", "parent_function_name")
        }

        return template.render(**context)

    def suggest_parallel_combo_loop(self, combo):
        """Suggests code for parallelizing a combo of while and for loops using Jinja2."""
        template = self.jinja_env.get_template('parallel_combo_loop.jinja')

        context = {
            'inner_var': combo.get("inner_var", "item"),
            'outer_condition': combo.get("outer_condition", "condition"),
            'inner_iterable': combo.get("inner_iterable", "items"),
            'loop_body': combo.get("body", "")
        }

        return template.render(**context)

    def suggest_parallel_recursive_loop(self, combo):
        """Suggests code for parallelizing a loop with recursive function calls using Jinja2."""
        template = self.jinja_env.get_template('parallel_recursive_loop.jinja')

        context = {
            'loop_var': combo.get("loop_var", "item"),
            'iterable_name': combo.get("iterable_name", "iterable"),
            'loop_body': combo.get("body", "")
        }

        return template.render(**context)

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
