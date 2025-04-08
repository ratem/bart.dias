from bdias_profiler import BDiasProfiler


class BDiasAssist:
    """
    Provides user interaction and orchestrates the code analysis and parallelization suggestions.
    """

    def __init__(self, parser, code_generator):
        """Initializes the assistant with a parser and code generator."""
        self.parser = parser
        self.code_generator = code_generator

    def get_user_input(self):
       """Gets user input from the console."""
       return input("Enter your Python code or a file path, or type 'exit' to quit: ")

    def process_code(self, code_or_path):
        """Parses code or a file content and presents results."""
        if code_or_path.lower() == 'exit':
            return False  # Signal to exit the interactive session

        code = self.read_code(code_or_path)  # Try to read from file, or the input is code
        if code is None:
            print("No code was analyzed, check your file or code")
            return True

        # Parse the code first (needed for both profiling and normal analysis)
        structured_code = self.parser.parse(code)
        if structured_code is None:
            print("No code was analyzed, check for syntax errors")
            return True

        profile_first = input(
            "Would you like to profile the code to identify computationally intensive sections? (y/n): ").lower() == 'y'

        if profile_first:
            profiler = BDiasProfiler(max_results=5)
            try:
                # Use the parser that already has the AST
                ranked_blocks = profiler.profile_code(self.parser, code)

                if not ranked_blocks:
                    print(
                        "No significant computationally intensive sections were identified. Proceeding with standard analysis.")
                    self.display_opportunities(structured_code, code)
                else:
                    # Let user select a block to optimize
                    code_lines = code.splitlines()
                    selected_block = profiler.get_user_selection(ranked_blocks, code_lines)

                    print(
                        f"\nAnalyzing {selected_block['type']}: {selected_block['name']} (Lines {selected_block['lineno']}-{selected_block['end_lineno']})")

                    # Filter opportunities to focus on the selected block
                    self.display_opportunities(structured_code, code, selected_block['lineno'],
                                               selected_block['end_lineno'])
            except Exception as e:
                print(f"Error during profiling: {e}")
                print("Proceeding with standard analysis.")
                self.display_opportunities(structured_code, code)
        else:
            self.display_opportunities(structured_code, code)

        return True

    def read_code(self, code_or_path):
       """Attempts to read code from a file, otherwise return the string."""
       try:
           with open(code_or_path, 'r') as f:
               return f.read()
       except FileNotFoundError:
            return code_or_path
       except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def display_opportunities(self, structured_code, code, start_line=None, end_line=None):
        """Presents parallelization opportunities to the user."""
        has_opportunities = any(structured_code[key] for key in structured_code)

        if not has_opportunities:
            print("No parallelization opportunities identified in the given code.")
            return

        print("Potential Parallelization Opportunities:")

        suggestions = self.code_generator.generate_suggestions(structured_code)

        # Filter suggestions if a specific block is selected
        if start_line is not None and end_line is not None:
            filtered_suggestions = []
            for suggestion in suggestions:
                if start_line <= suggestion.get("lineno", 0) <= end_line:
                    filtered_suggestions.append(suggestion)
            suggestions = filtered_suggestions

        for suggestion in suggestions:
            try:
                opportunity_type = suggestion.get("opportunity_type", "unknown")
                explanation_index = suggestion.get("explanation_index", "")

                # Basic loop patterns
                if opportunity_type in ['loop', 'nested loop']:
                    print(
                        f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index].format(**suggestion)}')

                # While loops
                elif opportunity_type == 'while':
                    print(f' - Line {suggestion["lineno"]}: This `while` loop may be parallelizable.')

                # Function patterns
                elif opportunity_type in ['function', 'recursive function definition', 'recursive function',
                                          'function call']:
                    print(
                        f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index].format(**suggestion)}')

                # List comprehension
                elif opportunity_type == 'list_comprehension':
                    print(f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index]}')

                # Loop and function combination
                elif opportunity_type == 'loop and function':
                    print(
                        f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index].format(**suggestion)}')

                # Combo patterns
                elif opportunity_type in ['for_with_recursive_call', 'while_with_for', 'for_in_while',
                                          'for_with_loop_functions', 'while_with_loop_functions']:
                    print(
                        f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index].format(**suggestion)}')

                else:
                    print(f" - Line {suggestion['lineno']}: Default explanation for {opportunity_type}")

                # Print partitioning suggestion
                if explanation_index in self.code_generator.PARTITIONING_SUGGESTIONS:
                    print(
                        f' Partitioning suggestion: {self.code_generator.PARTITIONING_SUGGESTIONS[explanation_index]}')
                else:
                    print(
                        f' Partitioning suggestion: Consider data or task partitioning based on the specific pattern.')

                # Print original code and suggestion
                print(f' ---Original code:\n {code.splitlines()[suggestion["lineno"] - 1]}')
                print(' ---Code suggestion:')
                print(f' {suggestion["code_suggestion"]}')

                if suggestion.get("llm_suggestion"):
                    print(suggestion["llm_suggestion"])

                print("---")  # Separator between suggestions

            except KeyError as e:
                print(f"KeyError in display_opportunities: Missing key '{e}' in suggestion: {suggestion}")
                print(
                    f' - Line {suggestion.get("lineno", "unknown")}: Default explanation for {suggestion.get("opportunity_type", "unknown opportunity type")}')
                print("---")

        print("---")  # Final separator
        print("End of suggestions.")

    def run_interactive_session(self):
       """Runs the interactive session, prompting the user and processing code."""
       print("Welcome to Bart.dIAs! I will analyze your Python code to find parallelization opportunities.")
       while True:
         code = self.get_user_input()
         if not self.process_code(code):
           break
       print("Exiting Bart.dIAs. Goodbye!")