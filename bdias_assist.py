from bdias_code_gen import BDiasCodeGen


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

      code = self.read_code(code_or_path) #Try to read from file, or the input is code
      if code is None:
        print("No code was analyzed, check your file or code")
        return True
      structured_code = self.parser.parse(code)
      if structured_code is None:
        print("No code was analyzed, check for syntax errors")
        return True
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

    def display_opportunities(self, structured_code, code):
        """Presents parallelization opportunities to the user."""
        has_opportunities = any(structured_code[key] for key in structured_code)
        if not has_opportunities:
            print("No parallelization opportunities identified in the given code.")
            return

        print("Potential Parallelization Opportunities:")
        suggestions = self.code_generator.generate_suggestions(structured_code)

        for suggestion in suggestions:
            try:
                if suggestion["opportunity_type"] == 'loop' and suggestion.get("iterable_name"):  # Correct condition for loops with iterable_name
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]].format(**suggestion)}')
                elif suggestion["opportunity_type"] == 'nested loop':
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]].format(**suggestion)}')
                elif suggestion["opportunity_type"] == 'while':  # Correct condition for while loops
                    print(f'  - Line {suggestion["lineno"]}: This `while` loop may be parallelizable.')  # Specific message for while loops, no formatting
                elif suggestion["opportunity_type"] == 'loop and function':
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]].format(**suggestion)}')
                elif suggestion["opportunity_type"] == 'recursive function definition':
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]].format(**suggestion)}')
                elif suggestion["opportunity_type"] == 'recursive function':
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]].format(**suggestion)}')
                elif suggestion["opportunity_type"] == 'function call':
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]].format(**suggestion)}')
                elif suggestion["opportunity_type"] == 'function':
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]].format(**suggestion)}')
                elif suggestion["opportunity_type"] == 'list_comprehension':
                    print(f'  - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[suggestion["explanation_index"]]}')
                else:  # Default case if the opportunity type is unknown
                    print(f"  - Line {suggestion['lineno']}: Default explanation for unknown opportunity type")  # Fallback message

            except KeyError as e:  # Handle any remaining KeyErrors
                print(f"KeyError in display_opportunities: Missing key '{e}' in suggestion: {suggestion}, explanation_index: {suggestion.get('explanation_index', 'N/A')}")
                print(f'  - Line {suggestion["lineno"]}: Default explanation for {suggestion.get("opportunity_type", "unknown opportunity type")}')

            try: #Try to print partitioning suggestions
                print(f'    Partitioning suggestion: {self.code_generator.PARTITIONING_SUGGESTIONS[suggestion["explanation_index"]]}')
            except KeyError as e:
                 print(f"KeyError in display_opportunities: Missing partitioning suggestion for explanation index '{e}'")

            print(f'    ---Original code:\n    {code.splitlines()[suggestion["lineno"]-1]}')
            print('    ---Code suggestion:')
            print(f'   {suggestion["code_suggestion"]}')
            if suggestion.get("llm_suggestion"):
                print(suggestion["llm_suggestion"])
            print("---")  # Separator between suggestions

        print("---")  # Final separator

    def run_interactive_session(self):
       """Runs the interactive session, prompting the user and processing code."""
       print("Welcome to Bart.dIAs! I will analyze your Python code to find parallelization opportunities.")
       while True:
         code = self.get_user_input()
         if not self.process_code(code):
           break
       print("Exiting Bart.dIAs. Goodbye!")