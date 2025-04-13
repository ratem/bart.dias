"""
BDiasAssist: User Interaction Module for Bart.dIAs

This module provides the user interface for the Bart.dIAs system. It handles
user interactions, code input, and presentation of parallelization suggestions.

Features:
- Runs an interactive session for code analysis
- Processes user input (code or file paths)
- Integrates with BDiasParser and BDiasCodeGen for code analysis and suggestion generation
- Offers options for viewing all parallelization opportunities or focusing on computationally intensive sections
- Displays parallelization suggestions with side-by-side code comparisons
- Handles static profiling to identify computationally intensive code sections

Classes:
- BDiasAssist: Main class for user interaction and result presentation

Dependencies:
- BDiasProfiler
"""


from bdias_profiler import BDiasProfiler


class BDiasAssist:
    def __init__(self, parser, code_generator):
        self.parser = parser
        self.code_generator = code_generator
        self.profiler = BDiasProfiler(max_results=5)

    def process_code(self, code_or_path):
        """Parses code or a file content and presents results."""
        if code_or_path.lower() == 'exit':
            return False  # Signal to exit the interactive session

        code = self.read_code(code_or_path)
        if code is None:
            print("No code was analyzed, check your file or code")
            return True

        # Parse the code first (needed for both profiling and normal analysis)
        structured_code = self.parser.parse(code)
        if structured_code is None:
            print("No code was analyzed, check for syntax errors")
            return True

        # Ask user if they want to see all opportunities or just the top N most intensive ones
        profile_option = input("How would you like to view parallelization opportunities?\n"
                               "1. Show all opportunities\n"
                               "2. Show only the most computationally intensive sections\n"
                               "Enter your choice (1/2): ")

        if profile_option == "2":
            try:
                # Display profiling limitations message
                print("\nNOTE: The profiling is based on a heuristic which provides a rough approximation")
                print("of computational intensity. For more accurate results, a more sophisticated")
                print("data flow analysis and dynamic runtime profiling would be needed.")
                print("The current static analysis cannot account for data-dependent performance characteristics.")

                # Use the parser that already has the AST
                ranked_blocks = self.profiler.profile_code(self.parser, code)

                if not ranked_blocks:
                    print(
                        "No significant computationally intensive sections were identified. Proceeding with standard analysis.")
                    self.display_opportunities(structured_code, code)
                else:
                    # Let user select a block to optimize
                    code_lines = code.splitlines()
                    selected_block = self.get_user_selection(ranked_blocks, code_lines)

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
            # Standard analysis showing all opportunities
            self.display_opportunities(structured_code, code)

        return True

    def read_code(self, code_or_path):
        """Reads code from a file or returns the input if it's a code snippet."""
        try:
            with open(code_or_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return code_or_path
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def get_user_selection(self, ranked_blocks, code_lines):
        """
        Present the ranked code blocks to the user and get their selection.
        """
        print("\nTop computationally intensive sections in your code (based on static analysis):")
        for i, block in enumerate(ranked_blocks):
            block_type = block["type"].replace("_", " ").title()
            block_name = block["name"]
            start_line = block["lineno"]
            end_line = block["end_lineno"]

            # Show code snippet
            code_snippet = "\n".join(code_lines[start_line - 1:min(end_line, len(code_lines))])

            print(f"{i + 1}. {block_type}: {block_name} (Lines {start_line}-{end_line})")
            print(f"   Estimated computational intensity: {block['intensity']:.2f}")
            print(f"   Code snippet:")
            for line in code_snippet.splitlines()[:3]:  # Show first 3 lines
                print(f"      {line}")
            if len(code_snippet.splitlines()) > 3:
                print("      ...")

        while True:
            try:
                selection = int(input(f"\nSelect a section to optimize (1-{len(ranked_blocks)}): "))
                if 1 <= selection <= len(ranked_blocks):
                    return ranked_blocks[selection - 1]
                else:
                    print(f"Please enter a number between 1 and {len(ranked_blocks)}")
            except ValueError:
                print("Please enter a valid number")

    def display_opportunities(self, structured_code, code, start_line=None, end_line=None):
        """Presents parallelization opportunities to the user with side-by-side code comparison."""
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

            if not suggestions:
                print(
                    f"No parallelization opportunities found in the selected code block (Lines {start_line}-{end_line}).")
                return

        for suggestion in suggestions:
            try:
                opportunity_type = suggestion.get("opportunity_type", "unknown")
                explanation_index = suggestion.get("explanation_index", "")

                # Print opportunity type and explanation
                if opportunity_type in ['loop', 'nested loop']:
                    print(
                        f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index].format(**suggestion)}')
                elif opportunity_type == 'while':
                    print(f' - Line {suggestion["lineno"]}: This `while` loop may be parallelizable.')
                elif opportunity_type in ['function', 'recursive function definition', 'recursive function',
                                          'function call']:
                    print(
                        f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index].format(**suggestion)}')
                elif opportunity_type == 'list_comprehension':
                    print(f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index]}')
                elif opportunity_type == 'loop and function':
                    print(
                        f' - Line {suggestion["lineno"]}: {self.code_generator.EXPLANATIONS[explanation_index].format(**suggestion)}')
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

                # Get original code lines
                original_code_lines = code.splitlines()
                lineno = suggestion["lineno"]

                # Get the original code block (try to extract a reasonable context)
                original_block = []
                start_idx = max(0, lineno - 1)  # Start from the identified line

                # Try to get a few lines of context
                context_lines = 3
                for i in range(start_idx, min(start_idx + context_lines, len(original_code_lines))):
                    original_block.append(original_code_lines[i])

                # Get the suggested code
                suggested_code = suggestion["code_suggestion"].splitlines()

                # Determine the maximum width needed for the original code
                max_original_width = max(len(line) for line in original_block)
                # Ensure minimum width for better readability
                max_original_width = max(max_original_width, 40)

                # Print side-by-side comparison
                print("\n Side-by-Side Comparison:")
                print(f" {'Original Code':<{max_original_width}} | {'Parallelized Version'}")
                print(f" {'-' * max_original_width} | {'-' * 40}")

                # Print the code side by side
                for i in range(max(len(original_block), len(suggested_code))):
                    original_line = original_block[i] if i < len(original_block) else ""
                    suggested_line = suggested_code[i] if i < len(suggested_code) else ""
                    print(f" {original_line:<{max_original_width}} | {suggested_line}")

                if suggestion.get("llm_suggestion"):
                    print("\nAdditional suggestions:")
                    print(suggestion["llm_suggestion"])

                print("\n---")  # Separator between suggestions

            except KeyError as e:
                print(f"KeyError in display_opportunities: Missing key '{e}' in suggestion: {suggestion}")
                print(
                    f' - Line {suggestion.get("lineno", "unknown")}: Default explanation for {suggestion.get("opportunity_type", "unknown opportunity type")}')
                print("---")

        print("---")  # Final separator
        print("End of suggestions.")

    def run_interactive_session(self):
        """Runs an interactive session for code analysis."""
        print("Welcome to Bart.dIAs! I will analyze your Python code to find parallelization opportunities.")
        while True:
            code = input("Enter your Python code or a file path, or type 'exit' to quit: ")
            if not self.process_code(code):
                break
        print("Exiting Bart.dIAs. Goodbye!")
