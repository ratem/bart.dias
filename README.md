# Bart.dIAs - Parallel Coding Assistant

[![GitHub](https://img.shields.io/badge/GitHub-ratem/bart.dias-blue?logo=github)](https://github.com/ratem/bart.dias)

Bart.dIAs is a Python assistant designed to analyze Python code and identify opportunities for parallelization using the `multiprocessing` module. 
It performs static analysis and provides code suggestions to help developers improve the performance of their applications.

## Features

*   **AST-Based Parsing:** Analyzes Python code using Abstract Syntax Trees (AST).
*   **Pattern Recognition:** Identifies various parallelizable patterns, including basic loops (`for`, `while`), nested loops, functions (regular and recursive), list comprehensions, and more complex "combo" patterns.
*   **Dependency Analysis:** Performs static dependency checks (variable dependencies, read-after-write, loop-carried dependencies, cross-function dependencies) to assess the safety of parallelization.
*   **Static Profiling:** Offers heuristic-based static profiling to estimate the computational intensity of code blocks (functions, loops) and ranks them, helping focus optimization efforts.
*   **Code Generation:** Generates parallel code suggestions using Python's `multiprocessing` module. Uses Jinja2 templates for properly structured and indented code.
*   **Side-by-Side Comparison:** Displays the original code alongside the suggested parallelized version for easy comparison.
*   **Interactive Session:** Provides a command-line interface to analyze code snippets or entire files.

## Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/ratem/bart.dias.git
    cd bart.dias
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 installed. Then, install the required libraries using pip:
    ```
    pip install -r requirements.txt
    ```

## Usage

Run the main script from the project's root directory:

```

python main.py

```

Bart.dIAs will start an interactive session:

```

Welcome to Bart.dIAs! I will analyze your Python code to find parallelization opportunities.
Enter your Python code or a file path, or type 'exit' to quit:

```

You can either:
*   Paste a Python code snippet directly.
*   Enter the path to a Python file (e.g., `teste.py`).
*   Type `exit` to quit the session.

The assistant will then ask how you want to view opportunities:

```

How would you like to view parallelization opportunities?

1. Show all opportunities
2. Show only the most computationally intensive sections
Enter your choice (1/2):
```

Based on your choice, it will present potential parallelization suggestions, explanations, partitioning ideas, and the side-by-side code comparison.

## Dependencies

*   **Python 3.x**
*   **Jinja2:** Used for template-based code generation.
*   **NumPy:** Used in some test cases (`teste.py`), and potentially useful for certain numerical code analyses in the future (though not strictly required by the core analysis modules currently).

See `requirements.txt` for details.

## Limitations

*   **Static Analysis Only:** Bart.dIAs performs *static* analysis. It does not execute the code, so its understanding is based solely on the code structure.
*   **Heuristic Profiling:** The static profiler provides a rough *heuristic* estimate of computational intensity. It cannot accurately predict actual runtime performance, which can be data-dependent or affected by external factors (I/O, system load).
*   **Dependency Analysis Limitations:** While enhanced, static dependency analysis cannot capture all possible runtime dependencies or side effects, especially those involving external libraries or complex object interactions.
*   **Focus on `multiprocessing`:** Suggestions are currently focused on Python's `multiprocessing` module.

**Important:** Always carefully review and test the suggested parallel code. Parallelization can introduce complexity (e.g., race conditions, deadlocks) if not done correctly. Bart.dIAs is a tool to *assist* in identifying opportunities, not a fully automatic parallelizer.

## Future Work (Ideas for v2.0)

*   Implement high-level parallel pattern detection (Map-Reduce, Pipeline, Stencil, etc.).
*   Integrate dynamic analysis (profiling actual execution) for more accurate performance insights.
*   Explore integration with LLMs for more nuanced suggestions or explanations (currently placeholders).
*   Support for other parallelization libraries (e.g., `concurrent.futures`, `Dask`, `Ray`).

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests on the [GitHub repository](https://github.com/ratem/bart.dias).
```

