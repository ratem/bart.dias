# Bart.dIAs - Parallel Programming Assistant

Bart.dIAs is a Python assistant designed to analyze code and identify opportunities for parallelization. It performs static analysis to detect parallelizable patterns and provides concrete suggestions for implementing parallel solutions using Python's multiprocessing module.

## Features

### Block-Based Analysis

- **Pattern Detection**: Identifies parallelizable patterns in Python code including loops, functions, nested structures, and list comprehensions
- **Dependency Analysis**: Performs sophisticated data flow analysis to determine which code blocks can be safely parallelized
- **Side-by-Side Comparison**: Shows original code alongside parallelized versions for easy implementation
- **Code Generation**: Creates properly indented, template-based code suggestions using Jinja2


### Critical Path Analysis

- **DAG-Based Modeling**: Constructs a Directed Acyclic Graph (DAG) representation of code to analyze dependencies
- **Work-Span Metrics**: Calculates theoretical metrics from parallel computing theory:
    - Total Work (T₁): Sum of computational costs across all operations
    - Critical Path Length (T∞): Longest chain of dependent operations
    - Inherent Parallelism (T₁/T∞): Theoretical upper bound on speedup
- **Amdahl's Law Integration**: Estimates maximum theoretical speedup based on sequential fraction
- **Bottleneck Identification**: Pinpoints sequential code sections that limit parallelization potential


### Combo Pattern Detection

- **Complex Pattern Recognition**: Detects and analyzes advanced patterns:
    - While loops containing for loops
    - For loops with recursive function calls
    - Nested loops with varying depths
    - Loops containing function calls that themselves contain loops
    - For loops inside while loops


## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/ratem/bart.dias.git
cd bart.dias
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```


## Requirements

- Python 3.6+
- Jinja2
- NetworkX (for DAG visualization)
- Matplotlib (optional, for visualization)


## Usage

Run the main script to start an interactive session:

```bash
python main.py
```


### Interactive Session

1. **Enter your code**: Either paste Python code directly or provide a path to a Python file
2. **Choose analysis type**:
    - **Block-based opportunities**: Identifies specific code blocks that can be parallelized
    - **Critical Path Analysis**: Performs theoretical analysis of inherent parallelism

### Example Output (Block-Based Analysis)

```
Potential Parallelization Opportunities:
 - Line 25: This 'for' loop iterates over a range of numbers using the loop variable 'i'
 Partitioning suggestion: Consider data partitioning by dividing the range into chunks for each process

 Side-by-Side Comparison:
 Original Code                           | Parallelized Version
 --------------------------------------- | ----------------------------------------
 for i in range(1000):                   | import multiprocessing
     result.append(i * i + 3 * i - 2)    | 
                                         | def process_item(i):
                                         |     result = []
                                         |     result.append(i * i + 3 * i - 2)
                                         |     return result
                                         | 
                                         | if __name__ == '__main__':
                                         |     with multiprocessing.Pool() as pool:
                                         |         results = pool.map(process_item, range(1000))
```


### Example Output (Critical Path Analysis)

```
=== Critical Path Analysis ===

Total Work (T₁): 1052.50
Critical Path Length (T∞): 78.50
Theoretical Parallelism (T₁/T∞): 13.41x
Amdahl's Law - Sequential Fraction: 7.46%
Amdahl's Law - Max Speedup: 13.41x

Top Bottlenecks:
1. For Loop (Line 42): Work 320.00, Span 40.00
   Code: for i in range(n): result = result + i
2. While Loop (Line 78): Work 210.00, Span 30.00
   Code: while i &lt; n: result += fibonacci(j % 5)
3. Function (Line 15): Work 180.00, Span 8.50
   Code: def recursive_fibonacci(n): if n &lt;= 1: return n

Recommendations:
- The sequential fraction is significant. Focus on parallelizing the bottlenecks identified above.
- The critical path contains high-intensity sequential sections. Consider:
  1. Breaking down these sections into smaller, independent tasks
  2. Using algorithmic transformations to reduce dependencies
  3. Applying domain-specific optimizations to these bottlenecks
```


## Architecture

Bart.dIAs consists of several key components:

1. **BDiasParser**: Parses Python code using AST analysis to identify parallelizable patterns and analyze dependencies
2. **BDiasCodeGen**: Generates parallelization suggestions using Jinja2 templates
3. **BDiasCriticalPathAnalyzer**: Performs theoretical analysis of code's inherent parallelism
4. **BDiasAssist**: Provides the user interface and coordinates the analysis process

## Theoretical Foundation

The critical path analysis is based on established parallel computing theory from Jesper Larsson Träff's "Lectures on Parallel Computing":

- **Work-Span Model**: Represents computation as a DAG where nodes are tasks and edges are dependencies
- **Critical Path Analysis**: Identifies the longest path of dependent operations that limits parallel execution
- **Amdahl's Law**: Calculates maximum theoretical speedup based on sequential fraction
- **Brent's Theorem**: Relates work, span, and processor count


## Limitations

- **Static Analysis Only**: Analysis is based on code structure, not runtime behavior
- **Focus on Multiprocessing**: Current code generation targets Python's multiprocessing module
- **Python-Specific**: Analysis is designed for Python code only
- **Theoretical Bounds**: Critical path analysis provides theoretical upper bounds that may not be achievable in practice


## Roadmap

See the [Roadmap](Roadmap.md) file for planned features and improvements.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Jesper Larsson Träff for the theoretical foundation in "Lectures on Parallel Computing"

