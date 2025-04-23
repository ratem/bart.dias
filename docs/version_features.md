## Version 0.9.1a (April 23, 2025)

### Major Features
- **Map-Reduce Pattern Implementation**: Added complete support for detecting and generating parallelized code for the Map-Reduce pattern
- **AST-Based Code Transformation**: Implemented robust code analysis and transformation using Python's ast module
- **Template-Based Code Generation**: Created Jinja2 templates for different partitioning strategies (SDP, SIP)
- **Hardware-Aware Code Generation**: Generated code now adapts to the actual number of processors available on the system

### Improvements
- **Enhanced Pattern Detection**: Improved detection of Map-Reduce patterns in both functions and loops
- **Partitioning Strategy Recommendations**: Added detailed rationales for partitioning strategy suggestions
- **Theoretical Performance Metrics**: Updated performance characteristics for Map-Reduce pattern based on Träff's lectures
- **Code Generation Error Handling**: Added robust error handling for code generation process

### Technical Details
- **New Modules**:
  - `bdias_pattern_codegen.py`: Implements pattern-specific code generators
  - `bdias_pattern_presenter.py`: Handles presentation of code transformations
- **New Templates**:
  - Function-specific templates for Map-Reduce pattern
  - Loop-specific templates for Map-Reduce pattern
  - Templates for different partitioning strategies (SDP, SIP)
- **Integration**: Integrated pattern detection with code generation in the BDiasAssist module

## Version 0.9.0 (March 15, 2025)

### Major Features
- **Critical Path Analysis**: Implemented DAG-based critical path analysis
- **Bottleneck Identification**: Added algorithms to identify performance bottlenecks
- **Basic Pattern Recognition**: Implemented initial pattern recognition for common parallel patterns
- **Theoretical Metrics**: Added calculation of work, span, and parallelism metrics

### Improvements
- **Code Structure Analysis**: Enhanced parsing and analysis of Python code structures
- **Performance Optimization**: Improved algorithm efficiency for large codebases
- **User Interface**: Developed command-line interface for easy access to analysis tools

### Technical Details
- **Core Modules**:
  - `bdias_parser.py`: Parses Python code into structured representation
  - `bdias_critical_path.py`: Implements critical path analysis algorithms
  - `bdias_pattern_analyzer.py`: Detects parallel patterns in code
  - `bdias_assist.py`: Main interface for the assistant

