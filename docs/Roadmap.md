# ROADMAP

## Current Stable Version: 0.9.0

### 1. Block Based Functionality

### 1.1 Dependency Analysis

- Track variable dependencies across function boundaries **DONE**
- Identify data dependencies in more complex scenarios **DONE**
- Better analyze side effects in function calls **DONE**
- Implement a proper data flow analysis to detect read-after-write dependencies **DONE**


### 1.2. Combo Detection

- While loops containing for loops **DONE**
- For loops with recursive function calls **DONE**
- Nested loops with varying depths **DONE**
- Loops containing function calls that themselves contain loops **DONE**


### 1.3. Testing and Validation

- Test suite to cover all block-based patterns **DONE**

### 1.4. Code Generation

- Use of Code Templates **DONE**

### 2. Pattern Based Functionality

### 2.1 Static Profiling & Code Generation
- Implementation of Amdahl's Law **DONE**
- Implementation of the Critical Path Analisys **DONE**
- Heuristics for Domain Partition Patterns **DONE**
- Heuristics for Parallel Programming Patterns **DONE**
- Templates for Code Generation
- Decision Tree for Domain Partition
- Decision Tree for Programming Patterns

### 2.2 Platform Resources
- Use of HPC Resources
- Dynamic Profiling:
	- Code generation 
	- Output analysis


### 2.3 Comprehensive Code Generation & Testing
- Expand Real-World Scenarios testing
- Include error handling
- Add proper cleanup of resources


### 3. Intelligence & Formalism: **V 2.0**

- Fluid dialogue with the user (LLM)
- Expanded code generation (LLM)
- Comprehensive use of Formal Methods

