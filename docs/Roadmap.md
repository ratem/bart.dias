# ROADMAP

## Current Stable Version: 1.9.0

### 1. Block Based

### 1.1 Enhanced Dependency Analysis

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

- Expand the test suite to cover combinations **DONE**

### 1.4. Code Generation

- Use of flexible Code Templates **DONE**

### 2. Pattern Based

### 2.1 Improved Background
- Implementation of the Critical Path Analisys **DONE**
- Implementation of Amdahl's Law **DONE**
- Static Profiling: Work Calculation **DONE**


### 2.2 Patterns & HDW Resources
- Use of Patterns 
- Use of HPC Resources


### 2.3 Comprehensive Code Generation
- Include error handling
- Add proper cleanup of resources


### 3. Intelligence & Formalism: **V 2.0**

- Fluid dialogue with the user (LLM)
- Expanded code generation (LLM)
- Use of Formal Methods

