# ROADMAP

## Current Stable Version: 1.6.0

### 1. Enhanced Dependency Analysis

- Track variable dependencies across function boundaries **DONE**
- Identify data dependencies in more complex scenarios **DONE**
- Better analyze side effects in function calls **DONE**
- Implement a proper data flow analysis to detect read-after-write dependencies **DONE**


### 2. Combo Detection

- While loops containing for loops **DONE**
- For loops with recursive function calls **DONE**
- Nested loops with varying depths **DONE**
- Loops containing function calls that themselves contain loops **DONE**


### 3. Testing and Validation

- Expand the test suite to cover all possible combinations **DONE**


### 4. User Experience Improvements

- Allow users to select which suggestions to implement **PARTIAL**
- Provide explanations of why certain code isn't parallelizable **SUSPENDED**
- Static Profiling **PROTOTYPE**


### 5. Code Generation Enhancements 

- Include error handling in generated code
- Add proper cleanup of resources in generated code
- Add support for shared memory in generated code when appropriate


### 6. Improved Documentation

- Add more comprehensive documentation for each module
- Document limitations and best practices **PROTOTYPE**


### 7. Intelligence **2.0**

- Fluid dialogue with the user
- Generate parallel code based on Patterns 
- Include performance estimates for parallelized vs. sequential code 
