def three_stage_pipeline(data):
    # Stage 1: Generate data
    buffer1 = []
    for item in data:
        buffer1.append(item * 2)
    
    # Stage 2: Transform data
    buffer2 = []
    for item in buffer1:
        buffer2.append(item + 10)
    
    # Stage 3: Process data
    results = []
    for item in buffer2:
        results.append(item * 3)
    
    return results

if __name__ == "__main__":
    input_data_sample = [1, 2, 3, 4, 5]
    result = three_stage_pipeline(input_data_sample)
    print(result)