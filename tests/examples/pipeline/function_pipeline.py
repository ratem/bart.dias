def process_data_stream(input_data):
    # Clearly separated stages
    stage1 = [x*2 for x in input_data]
    stage2 = [x+10 for x in stage1]
    return [x*3 for x in stage2]


if __name__ == "__main__":
    input_data_sample = [1, 2, 3, 4, 5]
    result = process_data_stream(input_data_sample)
    print(result)