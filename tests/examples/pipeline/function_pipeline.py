def process_data_stream(input_data):
    # Clearly separated stages
    stage1 = [x*2 for x in input_data]
    stage2 = [x+10 for x in stage1]
    return [x*3 for x in stage2]
