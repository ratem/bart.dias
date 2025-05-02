def etl_pipeline(data_sources):
    # Stage 1: Extract data
    extracted_data = []
    for source in data_sources:
        extracted_data.append(extract_data(source))
    
    # Stage 2: Transform data
    transformed_data = []
    for data in extracted_data:
        transformed_data.append(transform_data(data))
    
    # Stage 3: Load data
    results = []
    for data in transformed_data:
        results.append(load_data(data))
    
    return results

