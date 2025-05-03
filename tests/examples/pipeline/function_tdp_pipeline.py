def process_data_stream(input_data):
    # Stage 1: Data cleaning
    cleaned = []
    for item in input_data:
        cleaned.append(item.strip().lower())

    # Stage 2: Feature extraction
    features = []
    for text in cleaned:
        features.append(len(text))

    # Stage 3: Result formatting
    results = []
    for length in features:
        results.append(f"Length: {length}")

    return results
