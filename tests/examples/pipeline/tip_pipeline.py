def image_processing_pipeline(images):
    # Stage 1: Resize
    resized = []
    for img in images:
        resized.append(resize_image(img, (224, 224)))

    # Stage 2: Filter
    filtered = []
    for img in resized:
        filtered.append(apply_gaussian_filter(img))

    # Stage 3: Feature extract
    features = []
    for img in filtered:
        features.append(extract_cnn_features(img))

    return features
