def image_processing_pipeline(images):
    # Stage 1: Convert to grayscale
    grayscale_images = []
    for img in images:
        grayscale_images.append(convert_to_grayscale(img))
    
    # Stage 2: Apply blur filter
    blurred_images = []
    for img in grayscale_images:
        blurred_images.append(apply_blur(img))
    
    # Stage 3: Detect edges
    edge_images = []
    for img in blurred_images:
        edge_images.append(detect_edges(img))
    
    return edge_images

