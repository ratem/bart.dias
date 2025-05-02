def audio_processing_pipeline(audio_files):
    # Stage 1: Load audio
    audio_data = []
    for file in audio_files:
        audio_data.append(load_audio(file))
    
    # Stage 2: Normalize amplitude
    normalized_audio = []
    for audio in audio_data:
        normalized_audio.append(normalize(audio))
    
    # Stage 3: Apply filters
    filtered_audio = []
    for audio in normalized_audio:
        filtered_audio.append(apply_filters(audio))
    
    # Stage 4: Encode
    encoded_audio = []
    for audio in filtered_audio:
        encoded_audio.append(encode(audio))
    
    return encoded_audio

