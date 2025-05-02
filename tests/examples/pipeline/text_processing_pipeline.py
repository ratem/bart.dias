def text_processing_pipeline(documents):
    # Stage 1: Tokenize
    tokenized_docs = []
    for doc in documents:
        tokenized_docs.append(tokenize(doc))
    
    # Stage 2: Remove stop words
    filtered_docs = []
    for tokens in tokenized_docs:
        filtered_docs.append(remove_stopwords(tokens))
    
    # Stage 3: Stem words
    stemmed_docs = []
    for tokens in filtered_docs:
        stemmed_docs.append(stem_words(tokens))
    
    # Stage 4: Create bag of words
    bow_docs = []
    for tokens in stemmed_docs:
        bow_docs.append(create_bow(tokens))
    
    return bow_docs

