#!/usr/bin/env python3
"""word imbedding"""
from gensim.models import Word2Vec
from tensorflow.keras.layers import Embedding
import numpy as np

def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """creates and trains a gensim word2vec model
    sentences is a list of sentences to be trained on
    size is the dimensionality of the embedding layer
    min_count is the minimum number of occurrences of a word for use in
    training
    window is the maximum distance between the current and predicted word
    negative is the size of negative sampling
    cbow is a boolean to determine the training type; True is for CBOW
    iterations is the number of iterations to train over
    seed is the seed for the random number generator
    workers is the number of worker threads to train the model"""
    if cbow is True:
        cbow = 0
    else:
        cbow = 1
    return Word2Vec(sentences=sentences,
                    vector_size=size,
                    min_count=min_count,
                    window=window,
                    negative=negative,
                    sg=cbow,
                    epochs=iterations,
                    seed=seed,
                    workers=workers)


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a Keras Embedding layer
    
    Args:
    model: A trained gensim word2vec model
    
    Returns:
    trainable Keras Embedding layer
    """
    vocab_size = len(model.wv.key_to_index)
    embedding_dim = model.vector_size
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i in range(vocab_size):
        embedding_vector = model.wv[model.wv.index_to_key[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=True)
    
    return embedding_layer

# Train the Word2Vec model
model = word2vec_model(["the quick brown fox jumps over the lazy dog"])

# Convert the Gensim Word2Vec model to a Keras Embedding layer
keras_embedding = gensim_to_keras(model)

# Testing the conversion
print(keras_embedding)
    