#!/usr/bin/env python3
"""word imbedding"""
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.layers import Embedding


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
