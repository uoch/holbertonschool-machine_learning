#!/usr/bin/env python3
"""word embedding"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """transforms sentences to a bag of words embedding matrix"""
    if vocab is None:
        vocab = set(word for sentence in sentences for word in sentence.split())

    features = list(vocab)
    s = len(sentences)
    f = len(features)

    embeddings = np.zeros((s, f))

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, word in enumerate(features):
            embeddings[i][j] = words.count(word)

    return embeddings, sorted(features)
