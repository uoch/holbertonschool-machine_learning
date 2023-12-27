#!/usr/bin/env python3
"""word embedding"""
import numpy as np


def clean_string(word):
    """cleans a string"""
    word = word.lower()
    word = word.strip()
    word = word.strip('.')
    word = word.strip('!')
    word = word.strip('?')
    word = word.strip(',')
    word = word.strip(':')
    word = word.strip(';')
    return word


def bag_of_words(sentences, vocab=None):
    """transforms sentences to a bag of words embedding matrix"""
    if vocab is None:
        vocab = set(clean_string(word)
                    for sent in sentences for word in sent.split())

    features = list(vocab)
    s = len(sentences)
    f = len(features)

    # Initialize the embeddings matrix
    embeddings = np.zeros((s, f), dtype=int)

    # Loop through sentences and words to fill the embeddings matrix
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, word in enumerate(words):
            words[j] = clean_string(word)
            if words[j] in vocab:
                word_idx = features.index(words[j])
                embeddings[i][word_idx] += 1

    return embeddings, sorted(features)
