#!/usr/bin/env python3
"""0. Bag Of Words"""
import numpy as np
import re


def clean_sentence(sentences):
    """Clean sentence before embedding"""
    words = []
    for sentence in sentences:
        words.extend(re.sub(r"\b\w{1}\b", "", re.sub(
            r"[^a-zA-Z0-9\s]", " ", sentence.lower())).split())
    return words


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix"""
    if vocab is None:
        vocab = []
    sentence_words = clean_sentence(sentences)
    sentence_words = list(set(sentence_words))
    vocab.extend(sentence_words)
    vocab = sorted(vocab)
    words = {word: i for i, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)))
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            if word in words:
                embeddings[i][words[word]] += 1
    return embeddings.astype(int), vocab
