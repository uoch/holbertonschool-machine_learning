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
        vocab = sorted(set(clean_sentence(sentences)))
    
    embeddings = np.zeros((len(sentences), len(vocab)))
    cleaned_sentences = [clean_sentence([sentence]) for sentence in sentences]

    for i, cleaned_sentence in enumerate(cleaned_sentences):
        for word in cleaned_sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1
    return embeddings, vocab
