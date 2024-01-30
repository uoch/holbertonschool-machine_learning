#!/usr/bin/python3
"""semantic search"""
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def semantic_search(corpus_path, sentence):
    """semantic search"""
    corpus = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, 'r') as md_file:
                corpus.append(md_file.read() + '\n')

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_sen = model.encode(sentence)

    similarities = [cosine_similarity(
        [embeddings_sen], [model.encode(doc)]) for doc in corpus]
    most_similar_idx = np.argmax(similarities)
    most_similar_doc = corpus[most_similar_idx]
    return most_similar_doc
