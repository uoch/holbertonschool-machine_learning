#!/usr/bin/env python3
"""word imbedding"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5,
                   negative=5, window=5,
                   cbow=True,
                   iterations=5,
                   seed=0,
                   workers=1):
    """train a fasttext model"""
    if cbow is True:
        cbow = 0
    else:
        cbow = 1
    return FastText(sentences=sentences,
                    vector_size=size,
                    min_count=min_count,
                    window=window,
                    negative=negative,
                    sg=cbow,
                    epochs=iterations,
                    seed=seed,
                    workers=workers)
