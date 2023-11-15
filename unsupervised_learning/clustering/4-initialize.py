#!/usr/bin/env python3
"""clustering"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None
    m, clusters = kmeans(X, k)
    pi = np.ones(shape=(k,))/k
    d = X.shape[1]
    s = np.identity(d)[np.newaxis, :, :]
    s = np.tile(s, (k, 1, 1))
    return pi, m, s
