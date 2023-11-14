#!/usr/bin/env python3
"""clustering"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    init = np.random.uniform(min, max, (k, d))
    return init
