#!/usr/bin/env python3
"""clustering"""
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        C_copy = np.copy(C)

        distances = np.linalg.norm(X[:, None] - C_copy, axis=-1)
        clusters = np.argmin(distances, axis=-1)

        for j in range(k):
            # if no data points assigned to cluster j, leave it unchanged
            if X[clusters == j].size == 0:
                C[j] = initialize(X, 1)
            else:
                # else, mean of all data points assigned to cluster j
                C[j] = np.mean(X[clusters == j], axis=0)

        if np.array_equal(C, C_copy):
            break

    return C, clusters
