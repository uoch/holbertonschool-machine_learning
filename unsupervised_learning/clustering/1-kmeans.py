#!/usr/bin/env python3
"""clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    C = np.random.uniform(np.amin(X, axis=0),
                          np.amax(X, axis=0), (k, X.shape[1]))
    if C is None:
        return None, None

    for _ in range(iterations):
        # Calculate distances between data points and centroids
        distances = np.linalg.norm(X[:, None] - centroids, axis=-1)
        clusters = np.argmin(distances, axis=-1)

        new_centroids = np.array([X[clusters == j].mean(axis=0)
                                  if np.any(clusters == j)
                                  else np.random.uniform(
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            (X.shape[1],)) for j in range(k)])

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters
