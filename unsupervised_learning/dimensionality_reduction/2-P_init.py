#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def P_init(X, perplexity):
    """
    X ndarray (n, d) dataset to be transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point"""
    # Pairwise distance
    n, d = X.shape
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = np.linalg.norm(X[i] - X[j])
    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    H = np.log2(perplexity)
    return (D, P, betas, H)
