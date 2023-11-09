#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def Q_affinities(Y):
    """calculates the Q affinities"""
    n, ndim = Y.shape
    Q = np.zeros((n, n))
    # pairwise distance
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = (np.linalg.norm(Y[i] - Y[j]))**2
    num = (1 + D)**(-1)
    den = np.sum(num)
    Q = num / den
    return Q, num
