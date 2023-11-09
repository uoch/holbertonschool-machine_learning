#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def Q_affinities(Y):
    """calculates the Q affinities"""
    n, ndim = Y.shape
    Q = np.zeros((n, n))
    # pairwise distance in vectorized form
    diff = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    D = np.sum(np.square(diff), axis=2)
    num = (1 + D) ** (-1)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
