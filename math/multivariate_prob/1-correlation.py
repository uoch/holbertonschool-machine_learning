#!/usr/bin/env python3
"""multivariate normal distribution"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    # Calculate the standard deviation matrix
    corr = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            corr[i][j] = C[i][j] / np.sqrt(C[i][i] * C[j][j])
    # diag = np.sqrt(np.diag(C))
    # std = np.outer(diag, diag)
    # corr = C / std
    return corr
