#!/usr/bin/env python3
"""multivariate normal distribution"""
import numpy as np


def mean_cov(X):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    X = X - mean
    cov = np.matmul(X.T, X) / (n - 1)
    return mean, cov
