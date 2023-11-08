#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset
    X ndarray (n, d) dataset
    ndim new dimensionality of the transformed X"""
    n, d = X.shape
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    weights_matrix = vh.T
    T = X_m @ weights_matrix
    return T[:, :ndim]
