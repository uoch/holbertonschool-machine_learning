#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset
    X ndarray (n, d) dataset
    var fraction of variance that PCA transformation should maintain"""
    n,d = X.shape
    u, s, vh = np.linalg.svd(X)
    weights_matrix = vh.T
    # determine number of components to keep
    S_norm = s/np.sum(s)
    k = 0
    r = 0
    for i in range(len(s)):
        k += S_norm[i]
        r += 1
        if k >= var:
            r+=1
            break
    return weights_matrix[:, :r+1]