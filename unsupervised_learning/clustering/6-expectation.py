#!/usr/bin/env python3
"""clustering"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM
    X ndarray (n, d) data set
    pi ndarray (k,) priors for each cluster
    m ndarray (k, d) centroid means for each cluster
    S ndarray (k, d, d) covariance matrices for each cluster
    Returns: g, l, or None, None on failure
        g ndarray (k, n) posterior probs for each data point in each cluster
        l floa log likelihood of the model
        p = pdf(X, m, S) * pi/np.sum(pdf(X, m, S) * pi, axis=0)"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    g = np.zeros((m.shape[0], X.shape[0]))
    for i in range(m.shape[0]):
        g[i] = pdf(X, m[i], S[i]) * pi[i]
    g_sum = np.sum(g, axis=0)
    g /= g_sum
    l = np.sum(np.log(g_sum))
    return g, l
