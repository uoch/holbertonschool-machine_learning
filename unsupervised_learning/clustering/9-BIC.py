#!/usr/bin/env python3
"""clustering"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC"""
    if kmax is None:
        kmax = X.shape[0]
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if type(kmin) is not int or kmin != int(kmin) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if type(kmax) is not int or kmax != int(kmax) or kmax < 1:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations != int(iterations) or iterations < 1:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    n, d = X.shape
    lik = np.zeros((kmax - kmin + 1,))
    b = np.zeros((kmax - kmin + 1,))
    result_list = []

    best_bic = np.inf
    best_k = None
    best_result = None

    for k in range(kmin, kmax + 1):
        pi, m, S, _, lk = expectation_maximization(
            X, k, iterations, tol, verbose)
        lik[k - kmin] = lk
        p = k * (d + 2) * (d + 1) / 2 - 1
        bic = p * np.log(n) - 2 * lk
        b[k - kmin] = bic
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

        result_list.append((pi, m, S))

    return best_k, best_result, lik, b
