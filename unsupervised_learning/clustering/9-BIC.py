#!/usr/bin/env python3
"""clustering"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC"""
    if kmax is None:
        kmax = X.shape[0]

    n, d = X.shape
    lik_list = []
    bic_list = []
    result_list = []

    best_bic = np.inf
    best_k = None
    best_result = None

    for k in range(kmin, kmax + 1):
        try:
            pi, m, S, _, lk = expectation_maximization(
                X, k, iterations, tol, verbose)
            lik_list.append(lk)
            p = k * (d + d * (d + 1) // 2 + 1)
            bic = p * np.log(n) - 2 * lk
            bic_list.append(bic)

            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_result = (pi, m, S)

            result_list.append((pi, m, S))

        except Exception as e:
            return None, None, None, None

    l = np.array(lik_list)
    b = np.array(bic_list)

    return best_k, best_result, l, b
