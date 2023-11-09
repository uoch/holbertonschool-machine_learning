#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """calculates the symmetric P affinities of a data set"""
    D, P, betas, H = P_init(X, perplexity)
    n, d = X.shape
    for i in range(n):
        low, high = None, None
        # delete current point
        Di = np.delete(D[i], i, axis=0)
        Hi, Pi = HP(Di, betas[i])
        # binary search
        H_diff = Hi - H
        while np.abs(H_diff > tol):
            if H_diff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i, 0] = betas[i, 0] * 2
                else:
                    betas[i, 0] = (betas[i, 0] + high) / 2
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i, 0] = betas[i, 0] / 2
                else:
                    betas[i, 0] = (betas[i, 0] + low) / 2
            # update Hi and Pi for the current point
            Hi, Pi = HP(Di, betas[i])
            H_diff = Hi - H
        # update P
        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]
    return (P.T+P)/(2*n)
