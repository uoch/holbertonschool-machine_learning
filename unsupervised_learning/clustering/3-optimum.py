#!/usr/bin/env python3
"""clustering"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None

    if kmin <= 0 or kmax <= 0 or kmax <= kmin:
        return None, None

    vp = 0.0
    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        c, clusters = kmeans(X, k, iterations)
        v = variance(X, c)
        x = vp - v
        results.append((c, clusters))
        if x <= 0:
            vp = v
            x = 0.0

        d_vars.append(x)

    # Convert the NumPy array to a Python list
    d_vars = np.array(d_vars).tolist()

    return results, d_vars
