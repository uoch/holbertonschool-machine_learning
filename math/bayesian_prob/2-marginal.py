#!/usr/bin/env python3
"""Intersection Module"""
import numpy as np


def marginal(x, n, P, Pr):
    """calculates the denominator of the bayesian probability"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError(f"All values in {P} must be in the range [0, 1]")
    if np.isclose(np.sum(Pr), 1) is False:
        raise ValueError("Pr must sum to 1")
    pmf = np.math.factorial(n) / (np.math.factorial(x)
                                  * np.math.factorial(n - x))
    likelihood = pmf * (P ** x) * ((1 - P) ** (n - x))
    return np.sum(likelihood * Pr)
