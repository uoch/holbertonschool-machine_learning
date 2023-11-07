#!/usr/bin/env python3
"""bayesian prob"""
import numpy as np


def intersection(x, n, P, Pr):
    """ calculates the intersection of P and Pr
    x = num of patients that develop severe side effects
    n = total num of patients observed
    P = 1D np.ndarray containing the various hypothetical probabilities
    of developing severe side effects
    Pr = 1D np.ndarray containing the prior beliefs of P
    p.shape = (n,) where n is the number of patients"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError(f'All values in {P} must be in the range [0, 1]')
    if np.isclose(np.sum(Pr), 1) is False:
        raise ValueError('Pr must sum to 1')
    # pmf is the probability mass function of a binomial dist for n and x
    # likelihood = pmf * (P ** x) * ((1 - P) ** (n - x))
    # intersection = likelihood * Pr
    pmf = np.math.factorial(n) / (np.math.factorial(x)
                                  * np.math.factorial(n - x))
    likelihood = pmf * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    return intersection
