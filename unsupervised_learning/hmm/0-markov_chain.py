#!/usr/bin/env python3
"""markov hidden model"""
import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain being in a particular"""
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if type(t) is not int or t < 0:
        return None
    if P.shape[0] != P.shape[1] or s.shape[1] != P.shape[0]:
        return None
    p_pow = np.linalg.matrix_power(P, t)
    prob = np.matmul(s, p_pow)
    return prob
