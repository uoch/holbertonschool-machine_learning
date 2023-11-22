#!/usr/bin/env python3
"""markov hidden model"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.any(P <= 0):
        return None
    evals, evecs = np.linalg.eig(P.T)
    # get eigenvalues equal to 1
    index = np.where(np.abs(evals - 1) < 1e-8)[0]
    # get the eigenvector
    evecs = evecs[:, index]
    steady = evecs / evecs.sum()
    return steady
