#!/usr/bin/env python3
"""Normalization Constants"""""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Z is a numpy.ndarray of shape (m, n) that should be normalized
    m is the number of data points
    n is the number of features in Z
    gamma is a numpy.ndarray of shape (1, n) containing the scales used for
        batch normalization
    beta is a numpy.ndarray of shape (1, n) containing the offsets used for
        batch normalization
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix
    """
    xx = 1/Z.shape[0]*np.sum(Z, axis=0)
    Vsq = 1/Z.shape[0]*np.sum((Z-xx)**2, axis=0)
    Z_norm = (Z-xx)/np.sqrt(Vsq+epsilon)
    Z_tilde = gamma*Z_norm+beta
    return Z_tilde
