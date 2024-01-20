#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """calculates the gradients of Y"""
    Q, num = Q_affinities(Y)
    # (pi j − qi j )
    PQ = P - Q
    # (yi − yj)
    Y_diff = Y[:, np.newaxis, :] - Y
    dY = -np.sum((PQ * num)[:, :, np.newaxis] * Y_diff[:, :, :], axis=0)

    return dY, Q
