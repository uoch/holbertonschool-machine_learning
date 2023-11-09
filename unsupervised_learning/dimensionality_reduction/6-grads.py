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

    dY = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        # Use element-wise multiplication and sum along the appropriate axis
        dY[i] = np.sum((PQ * num)[:, i, np.newaxis] * Y_diff[:, i, :], axis=0)

    return -dY, Q
