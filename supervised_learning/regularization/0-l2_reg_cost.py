#!/usr/bin/env python3
import numpy as np
"""regularization"""


def l2_reg_cost(cost, lambtha, weights, L, m):
    """regularization"""
    x = 0
    for i in range(L):
        w = weights['W' + str(i + 1)]
        x += np.sum(np.square(w))
    cos = cost + lambtha / (2 * m) * x
    return cos
