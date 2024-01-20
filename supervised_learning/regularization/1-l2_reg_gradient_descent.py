#!/usr/bin/env python3
"""regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """gradient descent with L2 regularization"""
    m = Y.shape[1]
    for i in range(L, 0, -1):
        if i == L:
            dz = cache['A' + str(L)] - Y
        ap = cache['A' + str(i - 1)]
        dw = np.matmul(dz, ap.T) / m + lambtha / m * weights['W' + str(i)]
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz = np.matmul(weights['W' + str(i)].T, dz) * (1 - ap ** 2)
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
    return weights
