#!/usr/bin/env python3
"""regularization"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """gradient_descent with dropout
    the probability shoud be applied to dz"""
    m = Y.shape[1]
    cache['D0'] = np.ones((cache['A0'].shape[0], m))
    for i in range(L, 0, -1):
        if i == L:
            dz = cache['A' + str(L)] - Y
        ap = cache['A' + str(i - 1)] * cache['D' + str(i - 1)]
        dw = np.matmul(dz, ap.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz = ((np.matmul(weights['W' + str(i)].T, dz)
              * (1 - ap ** 2))*cache['D' + str(i - 1)])/keep_prob
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
    return weights
