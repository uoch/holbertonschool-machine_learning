#!/usr/bin/env python3
"""regularization"""
import numpy as np


def sigmoid(z):
    a = 1.0/(1.0+np.exp(-z))
    return a


def tanh(z):
    a = 2 * sigmoid(2 * z) - 1
    return a


def softmax(z):
    z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    a = z_exp / np.sum(z_exp, axis=0, keepdims=True)
    return a


def dropout_forward_prop(X, weights, L, keep_prob):
    """frwd prop with dropout"""
    # random.binomial(n, p, size=None)
    a = {}
    a['A0'] = X
    li = []
    for i in range(L):
        z = np.matmul(weights['W' + str(i + 1)], a['A' + str(i)])\
            + weights['b' + str(i + 1)]
        a['D' + str(i + 1)] = np.random.binomial(1, keep_prob, size=z.shape)
        if i == L - 1:
            a['A' + str(i + 1)] = softmax(z) * a['D' + str(i + 1)]
        else:
            a['A' + str(i + 1)] = tanh(z) * a['D' + str(i + 1)]
    return a
