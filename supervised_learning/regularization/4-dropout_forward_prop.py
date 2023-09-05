#!/usr/bin/env python3
"""regularization"""
import numpy as np


def sigmoid(z):
    """sigmoid activation function"""
    a = 1.0/(1.0+np.exp(-z))
    return a


def tanh(z):
    """tanh activation function"""
    a = 2 * sigmoid(2 * z) - 1
    return a


def softmax(z):
    """softmax activation function"""
    z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    a = z_exp / np.sum(z_exp, axis=0, keepdims=True)
    return a


def dropout_forward_prop(X, weights, L, keep_prob):
    """frwd prop with dropout"""
    # random.binomial(n, p, size=None)
    a = {}
    a['A0'] = X
    for i in range(L):
        z = np.matmul(weights['W' + str(i + 1)], a['A' + str(i)])\
            + weights['b' + str(i + 1)]
        if i != L - 1:
            a['D' + str(i + 1)] = np.random.binomial(1, keep_prob, size=z.shape)
        if i == L - 1:
            a['A' + str(i + 1)] = softmax(z)
        else:
            a['A' + str(i + 1)] = (tanh(z) * a['D' + str(i + 1)])/keep_prob
    return a
