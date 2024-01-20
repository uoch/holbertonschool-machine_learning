#!/usr/bin/env python3
"""dimensionality reduction"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    X ndarray (n, d) dataset to be transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    ndims is the new dimensional representation of X
    idims is the intermediate dimensional representation of X after PCA
    perplexity is the perplexity
    iterations is the number of iterations
    lr is the learning rate
    """
    n, d = X.shape
    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)
    Y = np.random.randn(n, ndims)
    for i in range(iterations):
        dY, Q = grads(Y, P)
        if i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))
        Y -= lr * dY
    return Y
