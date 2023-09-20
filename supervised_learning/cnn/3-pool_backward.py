#!/usr/bin/env python3
"""backward propagation over a pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """pooling layer backward prop"""
    m, h_new, w_new, c = dA.shape
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c):
                    x = j*sh
                    y = k*sw
                    zoom = A_prev[i, x:x+kh, y:y+kw, l]
                    if mode == 'max':
                        dA_prev[i, x:x+kh, y:y+kw, l] += np.where(
                            zoom == np.max(zoom), dA[i, j, k, l], 0)
                    elif mode == 'avg':
                        dA_prev[i, x:x+kh, y:y+kw, l] += dA[i, j, k, l] / (
                            kh * kw)
    return dA_prev
