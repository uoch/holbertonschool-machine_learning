#!/usr/bin/env python3
"""backward prop over a conv layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """backward prop over a conv layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h_prev-1)*sh+kh-h_prev)/2)
        pw = int(((w_prev-1)*sw+kw-w_prev)/2)
    elif padding == 'valid':
        ph, pw = 0, 0
    elif type(padding) == tuple:
        ph, pw = padding
    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_prev = np.zeros(padded.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c_new):
                    dA_prev[i, j*sh:j*sh+kh, k*sw:k*sw+kw,
                            :] += W[:, :, :, l] * dZ[i, j, k, l]
                    dW[:, :, :, l] += padded[i, j*sh:j*sh +
                                             kh, k*sw:k*sw+kw, :] * dZ[i, j, k, l]
                    db[:, :, :, l] += dZ[i, j, k, l]
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dW, db
