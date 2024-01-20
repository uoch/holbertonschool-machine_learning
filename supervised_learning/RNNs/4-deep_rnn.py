#!/usr/bin/env python3
"""Rucurrent Neural Network"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """frwd prop for deep RNN"""
    t, m, i = X.shape
    l, _, h = h_0.shape
    Y = []
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    for step in range(t):
        for layer in range(l):
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        Y.append(y)
    Y = np.array(Y)
    return H, Y
