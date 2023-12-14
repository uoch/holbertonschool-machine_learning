#!/usr/bin/env python3
"""Rucurrent Neural Network"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """builds a bidirectional RNN"""
    t, m, i = X.shape
    _, h = h_0.shape
    # H shape should be (t, m, 2h)
    hf = np.zeros((t + 1, m, h))
    hb = np.zeros((t + 1, m, h))
    H = np.zeros((t, m, 2*h))

    hf[0] = h_0
    hb[-1] = h_t

    for step in range(t):
        hf[step + 1] = bi_cell.forward(hf[step], X[step, :, :])
    for step in range(t - 1, -1, -1):
        hb[step] = bi_cell.backward(hb[step + 1], X[step, :, :])

    H = np.concatenate((hf[1:], hb[:-1]), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
