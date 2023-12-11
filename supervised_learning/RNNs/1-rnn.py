#!/usr/bin/env python3
"""Rucurrent Neural Network"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """forward prop for simple RNN"""
    t, m, i = X.shape
    _, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y
    return H, Y