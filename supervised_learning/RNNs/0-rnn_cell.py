#!/usr/bin/env python3
"""Rucurrent Neural Network"""
import numpy as np


def softmax(x):
    """softmax activation function"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class RNNCell:
    """Rnn cell class"""

    def __init__(self, i, h, o):
        """constructor for simple rnn cell"""
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward prop"""
        ht = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(ht, self.Wh) + self.bh)
        yt = np.matmul(h_next, self.Wy) + self.by
        y = softmax(yt)
        return h_next, y
