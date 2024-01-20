#!/usr/bin/env python3
"""Rucurrent Neural Network"""
import numpy as np


def sigmoid(x):
    """sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax activation function"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class GRUCell:
    """Gated Recurrent Unit"""

    def __init__(self, i, h, o):
        """constructor for simple rnn cell
        i is dimensionality of the data
        h is dimensionality of hidden state
        o is dimensionality of outputs"""
        self.Wz = np.random.randn(h + i, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(h + i, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward prop"""
        ht = np.concatenate((h_prev, x_t), axis=1)
        # rt = reset gate, zt = update gate
        rt = sigmoid(np.matmul(ht, self.Wr) + self.br)
        zt = sigmoid(np.matmul(ht, self.Wz) + self.bz)
        tanh_input = np.concatenate((rt * h_prev, x_t), axis=1)
        # ht_tilde = candidate hidden state
        ht_tilde = np.tanh(np.matmul(tanh_input, self.Wh) + self.bh)
        zt_ = (1 - zt)
        htx = h_prev * zt_
        ht_mat = zt * ht_tilde
        h_next = ht_mat + htx
        yt = np.matmul(h_next, self.Wy) + self.by
        y = softmax(yt)
        return h_next, y
