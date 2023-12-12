#!/usr/bin/env python3
"""Rucurrent Neural Network"""
import numpy as np


def sigmoid(x):
    """sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax activation function"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def rnn_layer(x, h):
    return np.matmul(x, h)


class LSTMCell:
    """Long Short Term Memory"""

    def __init__(self, i, h, o):
        """constructor for simple rnn cell
        i is dimensionality of the data
        h is dimensionality of hidden state
        o is dimensionality of outputs
        f is forget gate
        u is update gate
        c is intermediate cell state"""
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """forward prop in lstm 1 time step"""
        join = np.concatenate((h_prev, x_t), axis=1)
        ft_layer = rnn_layer(join, self.Wf) + self.bf
        ft = sigmoid(ft_layer)
        ut_layer = rnn_layer(join, self.Wu) + self.bu
        ut = sigmoid(ut_layer)
        ct_layer = rnn_layer(join, self.Wc) + self.bc
        ct = np.tanh(ct_layer)
        ct_next = ft * c_prev
        ct_next += ut * ct
        ot_layer = rnn_layer(join, self.Wo) + self.bo
        ot = sigmoid(ot_layer)
        ht = ot * np.tanh(ct_next)
        yt = rnn_layer(ht, self.Wy) + self.by
        y = softmax(yt)
        return ht, ct_next, y
