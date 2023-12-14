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
    """builds a recurrent neural network layer"""
    return np.matmul(x, h)


class BidirectionalCell:
    """bidirectional cell"""

    def __init__(self, i, h, o):
        self.Whf = np.random.normal(size=(h + i, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.bhb = np.zeros((1, h))
        # output size is 2h because it is the concat of forward and backward
        self.Wy = np.random.normal(size=(2*h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward prop"""
        ht = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(ht, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """backward prop"""
        ht = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(ht, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        t, batch, _ = H.shape
        y = np.zeros((t, batch, self.by.shape[1]))
