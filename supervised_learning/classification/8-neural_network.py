#!/usr/bin/env python3
"""neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """neural network with single hidden layer"""

    def __init__(self, nx, nodes):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(size=(nodes, nx))  # size =(out,in)
        self.b1 = 0
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))  # size =(out,in)
        self.b2 = 0
        self.A2 = 0
