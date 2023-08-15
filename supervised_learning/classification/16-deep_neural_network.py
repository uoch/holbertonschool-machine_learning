#!/usr/bin/env python3
"""neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network with multiple layers """

    def __init__(self, nx, layers):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # Number of layers
        self.cache = {}  # Store intermediate values during forward propagation
        self.weights = {}  # Store weight matrices
        self.biases = {}  # Store bias vectors

        # Initialize weights and biases using He initialization
        for l, layer_size in enumerate(layers, start=1):
            if not isinstance(layer_size, int):
                raise TypeError("layers must be a list of positive integers")
            else:
                self.weights[f"W{l}"] = np.random.randn(
                    layer_size, nx if l == 1 else layers[l - 2]) * np.sqrt(2 / (nx if l == 1 else layers[l - 2]))
                self.biases[f"b{l}"] = np.zeros((layer_size, 1))
