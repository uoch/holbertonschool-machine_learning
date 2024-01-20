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

        self.__weights = {}  # Store weight matrices

        # Initialize weights and biases using He initialization
        for li, layer_size in enumerate(layers):
            if not isinstance(layer_size, int) or layer_size < 1:
                raise TypeError("layers must be a list of positive integers")
            else:
                if li == 0:
                    self.__weights['W' + str(li+1)] = np.random.randn(
                        layers[li], nx) * np.sqrt(2/nx)
                else:
                    self.weights["W"+str(li+1)] = np.random.randn(
                        layers[li], layers[li-1]) * np.sqrt(2/layers[li-1])
                self.__weights['b'+str(li+1)] = np.zeros((layer_size, 1))

        self.__L = len(layers)  # Number of layers
        self.__cache = {}

    @property
    def weights(self):
        return self.__weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache
